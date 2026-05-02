"""Evidently drift scheduler for the TFG MLOps PoC.

Runs inside the `evidently-scheduler` Compose service. Polls
`prediction-log-postgres` on a configurable interval, splits rows into a
reference window (older) and a current window (recent), builds an
Evidently `Report` with the data-drift preset, and writes the resulting
snapshot into the same workspace volume the `evidently-ui` service
serves over port 15001.

When there isn't enough data on either side of the split, the loop just
sleeps and tries again on the next tick — so the first runs after a
fresh `make mlops-up` are quietly idempotent until traffic builds up.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import psycopg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("evidently-scheduler")

# Numeric feature columns Evidently will check for drift. Must match
# observability/sql/prediction-log-schema.sql.
NUMERIC_FEATURES = [
    "image_width",
    "image_height",
    "image_aspect_ratio",
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_brightness",
    "confidence_threshold",
    "box_count",
    "inference_latency_ms",
]

DSN = os.environ["PREDICTION_LOG_DSN"]
WORKSPACE_PATH = os.environ.get("EVIDENTLY_WORKSPACE", "/app/workspace")
PROJECT_NAME = os.environ.get("EVIDENTLY_PROJECT_NAME", "tfg-polyp-detection")
REF_WINDOW_HOURS = int(os.environ.get("EVIDENTLY_REFERENCE_WINDOW_HOURS", "72"))
CUR_WINDOW_HOURS = int(os.environ.get("EVIDENTLY_CURRENT_WINDOW_HOURS", "1"))
SCAN_INTERVAL = int(os.environ.get("EVIDENTLY_SCAN_INTERVAL_SECONDS", "60"))
MIN_ROWS = int(os.environ.get("EVIDENTLY_MIN_ROWS_PER_WINDOW", "5"))


def _load_evidently() -> tuple[Any, Any, Any]:
    """Locate the Report / DataDriftPreset / Workspace classes across
    Evidently versions. 0.7.x ships both legacy and future APIs;
    we prefer legacy because it has the broadest stability guarantees."""
    try:
        # Evidently 0.4+, 0.5+, 0.7+ legacy API
        from evidently.report import Report  # type: ignore
        from evidently.metric_preset import DataDriftPreset  # type: ignore
        from evidently.ui.workspace import Workspace  # type: ignore
        return Report, DataDriftPreset, Workspace
    except ImportError as exc:
        log.error("Could not import Evidently legacy API: %s", exc)
        log.error("Falling back to evidently.legacy.* shim (0.7+)")
    from evidently.legacy.report import Report  # type: ignore
    from evidently.legacy.metric_preset import DataDriftPreset  # type: ignore
    from evidently.legacy.ui.workspace import Workspace  # type: ignore
    return Report, DataDriftPreset, Workspace


def _fetch_window(
    conn: psycopg.Connection[Any],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    columns = ["created_at", *NUMERIC_FEATURES]
    cols_sql = ", ".join(columns)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {cols_sql}
            FROM predictions
            WHERE created_at >= %s AND created_at < %s
            """,
            (start, end),
        )
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def _get_or_create_project(workspace: Any) -> Any:
    for project in workspace.list_projects():
        if project.name == PROJECT_NAME:
            return project
    log.info("Creating Evidently project %r in workspace %s", PROJECT_NAME, WORKSPACE_PATH)
    return workspace.create_project(PROJECT_NAME)


def _run_once(
    conn: psycopg.Connection[Any],
    workspace: Any,
    project: Any,
    Report: Any,
    DataDriftPreset: Any,
) -> None:
    now = datetime.now(timezone.utc)
    cur_start = now - timedelta(hours=CUR_WINDOW_HOURS)
    ref_start = cur_start - timedelta(hours=REF_WINDOW_HOURS)

    cur_df = _fetch_window(conn, cur_start, now)
    ref_df = _fetch_window(conn, ref_start, cur_start)

    if len(cur_df) < MIN_ROWS or len(ref_df) < MIN_ROWS:
        log.info(
            "Skipping drift scan: not enough data (ref=%d, cur=%d, need=%d each)",
            len(ref_df),
            len(cur_df),
            MIN_ROWS,
        )
        return

    log.info(
        "Running drift report on ref=%d / cur=%d rows (windows: ref=%dh, cur=%dh)",
        len(ref_df),
        len(cur_df),
        REF_WINDOW_HOURS,
        CUR_WINDOW_HOURS,
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=ref_df[NUMERIC_FEATURES],
        current_data=cur_df[NUMERIC_FEATURES],
    )

    workspace.add_report(project.id, report)
    log.info("Drift report committed to workspace project %r", PROJECT_NAME)


def main() -> int:
    Report, DataDriftPreset, WorkspaceCls = _load_evidently()

    log.info("Opening workspace at %s", WORKSPACE_PATH)
    os.makedirs(WORKSPACE_PATH, exist_ok=True)
    workspace = WorkspaceCls.create(WORKSPACE_PATH)
    project = _get_or_create_project(workspace)

    log.info(
        "Scheduler running: scan every %ds, ref-window=%dh, cur-window=%dh, min-rows=%d",
        SCAN_INTERVAL,
        REF_WINDOW_HOURS,
        CUR_WINDOW_HOURS,
        MIN_ROWS,
    )

    stop = False

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop
        log.info("Received signal %d, stopping after current iteration", signum)
        stop = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    while not stop:
        try:
            with psycopg.connect(DSN, connect_timeout=5) as conn:
                _run_once(conn, workspace, project, Report, DataDriftPreset)
        except Exception:
            log.exception("Drift scan iteration failed; will retry next interval")

        for _ in range(SCAN_INTERVAL):
            if stop:
                break
            time.sleep(1)

    log.info("Scheduler exiting cleanly")
    return 0


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        sys.exit(main())

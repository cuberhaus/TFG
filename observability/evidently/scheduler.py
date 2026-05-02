"""Evidently drift scheduler for the TFG MLOps PoC.

Runs inside the `evidently-scheduler` Compose service. Polls
`prediction-log-postgres` on a configurable interval, splits rows into a
reference window (older) and a current window (recent), builds an
Evidently `Report` with the data-drift preset, and adds the resulting
run to the same workspace volume the `evidently-ui` service serves over
port 15001.

When there isn't enough data on either side of the split, the loop just
sleeps and tries again on the next tick — so the first runs after a
fresh `make mlops-up` are quietly idempotent until traffic builds up.

Uses the **non-legacy** Evidently 0.7 API throughout (the new UI service
can't parse legacy v1 workspace metadata; trying that path lights up
500s in the UI's `/api/projects` endpoint).
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

from evidently import Dataset, Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import RemoteWorkspace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("evidently-scheduler")

# Numeric feature columns Evidently checks for drift. Must match the
# schema in observability/sql/prediction-log-schema.sql.
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
EVIDENTLY_REMOTE_URL = os.environ.get("EVIDENTLY_REMOTE_URL", "http://evidently-ui:8000")
PROJECT_NAME = os.environ.get("EVIDENTLY_PROJECT_NAME", "tfg-polyp-detection")
REF_WINDOW_HOURS = int(os.environ.get("EVIDENTLY_REFERENCE_WINDOW_HOURS", "72"))
CUR_WINDOW_HOURS = int(os.environ.get("EVIDENTLY_CURRENT_WINDOW_HOURS", "1"))
SCAN_INTERVAL = int(os.environ.get("EVIDENTLY_SCAN_INTERVAL_SECONDS", "60"))
MIN_ROWS = int(os.environ.get("EVIDENTLY_MIN_ROWS_PER_WINDOW", "5"))


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


def _get_or_create_project(workspace: RemoteWorkspace) -> Any:
    for project in workspace.list_projects():
        if project.name == PROJECT_NAME:
            return project
    log.info("Creating Evidently project %r on %s", PROJECT_NAME, EVIDENTLY_REMOTE_URL)
    return workspace.create_project(PROJECT_NAME)


def _run_once(
    conn: psycopg.Connection[Any],
    workspace: RemoteWorkspace,
    project: Any,
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

    ref_ds = Dataset.from_pandas(ref_df[NUMERIC_FEATURES])
    cur_ds = Dataset.from_pandas(cur_df[NUMERIC_FEATURES])

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

    run_name = now.strftime("drift-%Y%m%d-%H%M%S")
    workspace.add_run(project.id, snapshot, name=run_name)
    log.info("Drift report %r committed to workspace project %r", run_name, PROJECT_NAME)


def _wait_for_ui(workspace: RemoteWorkspace, attempts: int = 30) -> None:
    """Block until the Evidently UI service responds to list_projects."""
    for attempt in range(attempts):
        try:
            workspace.list_projects()
            return
        except Exception:
            log.info("Waiting for Evidently UI at %s (attempt %d/%d)", EVIDENTLY_REMOTE_URL, attempt + 1, attempts)
            time.sleep(2)
    raise RuntimeError(f"Evidently UI at {EVIDENTLY_REMOTE_URL} not reachable after {attempts} attempts")


def main() -> int:
    log.info("Connecting to Evidently UI at %s", EVIDENTLY_REMOTE_URL)
    workspace = RemoteWorkspace(EVIDENTLY_REMOTE_URL)
    _wait_for_ui(workspace)
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
                _run_once(conn, workspace, project)
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

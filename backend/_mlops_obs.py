"""MLOps observability hook for TFG's FastAPI backend.

Mirrors the shape of `_sentry_obs.py`:

- Single env var (`MLOPS_PREDICTION_LOG_DSN`) toggles all behaviour.
  Unset → every entry point is a no-op. Set → predictions are written
  to the prediction-log Postgres started by `make mlops-up`.

- Defensive imports so missing `psycopg` (e.g. on a fresh checkout
  before `pip install -r requirements.txt`) doesn't kill the app.

- Per-request session id is read from the same `ContextVar` Sentry's
  `SessionIdMiddleware` already binds in `_sentry_obs.py` — so a single
  `X-Session-Id` from the React frontend lights up Sentry traces and
  prediction-log rows together (the cross-tier loop documented in
  `observability/INTEGRATION-NOTES.md` §5).

The actual DB write is offloaded onto a FastAPI BackgroundTask so a
slow / unavailable Postgres never blocks `/api/predict`. Exceptions are
swallowed and reported as a Sentry breadcrumb instead.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("mlops_obs")

# ── Configuration ─────────────────────────────────────────────────────────
DSN = os.environ.get("MLOPS_PREDICTION_LOG_DSN") or None
ENABLED = bool(DSN)

# Connect timeout in seconds — short on purpose: a slow Postgres is much
# worse than a missing prediction-log row.
_CONNECT_TIMEOUT_S = float(os.environ.get("MLOPS_PREDICTION_LOG_CONNECT_TIMEOUT_S", "2"))

# ── Lazy imports ──────────────────────────────────────────────────────────
try:
    import psycopg  # type: ignore[import-not-found]
except ImportError:
    psycopg = None  # type: ignore[assignment]

# Reuse the Sentry session-id contextvar so the same X-Session-Id flows
# into both layers. Optional: when _sentry_obs isn't importable (rare,
# usually means we're in a CI shim), the prediction-log row gets
# session_id=None which is still indexed.
try:
    from _sentry_obs import _SESSION_ID as _SENTRY_SESSION_ID  # type: ignore[attr-defined]
except Exception:  # pragma: no cover — safety net
    _SENTRY_SESSION_ID = None  # type: ignore[assignment]


def _crumb(category: str, message: str, **data: Any) -> None:
    """Best-effort Sentry breadcrumb emitter. No-op without Sentry."""
    try:
        from _sentry_obs import breadcrumb  # type: ignore
        breadcrumb(category, message, **data)
    except Exception:
        pass


# ── Public API ────────────────────────────────────────────────────────────
def current_session_id() -> Optional[str]:
    """Read the X-Session-Id bound for the current request. None if unbound."""
    if _SENTRY_SESSION_ID is None:
        return None
    try:
        return _SENTRY_SESSION_ID.get()
    except LookupError:
        return None


_INSERT_SQL = """
    INSERT INTO predictions (
        endpoint, session_id,
        image_width, image_height, image_aspect_ratio,
        mean_r, mean_g, mean_b, mean_brightness,
        model_arch, model_file, confidence_threshold,
        box_count, mean_confidence, max_confidence, inference_latency_ms
    ) VALUES (
        %(endpoint)s, %(session_id)s,
        %(image_width)s, %(image_height)s, %(image_aspect_ratio)s,
        %(mean_r)s, %(mean_g)s, %(mean_b)s, %(mean_brightness)s,
        %(model_arch)s, %(model_file)s, %(confidence_threshold)s,
        %(box_count)s, %(mean_confidence)s, %(max_confidence)s, %(inference_latency_ms)s
    )
"""


def _write_row(row: dict[str, Any]) -> None:
    """Background worker: actually inserts the row. Swallows everything."""
    if psycopg is None:
        return
    try:
        with psycopg.connect(DSN, connect_timeout=_CONNECT_TIMEOUT_S) as conn:
            with conn.cursor() as cur:
                cur.execute(_INSERT_SQL, row)
    except Exception as exc:
        logger.warning("MLOps prediction-log write failed: %s", exc)
        _crumb("mlops", "prediction-log write failed", error=str(exc))


def log_prediction(
    background_tasks: Any,
    *,
    endpoint: str,
    features: dict[str, Any],
    predictions: dict[str, Any],
) -> None:
    """Schedule a background write to prediction-log-postgres.

    No-op when MLOPS_PREDICTION_LOG_DSN is unset OR when psycopg isn't
    available. The write is offloaded to FastAPI's BackgroundTasks so a
    slow/unhealthy Postgres can't slow the user's response.

    Parameters
    ----------
    background_tasks
        The FastAPI BackgroundTasks instance from the request handler.
    endpoint
        Identifier for the route, e.g. ``"/api/predict"``.
    features
        Image-level scalar features to log. Keys must match the
        `predictions` schema (see ``observability/sql/prediction-log-schema.sql``).
    predictions
        Output statistics for the inference. Keys must match the
        prediction columns of the schema.
    """
    if not ENABLED:
        return

    row = {
        "endpoint": endpoint,
        "session_id": current_session_id(),
        **features,
        **predictions,
    }

    # FastAPI's BackgroundTasks runs the callable AFTER the response is
    # sent, so the user never waits on Postgres. add_task is the cheapest
    # way to do this; we don't even need an asyncio task group.
    try:
        background_tasks.add_task(_write_row, row)
    except Exception as exc:
        # Fallback: if for some reason we can't schedule, log and drop.
        logger.warning("Could not schedule MLOps log task: %s", exc)
        _crumb("mlops", "schedule failed", error=str(exc))


# ── Helper for /api/mlops/stats endpoint (read-only Postgres queries) ────
def get_stats() -> dict[str, Any]:
    """Read-only summary used by `GET /api/mlops/stats`.

    Returns ``{"enabled": False}`` when the DSN isn't set, otherwise the
    JSON shape documented in ``observability/INTEGRATION-NOTES.md`` §9.
    """
    base = {
        "enabled": ENABLED,
        "mlflow_url": os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:15000",
        "evidently_url": os.environ.get("EVIDENTLY_UI_URL") or "http://localhost:15001",
    }
    if not ENABLED or psycopg is None:
        return base

    try:
        with psycopg.connect(DSN, connect_timeout=_CONNECT_TIMEOUT_S) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT total, with_boxes, distinct_sessions, avg_latency_ms, max_latency_ms FROM predictions_today")
                row = cur.fetchone() or (0, 0, 0, 0, 0)
                cur.execute(
                    "SELECT max(created_at), count(*) FROM predictions WHERE created_at >= now() - interval '1 hour'"
                )
                last_hour = cur.fetchone() or (None, 0)
        total, with_boxes, distinct_sessions, avg_latency, max_latency = row
        return {
            **base,
            "predictions_today": int(total or 0),
            "predictions_with_boxes_today": int(with_boxes or 0),
            "distinct_sessions_today": int(distinct_sessions or 0),
            "avg_latency_ms_today": int(avg_latency or 0),
            "max_latency_ms_today": int(max_latency or 0),
            "predictions_last_hour": int(last_hour[1] or 0),
            "last_prediction_at": last_hour[0].isoformat() if last_hour[0] else None,
            # Drift status is owned by the evidently scheduler / Evidently UI
            # and is best fetched from there (no cheap query against
            # prediction-log alone). The frontend follows up with a call to
            # the Evidently UI when this stat is needed.
            "drift_status": "unknown",
        }
    except Exception as exc:
        logger.warning("MLOps stats read failed: %s", exc)
        return {**base, "error": str(exc)}

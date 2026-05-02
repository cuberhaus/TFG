# MLOps integration notes — TFG repo recon

Findings from the Phase-1 spike. These pin the assumptions that shape the
rest of the PoC; if any change later, the corresponding plan phase needs
updating.

## 1. W&B status

`wandb` is declared in `backend/requirements.txt` but only **one** of our
own files imports it:

- `code/src/j_notebooks/cycleGAN.ipynb` — exploratory notebook, not part
  of the FastAPI serving path or the three production training scripts.

Other matches under `code/tmp/pytorch-CycleGAN-and-pix2pix/` are vendored
third-party source from the pix2pix author's repo — not our code, won't
be touched.

**Implication for this PoC:** wandb stays as-is. Removing it is out of
scope (one-line follow-up if desired, see README's "things-that-bit-me"
section). MLflow is added alongside, not in place of.

## 2. Inference entry points

Exactly two:

- `@app.post("/api/predict")` → `def predict(...)` at `backend/main.py`
  L1211. Takes `UploadFile = File(...)` plus three `Form` fields.
- `@app.post("/api/predict/batch")` → `def predict_batch(...)`. Takes
  only form fields; picks 9 random images from `data/Test/Images` on
  disk (no upload).

Both are sync FastAPI handlers, both already call into the existing
`_sentry_obs._tag` / `_sentry_obs._crumb` instrumentation. Both are the
right place to add a single `_mlops_obs.log_prediction(...)` hook at the
bottom of the function (just before `return`).

## 3. Prediction logging is greenfield

No `psycopg` / `sqlalchemy` / `asyncpg` / `databases` import anywhere in
`backend/`. The FastAPI app currently has zero database dependencies.
Adding a single `psycopg[binary]` for the prediction-log writer is the
minimal increment. No connection-pool refactor needed.

## 4. Training-script layout

Flat under `code/src/`. The three production training scripts:

- `train_and_save_model.py` — single-shot baseline training.
- `optuna_train_model.py` — Optuna sweep.
- `raytune_train_model.py` — Ray Tune sweep.

Each is a standalone script with its own `if __name__ == "__main__":`
guard. No package, no shared `Trainer` class. The MLflow autolog wiring
is therefore three near-identical 2-line patches, one per file.

`code/requirements.txt` is encoded as **UTF-16-with-BOM** and contains
literal spaces between every character (`a i o s i g n a l = = 1 . 3 .
1`). It can't be `pip install`-ed in its current state. Out of scope for
this PoC. The training scripts run from a manual venv anyway; we'll add
the MLflow dep to `backend/requirements.txt` (which works) and document
the manual install in the README.

## 5. Existing Sentry session-id machinery (reuse target)

`backend/_sentry_obs.py` already implements:

- `_SESSION_ID: contextvars.ContextVar[str | None]` — a context-var
  named `_sentry_obs_session_id` that holds the per-request session id.
- `_bind_session_id(value)` / `_reset_session_id(token)` — set/clear
  helpers used by the ASGI middleware.
- `class SessionIdMiddleware` — ASGI middleware that reads the
  `X-Session-Id` request header on every incoming request, binds the
  context-var, and (currently) tags the Sentry scope with the same value.

**Implication for `_mlops_obs.py`:** import the same context-var. The
prediction-log row gets the same session id Sentry's traces are tagged
with, **with no extra plumbing on the inference endpoints**. Zero
duplication; one source of truth for the per-request session id.

The TFG frontend doesn't currently mint or send `X-Session-Id` itself —
the header is set by `PersonalPortfolio/src/lib/debug-network.ts` for
calls originating from the portfolio's React code. Inside the iframe, we
add a small frontend-side helper (`frontend/src/lib/session.ts` + axios
interceptor) so direct interactions with the TFG UI also produce a
session id. The same id then lights up Sentry traces and prediction-log
rows, plus — when the user is also on the portfolio shell — co-exists
peacefully with the portfolio's own session id (different namespace,
both valid).

## 6. BackgroundTasks already imported

`backend/main.py` L1: `from fastapi import FastAPI, File, UploadFile,
Form, BackgroundTasks, HTTPException`. Five existing endpoints already
use `BackgroundTasks` for fire-and-forget work. Wiring the prediction
logger through it is idiomatic; no new abstractions.

## 7. docker-compose.yml env propagation pattern

The existing `app` service uses `SENTRY_DSN: ${SENTRY_DSN_DOCKER:-${SENTRY_DSN:-}}`
(a chained fallback so the docker-rewrite variant from PersonalPortfolio
takes precedence, and a host-only DSN works if it isn't set). We mirror
with one new line:

```yaml
MLOPS_PREDICTION_LOG_DSN: ${MLOPS_PREDICTION_LOG_DSN:-}
```

No docker-rewrite needed — when the MLOps stack is up, the user runs
`make mlops-up` and the DSN points at `host.docker.internal:15432` (the
prediction-log-postgres host port we'll allocate in Phase 2). Setting
the env var then naturally points the FastAPI hook at the right place.
When unset, the hook no-ops.

## 8. Hook contract for `_mlops_obs.log_prediction`

```python
def log_prediction(
    *,
    features: dict[str, float | int | str | None],
    predictions: dict[str, float | int],
    session_id: str | None = None,
) -> None:
    """Insert one row into prediction-log-postgres. No-op if MLOPS_PREDICTION_LOG_DSN is unset."""
```

`features` columns we'll capture (scalar-only, per Risk #4 in the plan):

- `image_width` (px), `image_height` (px), `image_aspect_ratio` (float)
- `mean_r`, `mean_g`, `mean_b` (0-255 uint, per channel)
- `mean_brightness` (0-255 uint, computed once on the resized image)
- `model_arch` (str, e.g. `"FasterRCNN"`)
- `model_file` (str, the .pth basename — useful for model-version drift)
- `confidence_threshold` (float, the user-provided threshold)

`predictions` columns:

- `box_count` (int)
- `mean_confidence` (float, 0-1)
- `max_confidence` (float, 0-1)
- `inference_latency_ms` (int)

Plus the row scaffold:

- `id` (uuid PK)
- `created_at` (timestamptz default now)
- `session_id` (text, nullable — the `X-Session-Id` we read from the
  shared `_SESSION_ID` context-var)
- `endpoint` (text, `"/api/predict"` or `"/api/predict/batch"`)

This shape gives Evidently a dense tabular dataset for drift detection
(image-level features + prediction-level features) without ever
needing to handle the raw pixel data, which Evidently 0.5.x supports
but is less mature on.

## 9. /api/mlops/stats response shape (Phase 5)

```json
{
  "enabled": true,
  "predictions_today": 47,
  "drift_status": "ok",         // "ok" | "warning" | "critical" | "unknown"
  "latest_report_url": "http://localhost:15001/projects/.../reports/...",
  "mlflow_url": "http://localhost:15000",
  "evidently_url": "http://localhost:15001"
}
```

When `MLOPS_PREDICTION_LOG_DSN` is unset:

```json
{ "enabled": false }
```

Frontend renders a small "MLOps stack offline — `make mlops-up` to
enable" hint in that case. Same graceful-degradation pattern Sentry's
existing TFG integration uses.

## 10. Out-of-scope notes

- **`code/requirements.txt` UTF-16 fix** — a separate one-liner. Don't
  bundle here.
- **Removing wandb dep** — separate cleanup. Don't bundle here.
- **Splitting `code/src/` into a `tfg_training` package** — would
  unlock cleaner `import` of an MLflow helper module, but the
  refactor's scope dwarfs this PoC. Not in this plan.
- **Authenticated MLflow / Evidently UIs** — they're behind localhost
  during local dev. Production deploy would add auth. Out of scope.

# MLOps observability — MLflow + Evidently

Self-hosted MLOps stack for the polyp-detection PyTorch project. Adds
**experiment tracking + model registry** (MLflow) and **data /
prediction drift** (Evidently AI) alongside the existing Sentry
instrumentation. Lands on `main` as an additive layer — Sentry stays.

```mermaid
flowchart LR
  subgraph training [Training (code/src)]
    Trainer["optuna_train_model.py / train_and_save_model.py"]
  end
  subgraph serving [Serving (backend/main.py)]
    Predict["/api/predict"]
    Stats["/api/mlops/stats"]
  end
  subgraph mlflow [MLflow stack]
    MLF["mlflow-server :15000"]
    MLPG[(mlflow-postgres)]
    MLM["mlflow-minio"]
  end
  subgraph evidently [Evidently stack]
    EUI["evidently-ui :15001"]
    PLPG[(prediction-log-postgres :15432)]
    ESCH["evidently-scheduler"]
  end

  Trainer -->|"mlflow.pytorch.autolog"| MLF
  MLF --> MLPG
  MLF --> MLM
  Predict -->|"BackgroundTasks hook"| PLPG
  ESCH -->|"60s scan"| PLPG
  ESCH -->|"writes snapshots"| EUI
  Stats --> PLPG

  Sentry["Sentry SDK (existing)"] -.- Predict
  Sentry -.- Trainer
```

## Quickstart

```bash
# 1. Copy the env template and (optionally) edit
cp observability/.env.mlops.example observability/.env.mlops

# 2. Bring the stack up
make mlops-up

# 3. Open the UIs
#    MLflow    → http://localhost:15000
#    Evidently → http://localhost:15001

# 4. Enable prediction logging on the host-running FastAPI app
export MLOPS_PREDICTION_LOG_DSN=postgresql://mlops:mlops@localhost:15432/prediction_log
cd backend && ./run.sh   # (or `make backend` from the repo root)

# 5. Drive some traffic from http://localhost:5173 (the React UI), then
#    watch the scheduler logs:
make mlops-logs
```

To stop:

```bash
make mlops-down       # stops, keeps volumes (data preserved)
docker compose -f docker-compose.yml -f docker-compose.mlops.yml down -v   # nukes volumes
```

## Port map

| Service                  | Host port | Notes                                    |
|--------------------------|-----------|------------------------------------------|
| MLflow tracking UI       | 15000     | `http://localhost:15000`                 |
| Evidently UI             | 15001     | `http://localhost:15001`                 |
| prediction-log-postgres  | 15432     | DSN host for `MLOPS_PREDICTION_LOG_DSN`  |
| MLflow Postgres          | —         | internal only                            |
| MLflow MinIO             | —         | internal only (avoids Sentry/9000 clash) |
| Evidently scheduler      | —         | internal only                            |

Coexistence with the rest of the portfolio's PoC stacks is checked:
LGTM lives at 13000, Langfuse at 13100, Coroot at 18080, Sentry
self-hosted at 9000. None overlap.

## Memory budget

| Component                | Typical RAM |
|--------------------------|-------------|
| MLflow server            | ~200 MiB    |
| MLflow Postgres          | ~80 MiB     |
| MLflow MinIO             | ~150 MiB    |
| Evidently UI             | ~300 MiB    |
| Evidently scheduler      | ~150 MiB    |
| Prediction-log Postgres  | ~80 MiB     |
| **MLOps overhead total** | **~960 MiB**|

On top of the existing TFG container (~1 GiB FastAPI + cached PyTorch
model). Fits comfortably alongside Sentry self-hosted + the LGTM PoC
without swap pressure.

## Coexistence with Sentry

These stacks answer different questions:

| Question                                                 | Answered by             |
|----------------------------------------------------------|-------------------------|
| Did the FastAPI server crash on this user's upload?      | **Sentry**              |
| Which Optuna trial produced the best mAP score?          | **MLflow**              |
| Is `model_v3.pth` better than `model_v2.pth`?            | **MLflow Model Registry** |
| Have inference inputs drifted since training?            | **Evidently**           |
| Are predictions clustering differently than last week?   | **Evidently**           |

Both stacks can run together with no overlap. The same `X-Session-Id`
that Sentry's existing middleware reads from incoming requests is also
attached to every prediction-log row, so a single id pasted into Sentry
search and into the prediction-log Postgres returns rows for the same
user simultaneously.

## Optional: in-app `/api/mlops/stats` panel

When the stack is up and `MLOPS_PREDICTION_LOG_DSN` is set, the FastAPI
backend exposes `GET /api/mlops/stats` which returns:

```json
{
  "enabled": true,
  "predictions_today": 47,
  "drift_status": "ok",
  "latest_report_url": "http://localhost:15001/projects/.../reports/latest",
  "mlflow_url":   "http://localhost:15000",
  "evidently_url":"http://localhost:15001"
}
```

The React frontend's `MlopsStatusCard` polls this every 30 s and renders
the traffic-light + deep-links to the two UIs. When `MLOPS_PREDICTION_LOG_DSN`
is unset, the endpoint returns `{enabled: false}` and the card shows a
small "MLOps stack offline — `make mlops-up` to enable" hint instead.

## MLflow autologging in the training scripts

The three production training scripts gain a 2-line autolog block at
their `__main__` guard:

```python
import mlflow
mlflow.pytorch.autolog()
mlflow.set_experiment("polyp-detection-baseline")  # or -optuna-sweep, -raytune-sweep
```

When `MLFLOW_TRACKING_URI` is unset, MLflow falls back to a local
`./mlruns/` directory. Set the env var to point at the local server:

```bash
export MLFLOW_TRACKING_URI=http://localhost:15000
python code/src/optuna_train_model.py
```

Note: `code/requirements.txt` is encoded in UTF-16 with broken contents
(known issue, out of scope for this PoC). For the training side, install
MLflow into your training venv manually:

```bash
pip install mlflow==2.18.0
```

The backend side adds `mlflow` to `backend/requirements.txt` cleanly.

## Things that bit me along the way

- **Evidently 0.7's `LocalStorageComponent` autorefresh hits an
  inotify-instance ceiling.** The default config wires a watchdog
  observer onto the workspace directory, which consumes one of the
  host's `fs.inotify.max_user_instances` (default 128 on Linux). If
  Coroot, LGTM, Sentry-self-hosted, and Langfuse are also running, that
  ceiling is already exhausted and Evidently's UI 500s on every
  `/api/projects` call. Fix: `EVIDENTLY_STORAGE__TYPE=local` +
  `EVIDENTLY_STORAGE__AUTOREFRESH=false` (already in the Compose file).
  Symptom without the fix is a bare `500 Internal Server Error` because
  the watchdog crash happens before request handling. The
  `EVIDENTLY_DEBUG=1` env var makes the trace visible in the container
  logs. The scheduler pushes via `RemoteWorkspace` over HTTP, so
  watching the workspace dir for changes adds zero value here anyway.
- **Evidently 0.7 split the legacy and modern APIs.** Reports now live
  at `evidently.Report` (root namespace), presets at `evidently.presets`,
  and the workspace at `evidently.ui.workspace.RemoteWorkspace`. The
  legacy `evidently.report.Report` + `evidently.metric_preset` paths
  exist only under `evidently.legacy.*` and write a v1 metadata format
  the new UI can't parse. The scheduler uses the modern API throughout.
- **MinIO host port clash.** Sentry self-hosted publishes 9000 on the
  host; Langfuse-MinIO and the LGTM PoC's MinIO both want it too. We
  keep MLflow's MinIO entirely internal — MLflow-server reaches it via
  the Compose network. If you ever need to inspect the bucket, exec in:
  `docker compose -f docker-compose.mlops.yml exec mlflow-minio
  mc ls local/mlflow-artifacts`.
- **`mlflow server` doesn't ship `psycopg2` or `boto3`.** The official
  image `ghcr.io/mlflow/mlflow:v2.18.0` is intentionally minimal, so the
  server command does a one-shot `pip install` of `psycopg2-binary` and
  `boto3` before starting. First boot is ~30 s slower; subsequent boots
  cache and start fast.
- **Prediction logger inside the FastAPI Docker container.** When the
  app runs via `make docker-up` (TFG-side or PersonalPortfolio-orchestrated),
  it can't reach `localhost:15432` — the host port lives on the host.
  Use `host.docker.internal:15432` instead:
  ```bash
  export MLOPS_PREDICTION_LOG_DSN=postgresql://mlops:mlops@host.docker.internal:15432/prediction_log
  ```
  `host.docker.internal` is already wired into the `app` service's
  `extra_hosts` for the same reason as Sentry.
- **W&B is still in `backend/requirements.txt`.** Used only by one
  notebook (`code/src/j_notebooks/cycleGAN.ipynb`). Removing it is a
  separate one-line cleanup; the MLflow PoC doesn't bundle that.

## What's not covered

- No production deploy story (auth, TLS, multi-tenant isolation). The
  stack is local-dev only.
- No automated reference-snapshot capture. The Evidently scheduler uses
  a rolling reference window (default 72h before the current 1h
  window). If you want a frozen baseline, use the MLflow API to upload
  a golden reference dataset — out of scope for this PoC.
- No Sentry alert integration (e.g. raise an issue when drift goes
  critical). Easy follow-up: have the scheduler `POST` to Sentry's
  `events` endpoint when a drift score crosses a threshold.

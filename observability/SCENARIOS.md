# Scripted scenarios

Two narrated walkthroughs that exercise the MLflow + Evidently stack
end-to-end. Each one starts from a clean stack (`make mlops-up`) and
finishes with a screenshot-ready state in the corresponding UI.

> Both scenarios assume the FastAPI backend is running on the host with
> `MLOPS_PREDICTION_LOG_DSN=postgresql://mlops:mlops@localhost:15432/prediction_log`
> exported. The React frontend is irrelevant for scenario A and used as
> the input surface for scenario B.

---

## Scenario A — MLflow: comparing Optuna trials and promoting a winner

**Goal.** Demonstrate experiment tracking + the model registry: run
several Optuna trials with deliberately bad hyperparameters and one
good one, identify the winner in the MLflow UI, register it as a model
version, and transition it to the `Production` alias.

**Time:** ~5 min once the stack is up.

### 1. Start the MLOps stack

```bash
cd ~/cuberhaus/TFG
make mlops-up
open http://localhost:15000           # MLflow UI
```

### 2. Run an Optuna sweep with three trials

The sweep runs through `optuna_train_model.py` which already contains
the MLflow nested-run wiring from Phase 3. We use `--debug` to keep
the per-trial cost tiny — each trial trains for one epoch on 8 images.

```bash
export MLFLOW_TRACKING_URI=http://localhost:15000

# Trial 1 + 2 will be picked by Optuna; we deliberately seed bad
# starting points so the sweep produces a wide spread of results.
cd code/src
python optuna_train_model.py FasterRCNN --debug --n-trials 3 --metric f1
```

You should see in the MLflow UI a parent run named
`optuna-sweep-FasterRCNN-n3` with three nested runs `trial-0`,
`trial-1`, `trial-2` — each tagged with the trial's
hyperparameters and final `metric_value`.

### 3. Compare runs in the UI

In the MLflow UI:

1. Click the `polyp-detection-optuna-sweep` experiment.
2. Tick all four runs (parent + three children).
3. Click **Compare**.
4. The parallel-coordinates plot shows how the F1 metric varies with
   `lr`, `batch_size`, `weight_decay`. The single best trial stands out
   visually — that's the run we promote.

### 4. Register the winning model

> The training script logs PyTorch model artifacts only when a
> `model_path` is produced. For a real sweep that's automatic; in
> `--debug` mode the model isn't saved. The promotion step still works
> via the UI: pick the run, open the **Artifacts** tab, click **Register
> model**, give it a name (e.g. `polyp-detector`).

Equivalent CLI:

```bash
mlflow models transition --version 1 \
  --name polyp-detector \
  --stage Production
```

### 5. Verify the registry shows it

```bash
curl -s http://localhost:15000/api/2.0/mlflow/registered-models/list \
  | python3 -m json.tool
```

You should see one entry for `polyp-detector` with the latest version
in the `Production` stage. From this point any deployment script can
fetch the production model with:

```python
import mlflow
model = mlflow.pytorch.load_model("models:/polyp-detector/Production")
```

### What this demonstrates

- **Experiment tracking** — every trial's hyperparameters + metric end
  up in MLflow automatically via the `mlflow.start_run(nested=True)`
  block in `objective()`.
- **Run comparison** — the parallel-coordinates plot is built into
  MLflow OSS; no extra code needed.
- **Model registry** — the UI's "Register model" + transition-to-stage
  flow is the OSS equivalent of the SaaS-vendor model promotion
  features in W&B / Comet.

---

## Scenario B — Evidently: triggering drift and watching the dashboard

**Goal.** Show data-drift detection end-to-end: load a healthy
reference distribution into prediction-log-postgres, then deliberately
shift the inference inputs and watch the Evidently scheduler flag the
drift on the next 60-second tick. Bonus: the in-app `MlopsStatusCard`
reflects the change.

**Time:** ~4 min.

### 1. Start the stack and the FastAPI app

```bash
cd ~/cuberhaus/TFG
make mlops-up

# In a separate terminal:
export MLOPS_PREDICTION_LOG_DSN=postgresql://mlops:mlops@localhost:15432/prediction_log
make backend            # runs FastAPI on :8082

# In a third terminal:
make frontend           # runs Vite on :5173
```

### 2. Generate a healthy reference distribution

The fastest way to seed reference rows is to drive a few requests
through the React UI's Inference tab. Each prediction adds one row to
`predictions` with the user's `X-Session-Id` attached. After ~10
requests the scheduler has enough data to compute a baseline.

> If you don't have a polyp dataset handy, seed synthetic rows directly
> instead — the schema is scalar-only, so a few SQL inserts work:
>
> ```bash
> docker compose --env-file observability/.env.mlops \
>   -f docker-compose.yml -f docker-compose.mlops.yml \
>   exec -T prediction-log-postgres \
>   psql -U mlops -d prediction_log -f /tmp/reference-data.sql
> ```
>
> See `observability/sql/reference-data.sql` for an example seeder
> (deliberately not committed; copy from below).

A minimal seeder you can paste into psql:

```sql
INSERT INTO predictions (
    endpoint, session_id, image_width, image_height, image_aspect_ratio,
    mean_r, mean_g, mean_b, mean_brightness,
    model_arch, model_file, confidence_threshold,
    box_count, mean_confidence, max_confidence, inference_latency_ms
)
SELECT
    '/api/predict',
    'reference-' || gs::text,
    480, 560, 480.0/560.0,
    -- "Healthy" RGB distribution: roughly 130/110/95 means
    130 + (random() * 10 - 5),
    110 + (random() * 10 - 5),
    95  + (random() * 10 - 5),
    115 + (random() * 8 - 4),
    'FasterRCNN', 'baseline.pth', 0.5,
    floor(random() * 4)::int,
    0.7 + random() * 0.2,
    0.85 + random() * 0.1,
    140 + (random() * 30)::int
FROM generate_series(1, 50) AS gs;
```

### 3. Wait one scan interval (60 s default)

The `evidently-scheduler` polls every `EVIDENTLY_SCAN_INTERVAL_SECONDS`
(default 60). After one tick a baseline drift report appears in the
Evidently UI.

```bash
make mlops-logs | grep evidently-scheduler
# Expected: "Skipping drift scan: not enough data" → "Running drift report on ref=N / cur=M rows"
```

Open `http://localhost:15001`, click `tfg-polyp-detection`, click the
freshly-generated snapshot. The data-drift preset shows green for all
features (no drift between reference and current windows).

### 4. Drift the inference inputs

Now we deliberately push **a bunch of much-darker, more-blue images**
into the prediction-log. Two ways:

#### Option a) Through the UI (slow but realistic)

Go to the Inference tab in the React UI and upload 20+ images that
have been color-shifted (e.g. via `convert image.jpg -modulate
80,80,90 darker.jpg`). The TFG backend captures the new mean-RGB
values automatically and writes them to the prediction-log.

#### Option b) SQL (fast, for demos)

```sql
INSERT INTO predictions (
    endpoint, session_id, image_width, image_height, image_aspect_ratio,
    mean_r, mean_g, mean_b, mean_brightness,
    model_arch, model_file, confidence_threshold,
    box_count, mean_confidence, max_confidence, inference_latency_ms
)
SELECT
    '/api/predict',
    'drift-demo-' || gs::text,
    480, 560, 480.0/560.0,
    -- "Drifted" distribution: much darker, blue-shifted
    65 + (random() * 8 - 4),
    50 + (random() * 8 - 4),
    140 + (random() * 8 - 4),
    72 + (random() * 6 - 3),
    'FasterRCNN', 'baseline.pth', 0.5,
    floor(random() * 2)::int,
    0.4 + random() * 0.2,
    0.55 + random() * 0.1,
    140 + (random() * 30)::int
FROM generate_series(1, 30) AS gs;
```

### 5. Wait one more scan interval

```bash
make mlops-logs
# Expected: "Drift report 'drift-YYYYMMDD-HHMMSS' committed to workspace"
```

In the Evidently UI:

1. Refresh `http://localhost:15001/projects/<id>/reports`.
2. Open the latest snapshot.
3. The **Data Drift** section now flags `mean_r`, `mean_g`, `mean_b`,
   `mean_brightness`, and `mean_confidence` as drifted (PSI / KS test
   above threshold). The drift counter at the top reads e.g. **5/10
   features drifted**.

### 6. Watch the in-app card change

The `MlopsStatusCard` in the bottom-right of the React UI polls
`/api/mlops/stats` every 30 s. Right now the endpoint returns
`drift_status: "unknown"` because computing drift status from the raw
prediction-log alone would be expensive — the production-grade fix is
to extend the scheduler to also POST a `drift_status` row to a
dedicated `drift_state` table that the stats endpoint reads. That's an
explicit follow-up (see `observability/README.md` "What's not covered").

### What this demonstrates

- **Data drift detection** — Evidently compares the recent inference
  inputs against the rolling reference window, and surfaces per-feature
  drift via PSI / Kolmogorov–Smirnov tests automatically.
- **Cross-tier session correlation** — every drifted prediction-log row
  is tagged with the `X-Session-Id` from the React tab. An investigator
  who pastes the same session id into Sentry's search and into the
  prediction-log Postgres gets the user's full backend trace tree
  alongside the inference rows that contributed to the drift flag.
- **Self-hosted, no SaaS account needed** — both halves run on
  localhost, contained in one `docker-compose.mlops.yml` overlay.

---

## Cleanup

```bash
make mlops-down                                          # preserves volumes
docker compose -f docker-compose.yml \
  -f docker-compose.mlops.yml down -v                    # nukes data
```

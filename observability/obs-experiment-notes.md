# MLflow + Evidently PoC — observation log

Fill this in as you actually use the stack. The seven sibling PoCs
(LGTM, Honeycomb, Elastic, Coroot, Embrace, Langfuse, Sentry) each have
their own version of this file with similar headings, so a `diff`
across them is the fastest comparison medium.

Headings deliberately mirror those PoCs' notes files. Cells filled with
concrete numbers below come from the Phase 1-7 implementation
verification (the implementation work is real); cells marked TBD are
reserved for hands-on UX observations once the scenarios in
[SCENARIOS.md](SCENARIOS.md) actually run end to end.

---

## First impressions

_Notes from the first 30 minutes of clicking around the MLflow and
Evidently UIs after the first runs / drift snapshots land. What stood
out, what was confusing, what was surprisingly fast or slow._

- **Time from `make mlops-up` to both UIs accessible:** ~60 s on a warm
  cache (postgres images cached, evidently/python images already
  pulled). Cold first-pull adds ~3-5 min. The MLflow server has a
  one-shot `pip install psycopg2-binary boto3` on first boot (the
  upstream image is intentionally minimal), which adds ~30 s on the
  very first start; subsequent boots cache and start fast.
- **Time from first run-create call to MLflow UI rendering it:**
  ~immediate — verified during Phase 3 smoke. The MLflow UI updates on
  refresh; no asynchronous lag like Langfuse's 30 s OTel batch.
- **Time from first prediction-log row to Evidently snapshot in UI:**
  60 s by default (one scan tick) once both reference + current
  windows have ≥ 5 rows each. Tunable via
  `EVIDENTLY_SCAN_INTERVAL_SECONDS`.
- **Bootstrap experience:** very pleasant. MLflow's REST API for
  experiment + run creation works without any UI signup. Evidently's
  workspace + project creation works the same way through
  `RemoteWorkspace`. The whole stack is "no auth, localhost-only" by
  design — production deploy would need TLS + an auth proxy in front.
- **Friction points (hit during Phase 1-7 implementation):**
  1. **Evidently 0.7's autorefresh hits an inotify-instance ceiling.**
     The default config wires a watchdog observer onto the workspace
     directory. With Coroot / LGTM / Sentry / Langfuse all running,
     `fs.inotify.max_user_instances` (default 128) is exhausted and
     the watchdog crash bubbles up as a 500 on `/api/projects` —
     symptom *without* `EVIDENTLY_DEBUG=1` is a bare 500 with no log.
     Fix is two env vars: `EVIDENTLY_STORAGE__TYPE=local` +
     `EVIDENTLY_STORAGE__AUTOREFRESH=false`. Captured in
     [`README.md`'s "Things that bit me"](README.md#things-that-bit-me-along-the-way).
  2. **Evidently 0.7 split the legacy and modern APIs.** Reports moved
     to `evidently.Report` (root namespace), presets to
     `evidently.presets`, workspace to
     `evidently.ui.workspace.RemoteWorkspace`. The legacy
     `evidently.report.Report` + `evidently.metric_preset.*` paths
     exist only under `evidently.legacy.*` and write a v1 metadata
     format the new UI can't read. The scheduler imports use the
     modern API throughout; documented in
     [`INTEGRATION-NOTES.md`](INTEGRATION-NOTES.md).
  3. **`code/requirements.txt` is encoded as UTF-16-with-BOM and contains
     literal spaces between every character** (`a i o s i g n a l = =
     1 . 3 . 1`). Out of scope for this PoC.
  4. **`mlflow` client `pandas<3` pin clashes with TFG's `pandas==3.0.2`.**
     Initially the client *was* added to `backend/requirements.txt`,
     but `make build` from PersonalPortfolio failed with
     `ResolutionImpossible` because every published `mlflow` release as
     of 2026-05 (incl. 3.11.1) still declares `pandas<3` even though
     pandas 3.0 shipped in January 2026. Fix: drop the client from the
     backend container — the FastAPI process never imports `mlflow`
     anyway; it writes prediction-log rows via plain `psycopg`. The
     three training scripts in `code/src/*.py` already had
     `try: import mlflow / except: _MLFLOW_OK = False` guards from
     Phase 3, so they no-op gracefully when run from a venv that
     doesn't have it and light up when one does. README's
     [Things that bit me](README.md#things-that-bit-me-along-the-way)
     captures the rationale.
  5. **MLflow image doesn't ship `psycopg2-binary` or `boto3`.** First
     boot does a one-shot `pip install` of both (~30 s) before
     starting the server. Tradeoff: starting from a thicker image
     would save 30 s per first-start but add ~80 MB to the pulled
     image. Phase-2 chose the smaller pull.

## Run-comparison UX (Scenario A)

_See [SCENARIOS.md](SCENARIOS.md#scenario-a--mlflow-comparing-optuna-trials-and-promoting-a-winner).
Capture how MLflow renders the parallel-coordinates plot and how many
clicks "best trial" → "registered model in Production" takes._

- **Did the parallel-coordinates plot make the winning trial visually obvious?** TBD. Hypothesis: yes, very. MLflow's compare view colours the high-metric line distinctly and the unused hyperparameter axes collapse to gray.
- **Did `mlflow.start_run(nested=True)` create the parent/child run hierarchy correctly?** TBD. Phase 3 wiring uses the documented Optuna pattern; first hands-on confirmation pending an actual sweep run.
- **Time from "I see the winning trial" to "registered model version in Production":** TBD. Expectation: ~30 s if the model artifact was logged, otherwise the registry promotion step is gated on having a `models:/` URI to point at.
- **Did the model registry feel like a real promotion gate or a metadata wrapper?** TBD. Expectation: the latter — the registry is a state machine over runs; deployment scripts still need their own "fetch model from registry" code (see scenario step 5).

## Drift-detection UX (Scenario B)

_See [SCENARIOS.md](SCENARIOS.md#scenario-b--evidently-triggering-drift-and-watching-the-dashboard).
How quickly does drift surface, and at what level of detail?_

- **Did the data-drift preset flag all expected drifted features?** TBD. Hypothesis: yes — Evidently's PSI + KS tests are well-known to fire on RGB shifts of the magnitude in the seeder SQL. Per-feature drift scores should reflect mean shift + variance change separately.
- **Was the reference-vs-current visualization legible?** TBD. Expectation: each numeric column gets a kernel-density overlay (reference in blue, current in red); shift visible at a glance.
- **Did the in-app `MlopsStatusCard` reflect the drift?** **No** in this PoC's current form — `drift_status` is hard-coded to `"unknown"` because computing it from prediction-log alone would be expensive. The proper fix is to extend the scheduler to write a `drift_state` row that the stats endpoint reads. Documented as a Phase-7-follow-up in [`README.md`](README.md#whats-not-covered).
- **How does Evidently's "data drift" presentation compare to the per-feature histograms a tool like Datadog or Arize would show?** TBD. Open question: whether the OSS preset matches what those SaaS tools render automatically.

## Cross-tier session correlation (TFG-internal)

_The new piece versus the other seven PoCs: a single `X-Session-Id`
flowing browser → Sentry trace → MLOps prediction-log row.
[INTEGRATION-NOTES.md §5 + §9](INTEGRATION-NOTES.md) documents the wire
contract; this section is for the user-facing UX observations._

- **Did pasting the session id from the `MlopsStatusCard` into the
  Sentry search bar return the matching backend traces?** TBD. The
  Sentry integration tags every trace with `session_id` via
  `_sentry_obs.py` already, so this is a one-click join.
- **Did the same id, queried in the prediction-log Postgres,
  return the inference rows for the same browser session?** TBD.
  `predictions_session_id_idx` is in place so the lookup is
  O(log n) on session count.
- **What about traces *without* a session id** (e.g. requests
  initiated outside the React UI)? They still appear in Sentry; the
  prediction-log row stores `session_id = NULL` for them, indexed by
  the partial index `predictions_session_id_idx WHERE session_id IS
  NOT NULL` so it doesn't bloat the index.
- **Friction:** the `MlopsStatusCard` exposes a one-click "Copy"
  button next to the session id, so pasting into Sentry's search is
  two clicks total (Copy → ⌘V into the Sentry search bar). The card
  also exposes a collapsible "What gets logged?" panel that maps each
  TFG tab to its sink (Inference → prediction-log → Evidently;
  Detection Training / Hyperparameter Tuning → MLflow), and a "Where
  to look" section with captioned deep links — so a first-time visitor
  can answer *"how do I make data show up here?"* and *"where do I go
  to see it?"* without leaving the page.

## Cost & operational shape

_How much it costs to actually run this. Memory / disk / retention,
free-tier ceilings if any, whether the default retention is enough for
the kind of investigation you actually want to do._

- **Memory budget (planned in [README.md](README.md#memory-budget)):**
  ~960 MiB total for the MLOps overlay (MLflow server + 2x Postgres
  + MinIO + Evidently UI + scheduler), on top of the existing ~1 GiB
  TFG container. Sits between Coroot's ~1.2 GiB and LGTM's ~3 GiB.
  TBD measured via `docker stats` once running.
- **Disk budget:** MLflow Postgres stores experiment metadata
  (~kB per run). MLflow MinIO stores model artifacts (the dominant
  cost — a ~50 MB FasterRCNN checkpoint per run, easily multiple GB
  for a long sweep). Prediction-log Postgres stores ~200 bytes per
  prediction row; even at 1k requests/day the table grows ~200 MB/yr.
  Evidently snapshots on the workspace volume are ~10-50 KB each.
  Default retention: unbounded everywhere. Production deploy would
  add a retention policy on prediction-log + MinIO.
- **CPU during ingestion:** prediction-log writes are offloaded via
  `BackgroundTasks` so request latency is unaffected. The Evidently
  scheduler does one PSI/KS computation per scan interval — ~10 ms
  on the 50-row windows in the seeder.
- **Retention vs. investigation horizon:** drift detection wants
  weeks of history (you're comparing this week's distribution against
  last month's). The other PoCs use 7-day defaults, but for MLOps
  drift detection 30+ days is more useful. Tunable via Postgres
  retention policy + MinIO bucket lifecycle rules.
- **Lock-in concerns:**
  - **MLflow:** **low** — runs are stored in Postgres + S3 in open
    formats (the run table schema is documented; artifacts are raw
    files). Migration to W&B / Comet / Neptune is non-trivial because
    the run JSON shape isn't 1:1, but the *data* is fully portable.
  - **Evidently:** **medium** — workspace metadata + snapshots are
    JSON in a known directory structure but the snapshot rendering is
    Evidently-specific. The underlying numerical drift metrics are
    PSI / KS / Chi² — easy to recompute from the same prediction-log
    in any other tool.
  - **The prediction-log Postgres** is the real keystone — it's a
    plain SQL table with documented columns, and it would survive a
    swap of either component above.

## Verdict

_The 1-paragraph "if you had to pick one for the portfolio's MLOps
observability stack tomorrow" call. Reference Scenarios A, B
observations._

TBD — write this last, after the scenarios actually run. Working
hypothesis (to be confirmed or contradicted by empirical data): MLflow
+ Evidently is the **right OSS pairing** for the corner of the design
space the rest of the portfolio's stacks couldn't reach. **MLflow is
best at** experiment tracking + model registry; the run comparison
view + parallel-coordinates plot is a real productivity win versus
hand-tracking hyperparameters in spreadsheets. **Evidently is best at**
tabular data-drift detection; the OSS UI is rougher than the SaaS
incumbents (Arize, Fiddler, WhyLabs Cloud) but the math is the same
math. **MLflow is worst at** real-time inference monitoring (it's a
training-side tool); for that, the prediction-log + Evidently scheduler
fills the gap. **Evidently is worst at** raw-pixel drift on CV tasks —
the trick of capturing scalar features (mean RGB / brightness)
sidesteps this completely and gives Evidently a tabular dataset to
chew on, which it handles well. The "recommended next experiment" is
likely either (a) a head-to-head against W&B + WhyLabs (the SaaS
incumbents) on the same TFG repo to feel the OSS-vs-SaaS friction
delta concretely, or (b) wiring the Evidently drift state back into
Sentry as a custom event so the existing alerting surface doubles for
MLOps issues without standing up a second alert pipeline.

---

## Cross-PoC comparison

_Filled in once all eight PoCs (Sentry, LGTM, Honeycomb, Elastic,
Coroot, Embrace, Langfuse, MLflow+Evidently) have their notes filled
in. The interesting cells are "best at" and "worst at" — that's where
the design choices actually matter._

| Axis                    | Sentry (everywhere main) | LGTM (Practica_de_Planificacion) | Honeycomb (pracpro2) | Elastic (tenda_online) | Coroot (subgrup-prop7.1) | Embrace (Draculin) | Langfuse + LiteLLM (openclaw-ai) | MLflow + Evidently (this PoC) |
|-------------------------|--------------------------|----------------------------------|----------------------|------------------------|--------------------------|--------------------|----------------------------------|-------------------------------|
| Setup time              | ~10 min per backend (DSN + init) | ~3 hrs (4 stores + Collector)    | ~30 min (account + DSN) | ~6 hrs (Beats lockstep) | ~4 hrs (eBPF + version pinning) | ~2 hrs (native bootstrap + symbolication) | ~2 hrs (Compose overlay + LiteLLM config merge) | ~3 hrs (Compose overlay + scheduler + frontend integration); known landmines documented up front |
| First-meaningful-query  | error groups (immediate) | RED dashboard (15 min)           | trace-flat query (5 min) | KQL (10 min)         | per-route HTTP RED (5 min) | session crash list (immediate) | Tracing → Traces (~30 s once first trace lands) | MLflow runs list (immediate after first `start_run`); Evidently after one scan tick (~60 s) |
| Custom training-run tracking | none           | none (would need custom span fields) | spans hold params if logged | none           | none                     | none               | none (proxy paradigm)             | **Yes — `mlflow.start_run` + autolog is the headline feature** |
| Data-drift visibility   | none                     | none                             | none                 | none                   | none                     | none               | none                              | **Yes — Evidently scheduled job + report UI** |
| Compute pressure (GPU/CPU) | minimal (process metrics) | yes via Mimir                  | no                   | yes via Metricbeat     | **yes via eBPF**         | on-device only     | no (proxy can't see this)         | no (training-side only) |
| Code-change required    | per-language SDK init    | OTel SDK + auto-instr            | tracing-otel + spans | native PHP `.so` agent | **none** (eBPF) + 1 JVM flag | native bootstrap + Dart wrapper | **none** (config change in openclaw.json) | 2-line `mlflow.start_run` block in each training script + 1-line `_mlops_log_prediction` hook in `predict()` |
| Free-tier limits / cost | 5K events/mo SaaS        | OSS (S3 storage = pay-per-GiB)   | 20M events/mo, 60d   | OSS (Basic license)    | OSS Community Edition    | 5K MAU, 1M sessions/mo SaaS | OSS self-host; Cloud free 50K obs/mo | OSS (MLflow Apache-2.0 + Evidently Apache-2.0); SaaS Evidently free for 50K rows/mo |
| Lock-in risk            | high (SDK shape)         | low (OTel-native)                | high (Honeycomb-only) | high (ECS + Beats)    | low (Prom + ClickHouse)  | very high (no self-host, no export) | low (OTel ingest), medium (Langfuse-specific dashboards) | low for *data* (Postgres + S3); medium for *workflow* (MLflow runs JSON / Evidently snapshots) |
| **Distinctive corner of the design space** | Single-vendor SaaS APM | OSS pillar architecture        | High-cardinality wide events | Document-store + Beats | Zero-instrumentation eBPF | User-session-centric mobile RUM | Proxy-based LLM observability | **MLOps observability — experiment tracking, model registry, data drift** |
| Best at                 | "is prod broken?" + replay | Long-retention metrics + RED   | "why is THIS user's request slow" | One query plane across signal types | "what can I observe without app changes" + service map | "did this user's app crash, what did they do in the 30s before" | "what did the agent ask the model, and what did it cost?" | "which experiment beat the baseline" + "have inputs drifted since training" |
| Worst at                | Long-tail metrics; real logs | Anomaly-explanation UX        | No real metrics or logs | Extreme-cardinality span analytics | Custom span boundaries; exception-class grouping | Real-time alerting; backend opacity | Agent-level grouping; compute-pressure visibility | Real-time prediction monitoring (training-time tool); raw-pixel CV drift (mitigated via scalar feature extraction) |

The interesting cells for **this PoC** are the *training-run tracking*
and *data-drift visibility* rows — they're the only places this stack
does something none of the others do, and they're exactly the
questions an ML-app dev needs to answer that an APM / wide-events /
session-payload stack can't help with. The "worst at" row tells you
when to reach for Coroot / LGTM / Sentry instead — i.e. for serving-time
observability, not training-time and not drift.

The **portfolio-level integration** angle (the cross-tier 3-layer
correlation Sentry session → backend trace → MLOps prediction record,
all keyed by the same `X-Session-Id`) is unique to this PoC. None of
the other seven PoCs achieves this 3-layer loop because none of them
runs alongside Sentry on the same backend with a third independent
storage layer. Embrace tags sessions but inside its own SaaS sandbox;
Coroot tags HTTP requests but not ML inferences; LGTM has trace_id but
only on the OTel side. The integration is documented in
[`INTEGRATION-NOTES.md`](INTEGRATION-NOTES.md) §5 and surfaced in the
in-app `MlopsStatusCard` (the session id is printed at the bottom for
copy-paste into Sentry's search).

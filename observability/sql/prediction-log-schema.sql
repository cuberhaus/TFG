-- Prediction log — one row per inference.
--
-- Evidently scans the `predictions` table on a 1-min schedule:
-- the rows in the most-recent EVIDENTLY_CURRENT_WINDOW_HOURS window
-- become the "current" dataset, and the EVIDENTLY_REFERENCE_WINDOW_HOURS
-- window before that becomes the "reference" dataset. Drift metrics are
-- computed across all numeric columns marked NOT NULL.
--
-- Schema is deliberately scalar-only (no raw pixel data, no embeddings).
-- See observability/INTEGRATION-NOTES.md §8 for the rationale.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS predictions (
  -- ── row scaffold ──
  id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
  endpoint                 TEXT NOT NULL,            -- "/api/predict" or "/api/predict/batch"
  session_id               TEXT,                     -- X-Session-Id header, may be NULL

  -- ── input feature stats (drift detection target) ──
  image_width              INTEGER NOT NULL,
  image_height             INTEGER NOT NULL,
  image_aspect_ratio       DOUBLE PRECISION NOT NULL,
  mean_r                   DOUBLE PRECISION NOT NULL,
  mean_g                   DOUBLE PRECISION NOT NULL,
  mean_b                   DOUBLE PRECISION NOT NULL,
  mean_brightness          DOUBLE PRECISION NOT NULL,

  -- ── model metadata ──
  model_arch               TEXT NOT NULL,            -- e.g. "FasterRCNN"
  model_file               TEXT NOT NULL,            -- the .pth basename
  confidence_threshold     DOUBLE PRECISION NOT NULL,

  -- ── prediction stats (output drift detection target) ──
  box_count                INTEGER NOT NULL,
  mean_confidence          DOUBLE PRECISION,         -- NULL when no boxes detected
  max_confidence           DOUBLE PRECISION,         -- NULL when no boxes detected
  inference_latency_ms     INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS predictions_created_at_idx
  ON predictions (created_at DESC);

CREATE INDEX IF NOT EXISTS predictions_session_id_idx
  ON predictions (session_id)
  WHERE session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS predictions_model_file_idx
  ON predictions (model_file, created_at DESC);

-- Convenience view: per-day rollup for the /api/mlops/stats endpoint.
CREATE OR REPLACE VIEW predictions_today AS
SELECT
  count(*)                                 AS total,
  count(*) FILTER (WHERE box_count > 0)    AS with_boxes,
  count(DISTINCT session_id)               AS distinct_sessions,
  avg(inference_latency_ms)::INTEGER       AS avg_latency_ms,
  max(inference_latency_ms)                AS max_latency_ms
FROM predictions
WHERE created_at >= date_trunc('day', now());

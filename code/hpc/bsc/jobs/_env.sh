# shellcheck shell=bash
# Sourced at the top of every job: modules, virtualenv, offline caches, helpers.
# Needs remote.env (written by `bsc.sh push code`) in the directory the job was submitted from.
set -euo pipefail

BSC_REMOTE_ENV="${BSC_REMOTE_ENV:-${SLURM_SUBMIT_DIR:?submit this with sbatch (or bsc.sh submit)}/remote.env}"
# shellcheck disable=SC1090
source "$BSC_REMOTE_ENV"

# JOB_DRY_RUN=1 prints the commands a job would run, without modules / venv / GPU (used for local testing).
if [ -z "${JOB_DRY_RUN:-}" ]; then
  if ! command -v module >/dev/null 2>&1 && [ -r /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
  # shellcheck disable=SC2086  # module lists are deliberately word-split
  module load $BSC_PYTHON_MODULE ${BSC_EXTRA_MODULES:-}
  if [ -z "${BSC_SKIP_VENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$BSC_VENV/bin/activate"
  fi
fi

# No internet on any node: everything must come from the staged caches.
export HF_HOME="$BSC_WORK/hf" TORCH_HOME="$BSC_WORK/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

CPUS="${SLURM_CPUS_PER_TASK:-8}"
# shellcheck disable=SC2034  # used by the job scripts that source this file
WORKERS=$(( CPUS > 4 ? CPUS - 2 : 2 ))   # dataloader workers; leave a couple of cores for the main process

# Print and run a command (or only print it under JOB_DRY_RUN).
run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [ -z "${JOB_DRY_RUN:-}" ]; then "$@"; fi
}

require_gpu() {
  echo "host $(hostname) | job ${SLURM_JOB_ID:-local} | $(date -Is)"
  [ -z "${JOB_DRY_RUN:-}" ] || return 0
  python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible to this job (did the job request --gres=gpu:1?)")
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| cuda", torch.version.cuda)
PY
}

cd "$BSC_CODE/code/src"   # every polyp_synth / legacy entry point is run from code/src

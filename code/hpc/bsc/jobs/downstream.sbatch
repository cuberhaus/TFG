#!/bin/bash
#SBATCH --job-name=polyp-down
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Does adding synthetic polyps improve a polyp detector? One array task = one (seed, arm) training run
# (polyp_synth.downstream); every task writes its own JSON, so a failed task can simply be re-submitted.
#   ARMS="real neg synth" SEEDS="0 1 2" ./bsc.sh submit downstream --array=0-8 [-- <extra downstream args>]
# --array must be 0..(len(ARMS)*len(SEEDS)-1). Env: SYNTH_NAME, TAG (results folder, default "full").
# Afterwards:  ./bsc.sh pull polyp_synth/downstream/<TAG>   and, locally,
#              python -m polyp_synth.downstream report --dir ../out/polyp_synth/downstream/<TAG>
source "${SLURM_SUBMIT_DIR:?submit with sbatch}/jobs/_env.sh"

SYNTH_NAME="${SYNTH_NAME:-synth_hpc}"
TAG="${TAG:-full}"
# shellcheck disable=SC2206  # space-separated lists on purpose
arm_list=(${ARMS:-real neg synth})
# shellcheck disable=SC2206
seed_list=(${SEEDS:-0 1 2})
total=$(( ${#arm_list[@]} * ${#seed_list[@]} ))
idx="${SLURM_ARRAY_TASK_ID:?submit as an array job (--array=0-$((total - 1)))}"
if [ "$idx" -ge "$total" ]; then echo "array index $idx >= $total (arms x seeds): nothing to do"; exit 0; fi

seed="${seed_list[$(( idx / ${#arm_list[@]} ))]}"
arm="${arm_list[$(( idx % ${#arm_list[@]} ))]}"
echo "task $idx/$total -> arm=$arm seed=$seed"

require_gpu
run python -m polyp_synth.downstream run --arms "$arm" --seeds "$seed" \
  --synth-dir "../out/polyp_synth/$SYNTH_NAME" \
  --neg-dir ../data/polyp_synth/raw/negatives_heldout \
  --workers "$WORKERS" --out "../out/polyp_synth/downstream/$TAG" "$@"

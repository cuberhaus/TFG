#!/bin/bash
#SBATCH --job-name=polyp-gen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Paint synthetic polyps onto polyp-free frames (polyp_synth.generate), keeping the original frame next to
# each result (Before/) so the "no polyp" control arm of the downstream experiment uses the same frames.
#   ./bsc.sh submit generate [-- <extra polyp_synth.generate args>]
# Env: RUN_NAME (LoRA to use), SYNTH_NAME (output folder), N_SYNTH, SEED.
source "${SLURM_SUBMIT_DIR:?submit with sbatch}/jobs/_env.sh"

RUN_NAME="${RUN_NAME:-hpc_v1}"
SYNTH_NAME="${SYNTH_NAME:-synth_hpc}"
N_SYNTH="${N_SYNTH:-1800}"
SEED="${SEED:-1}"

require_gpu
# reserve held-out polyp-free frames (for the false-positive test) so they are never used as backgrounds
run python -m polyp_synth.split_negatives
run python -m polyp_synth.generate \
  --lora "../out/polyp_synth/runs/$RUN_NAME/final" \
  --frames ../data/polyp_synth/raw/negatives_gen \
  --n "$N_SYNTH" --seed "$SEED" --save-before --sheet 8 \
  --out "../out/polyp_synth/$SYNTH_NAME" "$@"

#!/bin/bash
#SBATCH --job-name=polyp-gate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Keep only synthetic frames whose painted polyp a detector trained on REAL polyps finds
# (polyp_synth.detector_check). Trains that detector first if out/polyp_synth/detector/det.pt is missing.
#   ./bsc.sh submit gate          Env: SYNTH_NAME, KEEP_THR (default 0.5)
source "${SLURM_SUBMIT_DIR:?submit with sbatch}/jobs/_env.sh"

SYNTH_NAME="${SYNTH_NAME:-synth_hpc}"

require_gpu
[ -f ../out/polyp_synth/detector/det.pt ] || run python -m polyp_synth.detector_check train --epochs 15
run python -m polyp_synth.detector_check score --dir "../out/polyp_synth/$SYNTH_NAME" --keep-thr "${KEEP_THR:-0.5}"

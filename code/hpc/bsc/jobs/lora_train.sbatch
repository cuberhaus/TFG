#!/bin/bash
#SBATCH --job-name=polyp-lora
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Fine-tune the Stable Diffusion inpainting LoRA that paints polyps into masks (polyp_synth.train).
#   ./bsc.sh submit lora_train [--time=04:00:00] [-- <polyp_synth.train args>]
# No job args -> the settings of the local "v1" run (4000 steps). Output: out/polyp_synth/runs/$RUN_NAME/
source "${SLURM_SUBMIT_DIR:?submit with sbatch}/jobs/_env.sh"

RUN_NAME="${RUN_NAME:-hpc_v1}"
if [ "$#" -gt 0 ]; then ARGS=("$@"); else ARGS=(--steps 4000 --no-grad-ckpt); fi

require_gpu
run python -m polyp_synth.train --run-name "$RUN_NAME" --workers "$WORKERS" "${ARGS[@]}"

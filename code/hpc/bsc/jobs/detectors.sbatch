#!/bin/bash
#SBATCH --job-name=tfg-detectors
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Thesis baselines: train Faster R-CNN / SSD / RetinaNet on LDPolypVideo from a JSON list (train_models.py).
# Needs the dataset on the cluster:  ./bsc.sh push data TrainValid && ./bsc.sh push data Test
#   ./bsc.sh submit detectors --time=12:00:00 --qos=<long-queue> [-- ../configurations/model_configs.json]
#   DEBUG=1 ./bsc.sh submit detectors        # tiny subset, to check the whole chain works
source "${SLURM_SUBMIT_DIR:?submit with sbatch}/jobs/_env.sh"

CONFIG_JSON="${1:-../configurations/model_configs.json}"
EXTRA=()
[ -z "${DEBUG:-}" ] || EXTRA+=(--debug)

require_gpu
run python train_models.py "$CONFIG_JSON" "${EXTRA[@]+"${EXTRA[@]}"}"

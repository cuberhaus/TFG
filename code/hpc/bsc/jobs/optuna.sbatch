#!/bin/bash
#SBATCH --job-name=tfg-optuna
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Hyper-parameter search (optuna_train_model.py): one GPU per model instead of the three background
# processes of the old optuna_train.sh. Array task 0/1/2 = FasterRCNN / RetinaNet / SSD.
# Needs the LDPolypVideo dataset on the cluster (see detectors.sbatch).
#   ./bsc.sh submit optuna --time=24:00:00 --qos=<long-queue> [-- --metric mean_iou]
#   N_TRIALS=30 MAX_EPOCHS=5 ./bsc.sh submit optuna      (defaults: the script's own, 5 and 5)
source "${SLURM_SUBMIT_DIR:?submit with sbatch}/jobs/_env.sh"

models=(FasterRCNN RetinaNet SSD)
idx="${SLURM_ARRAY_TASK_ID:?submit as an array job (--array=0-2)}"
model="${models[$idx]}"
EXTRA=()
[ -z "${N_TRIALS:-}" ] || EXTRA+=(--n-trials "$N_TRIALS")
[ -z "${MAX_EPOCHS:-}" ] || EXTRA+=(--max-epochs "$MAX_EPOCHS")

require_gpu
run python optuna_train_model.py "$model" "${EXTRA[@]+"${EXTRA[@]}"}" "$@"

#!/bin/bash
#SBATCH --job-name=tfg-setup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# One-off: build the virtualenv from the wheelhouse copied by `bsc.sh push stage` (no internet needed),
# link code/data and code/out to the work area, and check that every package imports.
# Run via:  ./bsc.sh setup
BSC_SKIP_VENV=1
source "${SLURM_SUBMIT_DIR:?submit with sbatch}/jobs/_env.sh"

WHEELS="$BSC_WORK/stage/wheelhouse"
[ -d "$WHEELS" ] || { echo "no wheelhouse at $WHEELS - run: ./bsc.sh build && ./bsc.sh push stage"; exit 1; }

run python3 -m venv "$BSC_VENV"
# shellcheck disable=SC1091
[ -n "${JOB_DRY_RUN:-}" ] || source "$BSC_VENV/bin/activate"
# shellcheck disable=SC2086  # BSC_TORCH_SPEC is a list of requirement specifiers
run pip install --no-index --find-links "$WHEELS" -r "$BSC_CODE/code/hpc/bsc/requirements-hpc.txt" $BSC_TORCH_SPEC

# the code resolves ../data and ../out relative to code/; keep the big stuff out of /gpfs/home
run mkdir -p "$BSC_WORK/data" "$BSC_WORK/out"
run ln -sfn "$BSC_WORK/data" "$BSC_CODE/code/data"
run ln -sfn "$BSC_WORK/out" "$BSC_CODE/code/out"

run python -c "import torch, torchvision, diffusers, peft, optuna, cv2; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| diffusers', diffusers.__version__)"
echo "setup ok"

# shellcheck shell=bash disable=SC2034,SC2016
# Copy to config.sh (git-ignored) and fill in every <placeholder>.  Sourced by bsc.sh.
#
# Written for MareNostrum 5 ACC (4x NVIDIA H100 per node, 20 CPUs per GPU).
# Facts this relies on (BSC support docs): the login nodes are the only way in, NOTHING in the
# cluster can reach the internet, source code goes in /gpfs/home, data and outputs in
# /gpfs/projects or /gpfs/scratch, and every job needs an --account and a --qos.

# --- who -------------------------------------------------------------------
BSC_USER="<your-bsc-username>"
BSC_ACCOUNT="<project-account>"   # Slurm account = your project's unix group. On the cluster: `bsc_project list`
BSC_QOS="acc_debug"               # `bsc_queues` lists yours. acc_debug = 2 h max (good for a first test);
                                  # for real runs use your project's queue, e.g. acc_bsccs / acc_resa / acc_ehpc
BSC_PARTITION=""                  # leave empty: the QOS selects the partition

# --- hosts -----------------------------------------------------------------
BSC_LOGIN="alogin1.bsc.es"        # MN5 ACC login (GPP would be glogin1.bsc.es)
BSC_TRANSFER="transfer1.bsc.es"   # big file transfers go here, not through the login node

# --- where things live on the cluster ---------------------------------------
# $HOME is expanded on the cluster, not here. Find your real scratch/projects path with `ls /gpfs/scratch`.
REMOTE_CODE='$HOME/TFG'                           # source + virtualenv (backed up, small)
REMOTE_WORK='/gpfs/scratch/<group>/<user>/TFG'    # datasets, model caches, outputs (scratch has NO backup:
                                                  # copy anything you want to keep to /gpfs/projects)

# --- resources per job -------------------------------------------------------
GRES="gpu:1"
CPUS_PER_JOB=20                   # MN5 ACC enforces 20 CPUs per GPU

# --- Python on the cluster ---------------------------------------------------
# Because nothing can be downloaded there, wheels are downloaded HERE for the cluster's Python version
# and copied over. Check the cluster with `module avail python`, then keep these two in sync.
PYTHON_MODULE="python/3.10"
PYTHON_VERSION="3.10"
EXTRA_MODULES=""                  # e.g. "cuda" - not needed: the PyTorch wheels bundle their own CUDA libraries
# pip needs every wheel tag the cluster accepts (older packages are tagged manylinux2014 / 2_17, not 2_28)
PIP_PLATFORMS="manylinux_2_28_x86_64 manylinux_2_17_x86_64 manylinux2014_x86_64"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_SPEC="torch==2.11.0+cu128 torchvision==0.26.0+cu128"   # the pair this repo's polyp_synth code was tested with

# Local interpreter that has torch, torchvision and diffusers (used only to download model weights).
LOCAL_PYTHON="$HOME/.venvs/tfg-gen/bin/python"

# --- CTE-AMD (AMD MI50 cluster the thesis planned to use) ---------------------
# Would need: BSC_LOGIN=amdlogin1.bsc.es, no --account/--qos, GRES="gpu:1", CPUS_PER_JOB=32,
# PYTHON_MODULE+rocm module, and a ROCm PyTorch build for TORCH_INDEX/TORCH_SPEC.
# Untested: MI50 (gfx906) support in current ROCm PyTorch wheels was not verified.

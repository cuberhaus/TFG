#!/usr/bin/env bash
# Runs on the cluster login node (started by `bsc.sh pipeline`): queues the whole synthetic-data experiment as a
# dependency chain, so nothing starts before the step it needs has finished successfully (afterok):
#
#   lora_train -> generate -> gate -> downstream (array: arms x seeds)
#
# Env (all optional): RUN_NAME SYNTH_NAME N_SYNTH SEED ARMS SEEDS TAG KEEP_THR, SBATCH_FLAGS (account/qos/gres).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
RUN_NAME="${RUN_NAME:-hpc_v1}"
SYNTH_NAME="${SYNTH_NAME:-synth_hpc}"
export RUN_NAME SYNTH_NAME   # sbatch exports the whole environment to the jobs by default

# shellcheck disable=SC2206
arms=(${ARMS:-real neg synth})
# shellcheck disable=SC2206
seeds=(${SEEDS:-0 1 2})
last=$(( ${#arms[@]} * ${#seeds[@]} - 1 ))

submit() {  # submit [sbatch flags...] script  -> prints the job id
  local id
  # shellcheck disable=SC2086  # SBATCH_FLAGS is a space-separated flag list
  id="$("$SBATCH_BIN" --parsable ${SBATCH_FLAGS:-} "$@")"
  echo "${id%%;*}"
}

j_lora="$(submit jobs/lora_train.sbatch)"
j_gen="$(submit --dependency="afterok:$j_lora" jobs/generate.sbatch)"
j_gate="$(submit --dependency="afterok:$j_gen" jobs/gate.sbatch)"
j_down="$(submit --dependency="afterok:$j_gate" --array="0-$last" jobs/downstream.sbatch)"

cat <<EOF
queued: lora_train=$j_lora -> generate=$j_gen -> gate=$j_gate -> downstream=$j_down (array 0-$last)
results when done:  ./bsc.sh pull polyp_synth/downstream/${TAG:-full}
then, locally:      python -m polyp_synth.downstream report --dir ../out/polyp_synth/downstream/${TAG:-full}
EOF

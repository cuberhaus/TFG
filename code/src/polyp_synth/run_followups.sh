#!/usr/bin/env bash
# Follow-up experiments to the full-data downstream test (does adding synthetic polyps help a detector?).
#
#   E1  gate ablation      full data, "synth_all" arm (synthetic polyps WITHOUT the detector gate)
#   E2  low data, 10%      ~142 real frames; LoRA + gate + mask bank restricted to the same frames; 3 draws x 2 seeds
#   E3  cross-dataset      train on Kvasir-SEG only, test on all of CVC-ClinicDB (never seen by anything)
#   E4  low data, 25%      ~356 real frames; 2 draws x 2 seeds
#
# In every low-data / cross-dataset unit EVERYTHING that learns from real polyps (the LoRA, the gate
# detector, the mask bank placing synthetic polyps, and the downstream detectors) sees the same
# subset of real frames, so no information from held-back real data leaks into the synthetic images.
#
# Resumable: each step leaves a marker file and is skipped on re-run, so after a crash or a reboot
# just run the same command again.
#
#   bash polyp_synth/run_followups.sh                  # from code/src; all phases
#   PHASES="E2" bash polyp_synth/run_followups.sh      # only some
#   SMOKE=1 bash polyp_synth/run_followups.sh          # tiny end-to-end check (minutes, results meaningless)
set -euo pipefail
cd "$(dirname "$0")/.."   # code/src

PY=${PY:-$HOME/.venvs/tfg-gen/bin/python}
OUT=../out/polyp_synth
RAW=../data/polyp_synth/raw
PHASES=${PHASES:-"E1 E2 E3 E4"}
SMOKE=${SMOKE:-0}
FU=$OUT/followups$([ "$SMOKE" = 1 ] && echo _smoke)
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # base weights are cached; avoid network stalls
mkdir -p "$FU"

log() { echo "[$(date +%T)] $*"; }

need_disk() {  # abort early instead of filling the disk mid-run
    local free_gb
    free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    if [ "$free_gb" -lt "$1" ]; then
        echo "only ${free_gb} GB free on /, need at least $1 GB" >&2
        exit 3
    fi
}

# unit TAG FRAC DRAW SOURCES LORA_STEPS N_GEN "DET_SEEDS" [EXTERNAL]
unit() {
    local tag=$1 frac=$2 draw=$3 src=$4 lsteps=$5 ngen=$6 seeds=$7 ext=${8:-}
    local dir=$FU/$tag
    local sub="--frac $frac --subset-seed $draw"
    [ -n "$src" ] && sub="$sub --sources $src"
    local dsx=""
    [ -n "$ext" ] && dsx="--external-eval $ext"
    local dextra=""
    if [ "$SMOKE" = 1 ]; then lsteps=20; ngen=12; dextra="--steps 30 --n-extra 8"; seeds=${seeds%% *}; fi
    mkdir -p "$dir"
    need_disk 3

    if [ ! -f "$dir/.lora_done" ]; then
        log "[$tag] LoRA ($lsteps steps) on the regime's real frames only"
        rm -rf "$OUT/runs/$tag"
        $PY -m polyp_synth.train --run-name "$tag" --steps "$lsteps" $sub --seed "$draw" --no-grad-ckpt --workers 4 \
            --sample-every 100000 --save-every 100000 > "$dir/lora.log" 2>&1
        rm -rf "$OUT/runs/$tag"/ckpt-*   # the final adapter is all we need
        touch "$dir/.lora_done"
    fi
    if [ ! -f "$dir/.gate_done" ]; then
        log "[$tag] gate detector on the same real frames"
        $PY -m polyp_synth.detector_check train --out "$dir/detector" $sub --steps 600 > "$dir/gate.log" 2>&1
        touch "$dir/.gate_done"
    fi
    if [ ! -f "$dir/.gen_done" ]; then
        log "[$tag] generating $ngen synthetic frames"
        rm -rf "$dir/synth"
        $PY -m polyp_synth.generate --lora "$OUT/runs/$tag/final" --frames "$RAW/negatives_gen" --out "$dir/synth" \
            --n "$ngen" --seed $((100 + draw)) --save-before --sheet 8 $sub > "$dir/gen.log" 2>&1
        touch "$dir/.gen_done"
    fi
    if [ ! -f "$dir/.score_done" ]; then
        log "[$tag] gating with the regime's own detector"
        $PY -m polyp_synth.detector_check score --dir "$dir/synth" --det "$dir/detector/det.pt" --keep-thr 0.5 > "$dir/score.log" 2>&1
        touch "$dir/.score_done"
    fi
    log "[$tag] kept $(tr -s '\n' '\n' < "$dir/synth/kept.txt" | grep -c .) of $ngen synthetic frames"
    log "[$tag] downstream detectors (seeds: $seeds)"
    $PY -m polyp_synth.downstream run --arms real neg synth --seeds $seeds $sub $dsx $dextra \
        --synth-dir "$dir/synth" --out "$dir/downstream" >> "$dir/downstream.log" 2>&1
    rm -f "$dir/detector/det.pt"   # 160 MB each; the scores are already in synth/scores.json
    log "[$tag] done"
}

want() { [[ " $PHASES " == *" $1 "* ]]; }
SECONDS=0

if want E1 && [ "$SMOKE" != 1 ]; then
    log "E1: gate ablation, synth_all x 3 seeds, full data"
    $PY -m polyp_synth.downstream run --arms synth_all --seeds 0 1 2 --out "$OUT/downstream/full" > "$FU/e1.log" 2>&1
    $PY -m polyp_synth.downstream report --dir "$OUT/downstream/full" > /dev/null
    log "E1 done -> $OUT/downstream/full/report.md"
fi

if want E2; then
    log "E2: low data, 10% of the real frames"
    for d in 0 1 2; do unit "low10_d$d" 0.10 "$d" "" 800 300 "0 1"; [ "$SMOKE" = 1 ] && break; done
    $PY -m polyp_synth.downstream report --dir "$FU"/low10_d*/downstream --md "$FU/report_low10.md" > /dev/null
    log "E2 done -> $FU/report_low10.md"
fi

if want E3; then
    log "E3: cross-dataset, train Kvasir-SEG only -> test on all of CVC-ClinicDB"
    unit "xd_kv2cvc" 1.0 0 kvasir 2500 1300 "0 1 2" cvc
    $PY -m polyp_synth.downstream report --dir "$FU/xd_kv2cvc/downstream" --set val --md "$FU/report_xd_kvasir_val.md" > /dev/null
    $PY -m polyp_synth.downstream report --dir "$FU/xd_kv2cvc/downstream" --set ext --md "$FU/report_xd_cvc.md" > /dev/null
    log "E3 done -> $FU/report_xd_cvc.md"
fi

if want E4 && [ "$SMOKE" != 1 ]; then
    log "E4: low data, 25% of the real frames"
    for d in 0 1; do unit "low25_d$d" 0.25 "$d" "" 1000 700 "0 1"; done
    $PY -m polyp_synth.downstream report --dir "$FU"/low25_d*/downstream --md "$FU/report_low25.md" > /dev/null
    log "E4 done -> $FU/report_low25.md"
fi

log "all requested phases finished in $((SECONDS / 60)) min"

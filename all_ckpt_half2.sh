#!/usr/bin/env bash
# ==========================================================================
# Dump 5000 molecules for EVERY trained checkpoint, across 3 seeds, with the
# full metric/plot suite (reward hist, FCD + descriptor comparison to GEOM,
# EDM atom/mol stability, per-molecule csv). Then run the aggregator to build
# the thesis master CSV + cross-model plots with seed error bars.
#
# 65 checkpoints x 3 seeds x 5000 mols @ diff_steps=18. This is a LOT of GPU
# time (FCD is a neural embedding; EDM stability is O(N^2)/mol). The driver is
# RESUMABLE (skips a (ckpt,seed) whose dump_summary.json exists) and runs up to
# MAX_PARALLEL at a time across NUM_GPUS.
#
# Requires: final_dump.py, gflow.py (patched), edm_metrics.py on PYTHONPATH,
# and a GEOM reference SMILES file (--ref_smiles / REF_SMILES).
# ==========================================================================

set -u

# ----------------------------- knobs --------------------------------------
CKPT_ROOT="${CKPT_ROOT:-logs/quetzal-gfn}"
OUT_ROOT="${OUT_ROOT:-dumps}"
N="${N:-5000}"
SEEDS="${SEEDS:-0 100 42}"
DIFF_STEPS="${DIFF_STEPS:-18}"        # matches training; do NOT lower
REF_SMILES="${REF_SMILES:-results/geom_drugs_smiles.txt}"   # GEOM reference SMILES .txt (~300k)
# how many reference SMILES to load for FCD/descriptor comparison. Defaults to N
# so the reference sample is the SAME size as the generated set (balanced FCD).
# The GEOM file has ~300k; FCD is O(ref) per call, so loading all 300k x 204 runs
# is very slow -- matching N keeps it cheap AND fair. See note below on FCD bias.
REF_LIMIT="${REF_LIMIT:-$N}"
DATASET="${DATASET:-geom}"
GUIDE_SOURCE="${GUIDE_SOURCE:-ema}"
CKPT_FILE="${CKPT_FILE:-last.ckpt}"   # filename inside each */checkpoints/ dir
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
MAIN_LOG="${MAIN_LOG:-dump_master_half2.log}"   # shared, tagged log of ALL runs
# start fresh main log for this invocation (per-run dump.log files are appended
# to separately and preserved across resumes)
: > "$MAIN_LOG"
echo "[main] logging all runs (tagged) to $MAIN_LOG ; per-run logs in <out>/dump.log" | tee -a "$MAIN_LOG"

if [[ ! -f "$REF_SMILES" ]]; then
  echo "[warn] REF_SMILES=$REF_SMILES not found -- FCD-vs-GEOM and descriptor-vs-GEOM"
  echo "       comparisons will be skipped. Set REF_SMILES=/path/to/geom.smi to enable."
fi

# ----------------------------- checkpoint list ----------------------------
CHECKPOINTS=(
  sweep-osim-base-rtb-replay_off-b10
  sweep-osim-base-rtb-replay_on-b1
  sweep-osim-base-rtb-replay_on-b10
  sweep-osim-hidden-db-replay_off-b1
  sweep-osim-hidden-db-replay_off-b10
  sweep-osim-hidden-db-replay_off-b100
  sweep-osim-hidden-db-replay_on-b1
  sweep-osim-hidden-db-replay_on-b10
  sweep-osim-hidden-db-replay_on-b100
  sweep-osim-hidden-revkl-replay_off-b1
  sweep-osim-hidden-revkl-replay_off-b10
  sweep-osim-hidden-rtb-replay_off-b1
  sweep-osim-hidden-rtb-replay_off-b10
  sweep-osim-hidden-rtb-replay_on-b1
  sweep-osim-hidden-rtb-replay_on-b10
  sweep-osim-tempgain-db-replay_off-b1
  sweep-osim-tempgain-db-replay_off-b10
  sweep-osim-tempgain-db-replay_on-b1
  sweep-osim-tempgain-db-replay_on-b10
  sweep-peri-base-db-replay_off-b1
  sweep-peri-base-db-replay_off-b10
  sweep-peri-base-db-replay_on-b1
  sweep-peri-base-db-replay_on-b10
  sweep-peri-hidden-db-replay_off-b1
  sweep-peri-hidden-db-replay_off-b10
  sweep-peri-hidden-db-replay_on-b1
  sweep-peri-hidden-db-replay_on-b10
  sweep-peri-tempgain-db-replay_off-b1
  sweep-peri-tempgain-db-replay_off-b10
  sweep-peri-tempgain-db-replay_on-b1
  sweep-peri-tempgain-db-replay_on-b10
  stability-geom-base-db-b1
  stability-geom-base-db-b10
  stability-geom-hidden-db-b1
  stability-geom-hidden-db-b10
)
echo "[info] ${#CHECKPOINTS[@]} checkpoints x $(echo $SEEDS | wc -w) seeds"

# ----------------------------- driver -------------------------------------
RAN=0; SKIPPED=0; MISSING=0
START_TS=$(date +%s)

throttle () {
  while [[ "$(jobs -r -p | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    wait -n 2>/dev/null || sleep 5
  done
}

# map a checkpoint name -> its reward family (base is shared within a family).
# osim/fexo/peri = distinct guacamol benchmarks; nitrogen + stability distinct.
reward_family () {
  case "$1" in
    *-osim-*)   echo "osim" ;;
    *-fexo-*)   echo "fexo" ;;
    *-peri-*)   echo "peri" ;;
    *-nitrogen-*) echo "nitrogen" ;;
    stability-geom-*) echo "stability" ;;
    *) echo "unknown" ;;
  esac
}

# find the ckpt file for a given model name (last.ckpt symlink or newest *.ckpt)
resolve_ckpt () {
  local name="$1"
  local c="${CKPT_ROOT}/${name}/checkpoints/${CKPT_FILE}"
  if [[ -f "$c" ]]; then echo "$c"; return 0; fi
  local alt; alt=$(ls -t "${CKPT_ROOT}/${name}/checkpoints/"*.ckpt 2>/dev/null | head -1)
  [[ -n "$alt" ]] && { echo "$alt"; return 0; }
  return 1
}

# launch one final_dump invocation in the background with tee/tag logging.
# args: TAG OUT CMD
launch () {
  local TAG="$1" OUT="$2" CMD="$3"
  throttle
  (
    stdbuf -oL -eL bash -c "$CMD" 2>&1 \
      | stdbuf -oL tee "${OUT}/dump.log" \
      | stdbuf -oL tr '\r' '\n' \
      | stdbuf -oL sed "s#^#[${TAG}] #" \
      >> "$MAIN_LOG"
    RC=${PIPESTATUS[0]}
    [[ $RC -ne 0 ]] && echo "[warn] ${TAG} exited $RC (see ${OUT}/dump.log)" | tee -a "$MAIN_LOG"
  ) &
  echo "[launch] ${TAG} (pid $!, active=$(( $(jobs -r -p | wc -l) )))" | tee -a "$MAIN_LOG"
  RAN=$((RAN+1)); sleep 2
}

REF_ARG=""
[[ -f "$REF_SMILES" ]] && REF_ARG="--ref_smiles ${REF_SMILES} --ref_limit ${REF_LIMIT}"

# ============================ PHASE 1: BASE ================================
# One base dump per (reward_family, seed), using a REPRESENTATIVE checkpoint of
# that family (base is the frozen prior -> identical across guides; we only need
# any checkpoint of the family to load the right reward config). Written to
# ${OUT_ROOT}/_base/<family>/seed<k>. Guided runs point --base_from at these.
echo "[phase] BASE (one per reward-family x seed)" | tee -a "$MAIN_LOG"
declare -A BASE_DIR      # "family,seed" -> dir

for name in "${CHECKPOINTS[@]}"; do
  fam=$(reward_family "$name")
  for seed in $SEEDS; do
    key="${fam},${seed}"
    [[ -n "${BASE_DIR[$key]:-}" ]] && continue     # already scheduled this family+seed
    OUT="${OUT_ROOT}/_base/${fam}/seed${seed}"
    BASE_DIR[$key]="$OUT"
    if [[ -f "${OUT}/dump_summary.json" ]]; then
      echo "[skip-base] ${fam} seed${seed} (exists)"; SKIPPED=$((SKIPPED+1)); continue
    fi
    CKPT=$(resolve_ckpt "$name") || { echo "[MISSING] $name"; MISSING=$((MISSING+1)); continue; }
    mkdir -p "$OUT"
    GPU=$(( RAN % NUM_GPUS ))
    CMD="CUDA_VISIBLE_DEVICES=${GPU} python final_dump.py \
      --ckpt ${CKPT} --n ${N} --seed ${seed} --skip_guided \
      --diff_steps ${DIFF_STEPS} --dataset ${DATASET} --progress \
      ${REF_ARG} --out_dir ${OUT}"
    [[ "$DRY_RUN" == "1" ]] && { echo "[base] ${fam} s${seed}: $CMD"; continue; }
    launch "BASE|${fam}|s${seed}|gpu${GPU}" "$OUT" "$CMD"
  done
done
wait   # ALL base dumps must finish before guided runs can --base_from them
echo "[phase] BASE done" | tee -a "$MAIN_LOG"

# ============================ PHASE 2: GUIDED =============================
# Every checkpoint, guided-only, reusing the matching family+seed base dump.
echo "[phase] GUIDED (all checkpoints, reusing base)" | tee -a "$MAIN_LOG"
for name in "${CHECKPOINTS[@]}"; do
  CKPT=$(resolve_ckpt "$name") || { echo "[MISSING] $name"; MISSING=$((MISSING+1)); continue; }
  fam=$(reward_family "$name")
  for seed in $SEEDS; do
    OUT="${OUT_ROOT}/${name}/seed${seed}"
    if [[ -f "${OUT}/dump_summary.json" ]]; then
      echo "[skip] $name seed$seed (exists)"; SKIPPED=$((SKIPPED+1)); continue
    fi
    BASE="${BASE_DIR[${fam},${seed}]:-${OUT_ROOT}/_base/${fam}/seed${seed}}"
    BASE_ARG=""
    if [[ -f "${BASE}/dump_summary.json" ]]; then
      BASE_ARG="--base_from ${BASE}"
    else
      echo "[warn] no base dump for ${fam} seed${seed}; ${name} will regenerate base" \
        | tee -a "$MAIN_LOG"
    fi
    mkdir -p "$OUT"
    GPU=$(( RAN % NUM_GPUS ))
    CMD="CUDA_VISIBLE_DEVICES=${GPU} python final_dump.py \
      --ckpt ${CKPT} --n ${N} --seed ${seed} --guide_source ${GUIDE_SOURCE} \
      --diff_steps ${DIFF_STEPS} --dataset ${DATASET} --progress \
      ${BASE_ARG} ${REF_ARG} --out_dir ${OUT}"
    [[ "$DRY_RUN" == "1" ]] && { echo "[run] $name s${seed}: $CMD"; continue; }
    launch "${name}|s${seed}|gpu${GPU}" "$OUT" "$CMD"
  done
done

wait

END_TS=$(date +%s)
{
  echo ""
  echo "==================================================================="
  echo "dumps done: ran=$RAN skipped=$SKIPPED missing=$MISSING  elapsed=$(( (END_TS-START_TS)/60 )) min"
  echo "==================================================================="
} | tee -a "$MAIN_LOG"

# ----------------------------- aggregate ----------------------------------
if [[ "$DRY_RUN" != "1" ]]; then
  echo "[aggregate] building master CSV + cross-model plots ..." | tee -a "$MAIN_LOG"
  python aggregate_dumps.py --dumps_root "${OUT_ROOT}" --out_dir "${OUT_ROOT}/_aggregate" \
    2>&1 | stdbuf -oL sed 's#^#[aggregate] #' | tee -a "$MAIN_LOG"
fi
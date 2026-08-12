#!/usr/bin/env bash
# ==========================================================================
# Run the logit-FLIP ablation (mechanistic ceiling test) on every single-guide
# checkpoint, at temp 1.0 AND 0.3, then collect into flips/_flip_table.csv.
#
# This is the DIRECT ceiling measurement: for each guide it reports whether the
# residual reaches the logits (delivered), whether it flips decisions (argmax /
# sample flip), and CRUCIALLY the flip rate on HIGH-gap (>8) vs LOW-gap decisions
# -- the ceiling predicts ~0 flips on high-gap states regardless of guide.
#
# Cheaper than a full dump (no molecules scored, ~400 trajectories), so this can
# run over all 68 checkpoints quickly.
# ==========================================================================
set -u

CKPT_ROOT="${CKPT_ROOT:-logs/quetzal-gfn}"
OUT_ROOT="${OUT_ROOT:-flips}"
N_TRAJ="${N_TRAJ:-400}"
TEMP="${TEMP:-1.0}"
ALSO_TEMP="${ALSO_TEMP:-0.3}"
DIFF_STEPS="${DIFF_STEPS:-18}"
GUIDE_SOURCE="${GUIDE_SOURCE:-ema}"
CKPT_FILE="${CKPT_FILE:-last.ckpt}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
MAIN_LOG="${MAIN_LOG:-flip_master.log}"
: > "$MAIN_LOG"

# all single-guide checkpoints (same list as dump_all_checkpoints.sh)
mapfile -t CHECKPOINTS < <(ls "${CKPT_ROOT}" | grep -E '^(sweep|stability)-' | sort)
echo "[flip] ${#CHECKPOINTS[@]} checkpoints x 2 temps" | tee -a "$MAIN_LOG"

throttle () { while [[ "$(jobs -r -p | wc -l)" -ge "$MAX_PARALLEL" ]]; do wait -n 2>/dev/null || sleep 3; done; }
RAN=0; SKIPPED=0; MISSING=0

for name in "${CHECKPOINTS[@]}"; do
  CKPT="${CKPT_ROOT}/${name}/checkpoints/${CKPT_FILE}"
  [[ -f "$CKPT" ]] || CKPT=$(ls -t "${CKPT_ROOT}/${name}/checkpoints/"*.ckpt 2>/dev/null | head -1)
  [[ -f "$CKPT" ]] || { echo "[MISSING] $name" | tee -a "$MAIN_LOG"; MISSING=$((MISSING+1)); continue; }

  OUT="${OUT_ROOT}/${name}"
  if [[ -f "${OUT}/flip_report.json" ]]; then
    echo "[skip] $name (flip_report.json exists)" | tee -a "$MAIN_LOG"; SKIPPED=$((SKIPPED+1)); continue
  fi
  mkdir -p "$OUT"

  CMD="python flip_ablation.py --ckpt ${CKPT} --n_traj ${N_TRAJ} \
    --flip_temp ${TEMP} --also_temp ${ALSO_TEMP} --diff_steps ${DIFF_STEPS} \
    --guide_source ${GUIDE_SOURCE} --progress --out_dir ${OUT}"

  echo "[run] $name" | tee -a "$MAIN_LOG"
  [[ "$DRY_RUN" == "1" ]] && { echo "$CMD"; continue; }

  throttle
  GPU=$(( RAN % NUM_GPUS ))
  TAG="${name}|gpu${GPU}"
  (
    CUDA_VISIBLE_DEVICES="$GPU" stdbuf -oL -eL bash -c "$CMD" 2>&1 \
      | stdbuf -oL tee "${OUT}/flip.log" \
      | stdbuf -oL tr '\r' '\n' \
      | stdbuf -oL sed "s#^#[${TAG}] #" >> "$MAIN_LOG"
  ) &
  echo "[launch] ${TAG} (pid $!)" | tee -a "$MAIN_LOG"
  RAN=$((RAN+1)); sleep 1
done
wait
echo "[flip] done: ran=$RAN skipped=$SKIPPED missing=$MISSING" | tee -a "$MAIN_LOG"

# ---- aggregate flip reports into a table ----
[[ "$DRY_RUN" == "1" ]] || python aggregate_flips.py --flips_root "${OUT_ROOT}" \
  --out "${OUT_ROOT}/_flip_table.csv" 2>&1 | tee -a "$MAIN_LOG"
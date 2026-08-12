#!/usr/bin/env bash
#
# warmstart_osim_quetzal.sh -- fine-tune the TempGainGuide (learned prior-temperature
# + residual-gain) on top of each component's ALREADY-TRAINED (pre-patch) residual.
#
# WHY warm-start instead of retrain-from-scratch:
#   The residual guides already learned a (weak-but-correct) direction. The new
#   temp/gain heads attack the SATURATED-PRIOR CEILING -- they let the guide act on
#   the ~72% of decisions where the prior gap > 8 (which the plain residual can't
#   touch). Loading the trained residual and fine-tuning only needs the temp/gain
#   heads to move, so it's fast and isolates the ceiling fix from the residual.
#
# The base residual is loaded from a PRE-patch LogitGuide checkpoint via
# --warm_start_guide; temp/gain start at identity (T=1, g=1) so training begins at
# exactly the old behavior and departs as the heads learn.
#
# Also applies the OFF-TARGET fix from the rollout diagnostics (train temp/eps too
# high -> trained off-target): sample_temp 2.0->1.0, rand_eps 0.2->0.05.
#
# Usage:
#   bash warmstart_osim_quetzal.sh
#   COMPONENTS="0 2 3" bash warmstart_osim_quetzal.sh    # skip dead c1
#   FROM_TAG=db-beta20 bash warmstart_osim_quetzal.sh    # which ckpts to warm-start FROM
#   DRY=1 bash warmstart_osim_quetzal.sh
#   nohup bash warmstart_osim_quetzal.sh > warmstart.log 2>&1 &
#
set -uo pipefail

SCRIPT="${SCRIPT:-gflow.py}"
PY="${PYTHON:-python}"
CKPT_ROOT="${CKPT_ROOT:-logs/quetzal-gfn}"

# ---- what to train ----
BENCHMARK="${BENCHMARK:-osimertinib}"
# c1 is a confirmed DEAD AXIS (std_logr_valid=0) -> excluded by default. Add it
# back only if you have reason to.
COMPONENTS="${COMPONENTS:-2 3}"
OBJECTIVE="${OBJECTIVE:-db}"

# ---- which existing runs to warm-start FROM ----
# base residual is read from: $CKPT_ROOT/gfn-geom-osim-comp{c}--$FROM_TAG/checkpoints/last.ckpt
FROM_TAG="${FROM_TAG:-db-beta20-t1.0-e0.05}"
WARM_SOURCE="${WARM_SOURCE:-ema}"          # ema | policy

# ---- ceiling-fix architecture flags (default ON) ----
USE_PRIOR_TEMP="${USE_PRIOR_TEMP:-1}"      # 1 -> --use_prior_temp
USE_RESIDUAL_GAIN="${USE_RESIDUAL_GAIN:-1}"
TEMPGAIN_HIDDEN="${TEMPGAIN_HIDDEN:-128}"

# ---- config (matches the corrected training regime) ----
BETA="${BETA:-20}"
SAMPLE_TEMP="${SAMPLE_TEMP:-1.0}"          # was 2.0 (off-target fix)
RAND_EPS="${RAND_EPS:-0.05}"              # was 0.2
INVALID_LOGR="${INVALID_LOGR:--5.0}"
DB_TARGET_CLIP="${DB_TARGET_CLIP:--6.0}"
DB_INTERIOR_WEIGHT="${DB_INTERIOR_WEIGHT:-1.0}"

# fine-tuning is short: temp/gain are small heads over a trained residual.
MAX_EPOCHS="${MAX_EPOCHS:-4}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
LR="${LR:-3e-4}"                           # a touch higher: only small heads move
GRAD_CLIP="${GRAD_CLIP:-1.0}"
BSZ="${BSZ:-128}"
DEVICES="${DEVICES:-1}"
QUETZAL_CKPT="${QUETZAL_CKPT:-geom.ckpt}"
TRAIN_MODULE="${TRAIN_MODULE:-train.py}"

EVAL_N="${EVAL_N:-2000}"
FINAL_N="${FINAL_N:-0}"
HIST_EVERY="${HIST_EVERY:-2}"
TAG="${TAG:-tempgain-ws-from-${FROM_TAG}-t${SAMPLE_TEMP}-e${RAND_EPS}}"
DRY="${DRY:-0}"

echo "[config] benchmark=$BENCHMARK objective=$OBJECTIVE components='$COMPONENTS'"
echo "[config] warm-start FROM tag=$FROM_TAG source=$WARM_SOURCE"
echo "[config] temp=$SAMPLE_TEMP eps=$RAND_EPS beta=$BETA lr=$LR epochs=$MAX_EPOCHS"
echo "[config] use_prior_temp=$USE_PRIOR_TEMP use_residual_gain=$USE_RESIDUAL_GAIN tag=$TAG"

run_one () {
  local comp="$1"
  local from_ckpt="${CKPT_ROOT}/gfn-geom-osim-comp${comp}--${FROM_TAG}/checkpoints/last.ckpt"
  local name="gfn-geom-osim-comp${comp}--${TAG}"

  if [[ ! -f "$from_ckpt" && "$DRY" != "1" ]]; then
    echo "    [SKIP] c${comp}: warm-start ckpt not found: $from_ckpt"
    return 0
  fi

  echo ">>> [$(date '+%H:%M:%S')] c${comp}: warm-start from $from_ckpt -> $name"

  local args=(
    --name "$name"
    --quetzal_ckpt "$QUETZAL_CKPT"
    --train_module "$TRAIN_MODULE"
    --objective "$OBJECTIVE"
    --reward guacamol_component
    --reward_benchmark "$BENCHMARK"
    --reward_component "$comp"
    --reward_beta "$BETA"
    --sample_temp "$SAMPLE_TEMP"
    --rand_eps "$RAND_EPS"
    --invalid_logr "$INVALID_LOGR"
    --db_target_clip "$DB_TARGET_CLIP"
    --db_interior_weight "$DB_INTERIOR_WEIGHT"
    --warm_start_guide "$from_ckpt"
    --warm_start_source "$WARM_SOURCE"
    --tempgain_hidden "$TEMPGAIN_HIDDEN"
    --max_epochs "$MAX_EPOCHS"
    --steps_per_epoch "$STEPS_PER_EPOCH"
    --lr "$LR"
    --grad_clip "$GRAD_CLIP"
    --bsz "$BSZ"
    --devices "$DEVICES"
    --eval_n "$EVAL_N"
    --final_n "$FINAL_N"
    --hist_every_n_epochs "$HIST_EVERY"
  )
  # boolean flags: config defaults are True, so only ADD the --no_ form when disabled
  [[ "$USE_PRIOR_TEMP" == "0" ]]    && args+=(--no_use_prior_temp)
  [[ "$USE_RESIDUAL_GAIN" == "0" ]] && args+=(--no_use_residual_gain)

  if [[ "$DRY" == "1" ]]; then
    printf '    %s %s' "$PY" "$SCRIPT"; printf ' %q' "${args[@]}"; printf '\n'
    return 0
  fi
  "$PY" "$SCRIPT" "${args[@]}"
  local rc=$?
  if [[ $rc -eq 0 ]]; then echo "    [done] $name"
  else echo "    [FAILED] $name (rc=$rc)"; fi
}

for c in $COMPONENTS; do
  run_one "$c"
done

echo "=============================================================="
echo " warm-start finished. Confirm the ceiling actually broke:"
echo "   bash run_ablations.sh ceiling   # flip_rate in gap>8 bins should now be > 0"
echo "   bash run_ablations.sh flip      # sample_flip_rate should rise vs pre-warmstart"
echo " Then compose (drop dead c1):"
echo "   compose.py --guide_ckpts <the new comp0/2/3 ckpts> --route flow ..."
echo "=============================================================="
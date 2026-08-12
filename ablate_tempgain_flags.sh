#!/usr/bin/env bash
#
# ablate_tempgain_flags.sh -- Category 2: which piece of the fix does the work?
#
# Warm-starts each component under FOUR flag combinations, everything else held
# identical, so any difference is attributable to the temp/gain mechanism:
#
#     neither  : plain residual (baseline)                 --no_use_prior_temp --no_use_residual_gain
#     temp     : temperature only (softens the prior)      --use_prior_temp   --no_use_residual_gain
#     gain     : gain only (amplifies residual)            --no_use_prior_temp --use_residual_gain
#     both     : full fix                                  --use_prior_temp   --use_residual_gain
#
# Hypothesis: TEMPERATURE is load-bearing (opens the high-gap window); GAIN alone
# just rediscovers the residual-scale validity cliff. This ablation confirms/refutes.
#
# All four warm-start from the SAME pre-patch residual so the only variable is the
# ceiling mechanism.
#
# Usage:
#   nohup bash ablate_tempgain_flags.sh > tmp_ablate.log 2>&1 &
#   COMPONENTS="3" bash ablate_tempgain_flags.sh          # single component, all 4 variants
#   VARIANTS="neither both" bash ablate_tempgain_flags.sh # just the two endpoints
#   DRY=1 bash ablate_tempgain_flags.sh
#   nohup bash ablate_tempgain_flags.sh > ablate_flags.log 2>&1 &
#
set -uo pipefail

SCRIPT="${SCRIPT:-gflow.py}"
PY="${PYTHON:-python}"
CKPT_ROOT="${CKPT_ROOT:-logs/quetzal-gfn}"

BENCHMARK="${BENCHMARK:-osimertinib}"
COMPONENTS="${COMPONENTS:-0 2 3}"          # c1 dead axis, skipped
VARIANTS="${VARIANTS:-neither temp gain both}"

FROM_TAG="${FROM_TAG:-tempgain-ws-from-db-beta20-t1.0-e0.05-t1.0-e0.05}"
WARM_SOURCE="${WARM_SOURCE:-ema}"

# config held identical across variants
BETA="${BETA:-20}"
SAMPLE_TEMP="${SAMPLE_TEMP:-1.0}"
RAND_EPS="${RAND_EPS:-0.05}"
INVALID_LOGR="${INVALID_LOGR:--5.0}"
DB_TARGET_CLIP="${DB_TARGET_CLIP:--6.0}"
TEMPGAIN_HIDDEN="${TEMPGAIN_HIDDEN:-128}"
MAX_EPOCHS="${MAX_EPOCHS:-4}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
LR="${LR:-3e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
BSZ="${BSZ:-128}"
DEVICES="${DEVICES:-1}"
QUETZAL_CKPT="${QUETZAL_CKPT:-geom.ckpt}"
TRAIN_MODULE="${TRAIN_MODULE:-train.py}"
EVAL_N="${EVAL_N:-2000}"
FINAL_N="${FINAL_N:-2000}"
HIST_EVERY="${HIST_EVERY:-2}"
DRY="${DRY:-0}"

echo "[config] benchmark=$BENCHMARK components='$COMPONENTS' variants='$VARIANTS'"
echo "[config] warm-start from=$FROM_TAG beta=$BETA temp=$SAMPLE_TEMP eps=$RAND_EPS"

# map variant -> the two boolean flags
flags_for () {
  case "$1" in
    neither) echo "--no_use_prior_temp --no_use_residual_gain" ;;
    temp)    echo "--use_prior_temp --no_use_residual_gain" ;;
    gain)    echo "--no_use_prior_temp --use_residual_gain" ;;
    both)    echo "--use_prior_temp --use_residual_gain" ;;
    *) echo "[ERROR] unknown variant $1" >&2; return 1 ;;
  esac
}

run_one () {
  local comp="$1" variant="$2"
  local from_ckpt="${CKPT_ROOT}/gfn-geom-osim-comp${comp}--${FROM_TAG}/checkpoints/last.ckpt"
  local name="gfn-geom-osim-comp${comp}--tgflag-${variant}"
  local vflags; vflags="$(flags_for "$variant")" || return 1

  if [[ ! -f "$from_ckpt" && "$DRY" != "1" ]]; then
    echo "    [SKIP] c${comp}/${variant}: warm-start ckpt not found: $from_ckpt"
    return 0
  fi
  echo ">>> [$(date '+%H:%M:%S')] c${comp} variant=${variant} -> $name"

  local args=(
    --name "$name"
    --quetzal_ckpt "$QUETZAL_CKPT"
    --train_module "$TRAIN_MODULE"
    --objective db
    --reward guacamol_component
    --reward_benchmark "$BENCHMARK"
    --reward_component "$comp"
    --reward_beta "$BETA"
    --sample_temp "$SAMPLE_TEMP"
    --rand_eps "$RAND_EPS"
    --invalid_logr "$INVALID_LOGR"
    --db_target_clip "$DB_TARGET_CLIP"
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
    $vflags
  )
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
  for v in $VARIANTS; do
    run_one "$c" "$v"
  done
done

echo "=============================================================="
echo " flag ablation finished. Compare with the probe + ceiling ablation:"
echo "   python probe_tempgain.py --guide_ckpts <the 4 variants' ckpts> ..."
echo "   bash run_ablations.sh ceiling   # flip_rate_by_gap: temp>0 in high-gap bins?"
echo " Expectation: 'temp' and 'both' break the ceiling; 'gain' alone does not"
echo " (and may drop validity -- check probe_tempgain [B])."
echo "=============================================================="
#!/usr/bin/env bash
#
# baseline_loss_compare.sh -- ABLATE THE LOSS, hold the guidance mechanism fixed.
#
# Trains the SAME logit-injection guide (prior + guide(h), plain residual, NO
# temp/gain) under three losses:
#     db      : detailed balance (your main objective)
#     revkl   : reverse KL  KL(q || p*)   -- mode-seeking, on-policy REINFORCE
#     fwdkl   : forward KL  KL(p* || q)   -- mass-covering, importance-weighted
# all targeting the SAME tilted terminal distribution p*(x) ~ p_prior * R^beta.
#
# PURPOSE: show the SATURATED-PRIOR CEILING is a property of logit-residual
# guidance, NOT of the DB objective. If revkl/fwdkl guides ALSO only flip the
# low-gap ~18% of decisions (run ablate_ceiling/flip on them), the ceiling
# generalizes across losses -> strengthens the architectural claim, and makes the
# temp/gain fix the thing that actually breaks it.
#
# IMPORTANT: this trains with temp/gain OFF (--no_use_prior_temp --no_use_residual_gain)
# so it's the PLAIN residual guide under each loss -- the honest baseline. The
# temp/gain runs (warmstart_osim_quetzal.sh) are the comparison point.
#
# Usage:
#   bash baseline_loss_compare.sh                 # all 3 losses, c0 c2 c3
#   LOSSES="db revkl" bash baseline_loss_compare.sh
#   COMPONENTS="3" bash baseline_loss_compare.sh  # single component, all losses
#   DRY=1 bash baseline_loss_compare.sh
#   nohup bash baseline_loss_compare.sh > baseline_compare.log 2>&1 &
#
set -uo pipefail

SCRIPT="${SCRIPT:-gflow.py}"
PY="${PYTHON:-python}"

BENCHMARK="${BENCHMARK:-osimertinib}"
COMPONENTS="${COMPONENTS:-0 2 3}"          # c1 is a dead axis; skip by default
LOSSES="${LOSSES:-revkl fwdkl}"        # which objectives to compare

# config held IDENTICAL across losses so only the objective differs
BETA="${BETA:-20}"
SAMPLE_TEMP="${SAMPLE_TEMP:-1.0}"
RAND_EPS="${RAND_EPS:-0.05}"
INVALID_LOGR="${INVALID_LOGR:--5.0}"
DB_TARGET_CLIP="${DB_TARGET_CLIP:--6.0}"
MAX_EPOCHS="${MAX_EPOCHS:-6}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
LR="${LR:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
BSZ="${BSZ:-128}"
DEVICES="${DEVICES:-1}"
QUETZAL_CKPT="${QUETZAL_CKPT:-geom.ckpt}"
TRAIN_MODULE="${TRAIN_MODULE:-train.py}"
EVAL_N="${EVAL_N:-2000}"
FINAL_N="${FINAL_N:-0}"
HIST_EVERY="${HIST_EVERY:-2}"
DRY="${DRY:-0}"

echo "[config] benchmark=$BENCHMARK components='$COMPONENTS' losses='$LOSSES'"
echo "[config] beta=$BETA temp=$SAMPLE_TEMP eps=$RAND_EPS  (temp/gain OFF -> plain residual baseline)"

run_one () {
  local comp="$1" loss="$2"
  local name="gfn-geom-osim-comp${comp}--baseline-${loss}-beta${BETA}"
  echo ">>> [$(date '+%H:%M:%S')] c${comp} loss=${loss} -> $name"
  local args=(
    --name "$name"
    --quetzal_ckpt "$QUETZAL_CKPT"
    --train_module "$TRAIN_MODULE"
    --objective "$loss"
    --reward guacamol_component
    --reward_benchmark "$BENCHMARK"
    --reward_component "$comp"
    --reward_beta "$BETA"
    --sample_temp "$SAMPLE_TEMP"
    --rand_eps "$RAND_EPS"
    --invalid_logr "$INVALID_LOGR"
    --db_target_clip "$DB_TARGET_CLIP"
    --max_epochs "$MAX_EPOCHS"
    --steps_per_epoch "$STEPS_PER_EPOCH"
    --lr "$LR"
    --grad_clip "$GRAD_CLIP"
    --bsz "$BSZ"
    --devices "$DEVICES"
    --eval_n "$EVAL_N"
    --final_n "$FINAL_N"
    --hist_every_n_epochs "$HIST_EVERY"
    # PLAIN residual guide: temp/gain OFF so this is the honest baseline
    --no_use_prior_temp
    --no_use_residual_gain
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
  for loss in $LOSSES; do
    run_one "$c" "$loss"
  done
done

echo "=============================================================="
echo " baseline loss comparison finished. Now show the ceiling is loss-INDEPENDENT:"
echo "   # point ablate_ceiling/flip at each loss's guides and compare flip_rate_by_gap."
echo "   # expectation: db, revkl, fwdkl ALL flat in the gap>8 bins (ceiling holds"
echo "   # across losses); only the temp/gain warm-start breaks it."
echo "=============================================================="
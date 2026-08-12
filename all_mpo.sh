#!/usr/bin/env bash
#
# reproduce_osim_good.sh -- reproduce the KNOWN-GOOD full-benchmark runs
# (the config that reached FCD ~30 and steered strongly) for Osimertinib,
# Fexofenadine, and Perindopril, to (a) recover reproducible strong baselines
# and (b) isolate config-drift from the TempGainGuide wrapper.
#
# Old good config (bare LogitGuide, pre-tempgain-patch):
#     objective = db (assumed; set OBJ=rtb if the old run was RTB)
#     reward_beta = 10          (NOT 20 -- the diagnostics-lowered value overshot)
#     sample_temp = 2.0         (NOT 1.0 -- exploration helped the full task)
#     lr = 1e-4, logz_lr = 1e-2
#     14 epochs x 100 steps = 1400 grad steps   (NOT 8)
#
# tempgain heads DISABLED so guided_logits == prior + residual, matching the
# old bare-guide behaviour exactly. This isolates:
#   * if this recovers FCD~30  -> the regression was pure CONFIG DRIFT
#   * if it does NOT            -> the wrapper changes things even when "off"
#
# Runs sequentially for each benchmark in BENCHES.
#
# Usage:
#   bash reproduce_osim_good.sh
#   BENCHES="osim" bash reproduce_osim_good.sh          # just one
#   OBJ=rtb bash reproduce_osim_good.sh                 # if the old run was RTB
#   WITH_TEMPGAIN=1 bash reproduce_osim_good.sh         # same config but heads ON (A/B)
#   DRY=1 bash reproduce_osim_good.sh
#   nohup bash all_mpo.sh > repro.log 2>&1 &
#
set -uo pipefail

SCRIPT="${SCRIPT:-gflow.py}"
PY="${PYTHON:-python}"
QUETZAL_CKPT="${QUETZAL_CKPT:-geom.ckpt}"
TRAIN_MODULE="${TRAIN_MODULE:-train.py}"

# ---- which benchmarks (friendly key -> raw sb fn name for reward_smiles) ----
BENCHES="${BENCHES:-osim fexo peri}"
declare -A REWARD_SMILES_FOR=(
  [osim]="hard_osimertinib"
  [fexo]="hard_fexofenadine"
  [peri]="perindopril_rings"
)

# ---- the known-good config ----
OBJECTIVES="${OBJECTIVES:-rtb}"  # space-separated; runs each. Set to "db" for just DB.
BETA="${BETA:-10}"                # old good value
SAMPLE_TEMP="${SAMPLE_TEMP:-2.0}" # old good value
RAND_EPS="${RAND_EPS:-0.2}"       # old good value (paired with temp=2)
LR="${LR:-1e-4}"
LOGZ_LR="${LOGZ_LR:-1e-2}"
MAX_EPOCHS="${MAX_EPOCHS:-8}"    # 1400 grad steps
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
BSZ="${BSZ:-512}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
INVALID_LOGR="${INVALID_LOGR:--5.0}"

# ---- eval / hist / final ----
EVAL_N="${EVAL_N:-2000}"
HIST_EVERY="${HIST_EVERY:-7}"
VIS_EVERY="${VIS_EVERY:-7}"
FINAL_N="${FINAL_N:-0}"        # dump molecules so you can recompute FCD/SA

# ---- tempgain toggle (default OFF to match the bare old guide) ----
WITH_TEMPGAIN="${WITH_TEMPGAIN:-0}"
DRY="${DRY:-0}"

tag_tg="baregt"
tempgain_flags=(--no_use_prior_temp --no_use_residual_gain)
if [[ "$WITH_TEMPGAIN" == "1" ]]; then
  tag_tg="tempgain"
  tempgain_flags=(--use_prior_temp --use_residual_gain)
fi

echo "[config] reproduce old-good: objectives='$OBJECTIVES' beta=$BETA temp=$SAMPLE_TEMP eps=$RAND_EPS"
echo "[config] epochs=$MAX_EPOCHS (x$STEPS_PER_EPOCH) lr=$LR logz_lr=$LOGZ_LR tempgain=$WITH_TEMPGAIN"
echo "[config] benchmarks: $BENCHES"

run_one () {
  local key="$1"
  local obj="$2"
  local reward_smiles="${REWARD_SMILES_FOR[$key]:-}"
  if [[ -z "$reward_smiles" ]]; then
    echo "    [SKIP] unknown benchmark key '$key' (known: ${!REWARD_SMILES_FOR[*]})"
    return 0
  fi
  local name="gfn-quetzal-${key}-repro-${obj}-beta${BETA}-t${SAMPLE_TEMP}-e${MAX_EPOCHS}-${tag_tg}"
  echo ">>> [$(date '+%H:%M:%S')] $key/$obj -> $name  (reward_smiles=$reward_smiles)"
  local args=(
    --name "$name"
    --quetzal_ckpt "$QUETZAL_CKPT"
    --train_module "$TRAIN_MODULE"
    --objective "$obj"
    --reward guacamol
    --reward_smiles "$reward_smiles"
    --reward_beta "$BETA"
    --sample_temp "$SAMPLE_TEMP"
    --rand_eps "$RAND_EPS"
    --invalid_logr "$INVALID_LOGR"
    --lr "$LR"
    --logz_lr "$LOGZ_LR"
    --max_epochs "$MAX_EPOCHS"
    --steps_per_epoch "$STEPS_PER_EPOCH"
    --bsz "$BSZ"
    --grad_clip "$GRAD_CLIP"
    --eval_n "$EVAL_N"
    --hist_every_n_epochs "$HIST_EVERY"
    --vis_every_n_epochs "$VIS_EVERY"
    --final_n "$FINAL_N"
    --devices 1
    "${tempgain_flags[@]}"
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

for obj in $OBJECTIVES; do
  for b in $BENCHES; do
    run_one "$b" "$obj"
  done
done

echo "=============================================================="
echo " done. Compare each to its old good run:"
echo "   * wandb: eval/log_reward_top10, eval_delta/log_reward_mean, fcd/guided_vs_base"
echo "   * expect FCD ~30 + large positive reward delta if config drift was the cause."
echo " Then A/B the wrapper:  WITH_TEMPGAIN=1 bash reproduce_osim_good.sh"
echo "=============================================================="
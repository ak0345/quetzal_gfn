#!/usr/bin/env bash
#
# train_osim_quetzal.sh -- retrain the 4 Osimertinib component guides on top of
# frozen Quetzal, one at a time, with config informed by the ablation diagnostics.
#
# WHY these settings (from the diagnostics):
#   * beta 50 -> 20 (or lower). At beta=50 the DB terminal target beta*logR hit
#     magnitude 1e4-1e5, unfittable by the flow head (terminal_loss never dropped,
#     reward never climbed). Lower beta shrinks the target into a learnable range.
#   * sample_temp 2.0 -> 1.0, rand_eps 0.2 -> 0.05. At 2.0/0.2 the guide trained on
#     a near-random, low-reward region (eval reward was +1 to +4 higher than the
#     training-settings reward -> trained off-target). Lower exploration keeps
#     training near the distribution it samples at eval.
#   * db_interior_weight up a bit. interior_over_terminal was ~0.005 (terminal
#     term 100-200x the interior), so the flow-matching structure was ignored.
#
# All knobs are env-overridable so you can sweep without editing.
#
# Usage:
#   bash train_osim_quetzal.sh                 # all 4 components, beta=20
#   BETA=10 bash train_osim_quetzal.sh         # try beta=10
#   OBJECTIVE=rtb bash train_osim_quetzal.sh   # RTB (policy route) instead of DB
#   COMPONENTS="0 3" bash train_osim_quetzal.sh  # only c0 and c3
#   DRY=1 bash train_osim_quetzal.sh           # print commands, run nothing
#   nohup bash train_osim_quetzal.sh > train_osim.log 2>&1 &
#
set -uo pipefail

SCRIPT="${SCRIPT:-gflow.py}"
PY="${PYTHON:-python}"

# ---- what to train ----
BENCHMARK="${BENCHMARK:-osimertinib}"
COMPONENTS="${COMPONENTS:-0 1 2 3}"       # space-separated component indices
OBJECTIVE="${OBJECTIVE:-db}"              # db (flow route) | rtb | vargrad

# ---- diagnostics-informed config ----
BETA="${BETA:-20}"                        # was 50; the big fix
SAMPLE_TEMP="${SAMPLE_TEMP:-1.0}"         # was 2.0
RAND_EPS="${RAND_EPS:-0.05}"             # was 0.2
DB_INTERIOR_WEIGHT="${DB_INTERIOR_WEIGHT:-1.0}"   # raise if terminal still dominates
INVALID_LOGR="${INVALID_LOGR:--5.0}"
DB_TARGET_CLIP="${DB_TARGET_CLIP:--6.0}"  # clamp very-negative valid targets

# ---- training length / infra ----
MAX_EPOCHS="${MAX_EPOCHS:-6}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
LR="${LR:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
BSZ="${BSZ:-128}"
DEVICES="${DEVICES:-1}"
QUETZAL_CKPT="${QUETZAL_CKPT:-geom.ckpt}"
TRAIN_MODULE="${TRAIN_MODULE:-train.py}"

# ---- naming / eval ----
# a short tag describing this config, folded into the run name so you can tell
# retrains apart in wandb (e.g. osim-comp0--db-beta20-t1.0-e0.05)
TAG="${TAG:-db-beta${BETA}-t${SAMPLE_TEMP}-e${RAND_EPS}}"
EVAL_N="${EVAL_N:-1000}"
FINAL_N="${FINAL_N:-0}"
HIST_EVERY="${HIST_EVERY:-5}"
DRY="${DRY:-0}"

echo "[config] benchmark=$BENCHMARK objective=$OBJECTIVE beta=$BETA temp=$SAMPLE_TEMP eps=$RAND_EPS"
echo "[config] db_interior_weight=$DB_INTERIOR_WEIGHT tag=$TAG components='$COMPONENTS'"

run_one () {
  local comp="$1"
  local name="gfn-geom-osim-comp${comp}--${TAG}"
  echo ">>> [$(date '+%H:%M:%S')] training $name"
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
  if [[ "$DRY" == "1" ]]; then
    printf '    %s %s' "$PY" "$SCRIPT"; printf ' %q' "${args[@]}"; printf '\n'
    return 0
  fi
  "$PY" "$SCRIPT" "${args[@]}"
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "    [done] $name"
  else
    echo "    [FAILED] $name (rc=$rc)"
  fi
}

for c in $COMPONENTS; do
  run_one "$c"
done

echo "=============================================================="
echo " all requested components finished."
echo " next: re-run the diagnostics on these new runs to confirm the fix:"
echo "   WANDB_ENTITY=you bash run_ablations.sh diag_logs   # terminal loss should be MUCH smaller"
echo "   bash run_ablations.sh diag_rollout                 # eval-minus-train reward should shrink"
echo "=============================================================="
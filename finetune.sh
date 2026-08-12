#!/usr/bin/env bash
# =============================================================================
# RTB fine-tuning of Quetzal itself -- sequential driver.
#
# Order:
#   0. sanity   : proj scope, nitrogen_count, 2 epochs. Dense reward. If this
#                 does not move, the training loop is broken and every
#                 osimertinib result below is uninterpretable. GATES THE REST.
#   1. proj     : atom-type only (proj_logits). RTB ratio EXACT. Headline run.
#   2. full     : whole model. Ratio approximate -- watch diag/zprefix_drift.
#   3. lora r=4,16,64 : capacity dial on proj_logits (optional, RUN_LORA=0 to skip)
#   4. harvest  : score every run as a goal-directed benchmark at a 10k budget
#
# Molecules stream to /oracle_gfn_mols/<name>/molecules.jsonl during training,
# so harvest_eval.py can be run at any point without retraining.
#
# Usage:
#   bash run_rtb_finetune.sh                    # everything
#   RUN_LORA=0 bash run_rtb_finetune.sh         # skip the LoRA sweep
#   SKIP_SANITY=1 bash run_rtb_finetune.sh      # sanity already passed
#   ONLY=rtb-proj-osim-b10 bash run_rtb_finetune.sh   # one run by name
#   DRY=1 bash run_rtb_finetune.sh              # print commands, run nothing
#
# Both training and harvesting are resumable: re-running skips completed runs
# (marker files under $MOLROOT/_state) and Lightning picks up the latest ckpt.
# =============================================================================

set -uo pipefail

# ------------------------------- knobs --------------------------------------
REPO="${REPO:-/workspace/quetzal}"
CKPT="${CKPT:-geom.ckpt}"
TRAIN_MODULE="${TRAIN_MODULE:-train.py}"
MOLROOT="${MOLROOT:-/workspace/oracle_gfn_mols}"

BENCH="${BENCH:-perindopril_rings}"
BETA="${BETA:-10}"
BETA_START="${BETA_START:-2}"
BETA_ANNEAL="${BETA_ANNEAL:-4}"

STEPS="${STEPS:-100}"
EPOCHS="${EPOCHS:-6}"
FULL_EPOCHS="${EPOCHS:-12}"
SANITY_EPOCHS="${SANITY_EPOCHS:-2}"

BSZ="${BSZ:-64}"
BSZ_ATOM="${BSZ_ATOM:-16}"
BSZ_FULL="${BSZ_FULL:-8}"     # halved: full scope keeps the trunk graph alive

BUDGET="${BUDGET:-10000}"
DEVICE="${DEVICE:-0}"          # CUDA_VISIBLE_DEVICES for the whole script

RUN_LORA="${RUN_LORA:-1}"
LORA_RANKS="${LORA_RANKS:-4 64}"
SKIP_SANITY="${SKIP_SANITY:-1}"
ONLY="${ONLY:-}"
DRY="${DRY:-0}"

# ------------------------------- setup --------------------------------------
export CUDA_VISIBLE_DEVICES="$DEVICE"
cd "$REPO" || { echo "FATAL: no repo at $REPO"; exit 1; }

STATE="$MOLROOT/_state"
LOGS="$MOLROOT/_logs"
RESULTS="$MOLROOT/_results"
mkdir -p "$STATE" "$LOGS" "$RESULTS"

MAIN_LOG="$LOGS/driver_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"; }

FAILED=()
COMPLETED=()

# run <name> <extra args...>
run() {
  local name="$1"; shift
  if [[ -n "$ONLY" && "$name" != "$ONLY" ]]; then
    say "SKIP  $name (ONLY=$ONLY)"; return 0
  fi
  if [[ -f "$STATE/$name.done" ]]; then
    say "SKIP  $name (already complete; rm $STATE/$name.done to redo)"
    COMPLETED+=("$name"); return 0
  fi

  local reclog="$LOGS/$name.log"
  say "START $name  -> $MOLROOT/$name  (log: $reclog)"

  local cmd=(python rtb_finetune.py
    --name "$name"
    --quetzal_ckpt "$CKPT"
    --train_module "$TRAIN_MODULE"
    --objective rtb
    --sample_temp 1.0 --rand_eps 0.0
    --logz_lr 1e-2 --grad_clip 1.0
    --record_dir "$MOLROOT/$name"
    "$@")

  if [[ "$DRY" == "1" ]]; then
    printf '  %q' "${cmd[@]}"; echo; return 0
  fi

  local t0=$SECONDS
  if "${cmd[@]}" 2>&1 | tee -a "$reclog"; then
    local n=0
    [[ -f "$MOLROOT/$name/molecules.jsonl" ]] && n=$(wc -l < "$MOLROOT/$name/molecules.jsonl")
    touch "$STATE/$name.done"
    say "DONE  $name  ($(( (SECONDS-t0)/60 ))m, $n molecules recorded)"
    COMPLETED+=("$name")
  else
    say "FAIL  $name  (see $reclog)"
    FAILED+=("$name")
    return 1
  fi
}

say "================================================================"
say "repo=$REPO ckpt=$CKPT gpu=$DEVICE"
say "bench=$BENCH beta=$BETA_START->$BETA over $BETA_ANNEAL ep"
say "epochs=$EPOCHS steps=$STEPS  |  proj bsz=$BSZ -> $((BSZ*STEPS)) mols/epoch"
say "                              full bsz=$BSZ_FULL -> $((BSZ_FULL*STEPS)) mols/epoch"
say "harvest budget=$BUDGET  (proj reaches it in epoch $(( (BUDGET+BSZ*STEPS-1)/(BSZ*STEPS) )), full in epoch $(( (BUDGET+BSZ_FULL*STEPS-1)/(BSZ_FULL*STEPS) )))"
say "molecules -> $MOLROOT/<name>/molecules.jsonl"
say "================================================================"

# ---------------------------- 0. sanity check --------------------------------
# Dense, atom-local, prior-aligned reward (>0.9 reward_valid_frac). Full-rank
# proj_logits under RTB SHOULD move train/log_reward_mean within an epoch. If it
# does not, the failure is in the loop, not in the saturated-prior ceiling, and
# the osimertinib runs below cannot be interpreted.
if [[ "$SKIP_SANITY" == "0" ]]; then
  run rtb-proj-nitrogen-sanity \
    --finetune_scope proj \
    --reward nitrogen_count \
    --reward_beta "$BETA" \
    --bsz "$BSZ" --steps_per_epoch "$STEPS" --max_epochs "$SANITY_EPOCHS" \
    --lr 1e-5

  if [[ ${#FAILED[@]} -gt 0 ]]; then
    say "ABORT: sanity run failed; not launching the osimertinib runs."
    exit 1
  fi
  if [[ "$DRY" != "1" ]]; then
    say "----------------------------------------------------------------"
    say "CHECK THE SANITY RUN BEFORE TRUSTING WHAT FOLLOWS:"
    say "  grep log_reward_mean $LOGS/rtb-proj-nitrogen-sanity.log | tail -20"
    say "  train/log_reward_mean should CLIMB; reward_valid_frac should be >0.9"
    say "----------------------------------------------------------------"
  fi
fi

# ------------------- 1. atom-type only (exact RTB ratio) ---------------------
#run rtb-proj-peri-b$BETA \
#  --finetune_scope proj \
#  --reward guacamol --reward_smiles "$BENCH" \
#  --reward_beta "$BETA" --beta_start "$BETA_START" --beta_anneal_epochs "$BETA_ANNEAL" \
#  --bsz "$BSZ" --steps_per_epoch "$STEPS" --max_epochs "$EPOCHS" \
#  --lr 1e-5

# ---------------- 2. whole model (approximate RTB ratio) ---------------------
# lower lr (pretrained weights already fit GEOM), trunk slower still, and
# logp_grad_frac keeps only a quarter of the token graph -- unbiased, see
# rtb_finetune.py docstring.
#run rtb-full-peri-b$BETA \
#  --finetune_scope full \
#  --reward guacamol --reward_smiles "$BENCH" \
#  --reward_beta "$BETA" --beta_start "$BETA_START" --beta_anneal_epochs "$BETA_ANNEAL" \
#  --bsz "$BSZ_FULL" --steps_per_epoch "$STEPS" --max_epochs "$FULL_EPOCHS" \
#  --lr 5e-6 --trunk_lr_mult 0.1 --logp_grad_frac 0.25


#run rtb-atom-peri-b$BETA \
#  --finetune_scope full \
#  --reward guacamol --reward_smiles "$BENCH" \
#  --reward_beta "$BETA" --beta_start "$BETA_START" --beta_anneal_epochs "$BETA_ANNEAL" \
#  --bsz "$BSZ_ATOM" --steps_per_epoch "$STEPS" --max_epochs "$FULL_EPOCHS" \
#  --lr 5e-6 --trunk_lr_mult 0.1 --logp_grad_frac 0.25

# ----------------------- 3. LoRA capacity dial -------------------------------
# rank r on proj_logits induces delta_logits = (alpha/r) B A h, i.e. exactly a
# rank-r linear guide. Sweeping r against the full-rank proj run gives steering
# vs capacity on an axis the norm-limited residual guide never explored.

if [[ "$RUN_LORA" == "1" ]]; then
  for R in $LORA_RANKS; do
    run rtb-proj-lora${R}-peri-b$BETA \
      --finetune_scope proj \
      --lora_rank "$R" --lora_targets proj_logits --lora_alpha 16 \
      --reward guacamol --reward_smiles "$BENCH" \
      --reward_beta "$BETA" --beta_start "$BETA_START" --beta_anneal_epochs "$BETA_ANNEAL" \
      --bsz "$BSZ" --steps_per_epoch "$STEPS" --max_epochs "$EPOCHS" \
      --lr 1e-4
  done
fi

# ---------------------------- 4. harvest -------------------------------------
say "================================================================"
if [[ "$DRY" == "1" ]]; then
  say "DRY run complete."
  exit 0
fi

DIRS=()
for d in "$MOLROOT"/rtb-*; do
  [[ -f "$d/molecules.jsonl" ]] && DIRS+=("$d")
done

if [[ ${#DIRS[@]} -eq 0 ]]; then
  say "no records found under $MOLROOT -- nothing to harvest"
else
  say "harvesting ${#DIRS[@]} run(s) at budget=$BUDGET against $BENCH"
  # the nitrogen sanity dir is included on purpose: its periertinib score is a
  # useful floor (a run never optimised for peri), not a result.
  python harvest_eval.py \
    --record_dir "${DIRS[@]}" \
    --bench "$BENCH" \
    --budget "$BUDGET" \
    --topk 1,10,100 \
    --out "$RESULTS/${BENCH}_budget${BUDGET}.json" \
    --csv "$RESULTS/${BENCH}_budget${BUDGET}.csv" \
    --dump_best "$RESULTS/best" 2>&1 | tee -a "$MAIN_LOG"
  say "results -> $RESULTS/${BENCH}_budget${BUDGET}.{json,csv}"
fi

# ---------------------------- summary ----------------------------------------
say "================================================================"
say "completed: ${#COMPLETED[@]}  ${COMPLETED[*]:-(none)}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  say "FAILED:    ${#FAILED[@]}  ${FAILED[*]}"
  say "re-run this script to retry only the failures"
  exit 1
fi
say "all runs complete"
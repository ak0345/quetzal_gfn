#!/usr/bin/env bash
# =============================================================================
# Stage 3 -- RTB fine-tuning of the prior's own weights.
#
# The capacity control for the guide sweep: if the bound were set by how small
# the guides are, updating Quetzal itself should relax it. Four scopes on one
# capacity axis, from a rank-4 adapter (3.6k parameters) to every weight in the
# model (85M):
#
#   proj   only proj_logits. encode1/encode2/the coordinate denoiser stay
#          frozen, so z_prefix is identical between policy and prior for a given
#          atom sequence, p(coords|atoms) cancels exactly, and the atom-only RTB
#          ratio is exact rather than approximate. The clean comparison.
#   atom   proj_logits plus the encode1 trunk. z_prefix now drifts and the ratio
#          becomes an approximation; diag/zprefix_drift logs how large.
#   full   everything, including the coordinate diffusion. Largest drift.
#   lora   a zero-init rank-r adapter on proj_logits, which induces
#          delta_logits = (alpha/r) B A h -- exactly a rank-r linear guide.
#
# Every molecule generated during training is appended in order to
# <record_dir>/molecules.jsonl, so stage 8 can score any run as a goal-directed
# benchmark at a fixed oracle budget without retraining.
#
# Runs are resumable: a completed run drops a marker under $MOLROOT/_state and
# is skipped on re-invocation.
#
# The sanity run comes first. It targets nitrogen fraction, which is dense
# (>0.9 of samples score above the invalid floor, against 0.02-0.06 for the
# assembled MPO objectives) and decomposes over exactly the decisions being
# updated. If it does not move, the loop is broken and nothing below can be
# interpreted -- so it gates the rest.
#
# Usage:
#   bash scripts/03_finetune.sh                       # sanity + osim + peri
#   bash scripts/03_finetune.sh osim
#   SKIP_SANITY=1 bash scripts/03_finetune.sh
#   ONLY=rtb-proj-osim-b10 bash scripts/03_finetune.sh
#   DRY=1 bash scripts/03_finetune.sh
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_prior

WHICH="${1:-all}"

MOLROOT="${MOLROOT:-${RESULTS_ROOT}/oracle_gfn_mols}"
SKIP_SANITY="${SKIP_SANITY:-0}"
ONLY="${ONLY:-}"
DEVICE="${DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="$DEVICE"

STATE="$MOLROOT/_state"
LOGS="$MOLROOT/_logs"
mkdir -p "$STATE" "$LOGS"

FAILED=(); COMPLETED=()

# run <name> <extra args...>
#
# Settings shared by every configuration: on-policy sampling at temperature 1
# with no epsilon-uniform mixing, so the recorded stream is a clean sample of
# the policy and can be scored as a goal-directed run without correction; and a
# logZ learning rate three orders of magnitude above the weight learning rate,
# because a lagging logZ offsets every trajectory's residual equally and
# displaces all log-probabilities in a common direction.
run () {
  local name="$1"; shift
  if [[ -n "$ONLY" && "$name" != "$ONLY" ]]; then return 0; fi
  if [[ -f "$STATE/$name.done" ]]; then
    say "SKIP  $name (complete; rm $STATE/$name.done to redo)"
    COMPLETED+=("$name"); return 0
  fi

  local cmd=("$PY" rtb_finetune.py
    --name "$name"
    --quetzal_ckpt "$QUETZAL_CKPT"
    --objective rtb
    --sample_temp 1.0 --rand_eps 0.0
    --logz_lr 1e-2 --grad_clip 1.0
    --record_dir "$MOLROOT/$name"
    "$@")

  if [[ "$DRY" == "1" ]]; then printf '  %q' "${cmd[@]}"; echo; return 0; fi

  say "START $name -> $MOLROOT/$name"
  local t0=$SECONDS
  if "${cmd[@]}" 2>&1 | tee -a "$LOGS/$name.log"; then
    local n=0
    [[ -f "$MOLROOT/$name/molecules.jsonl" ]] && n=$(wc -l < "$MOLROOT/$name/molecules.jsonl")
    touch "$STATE/$name.done"
    say "DONE  $name ($(( (SECONDS-t0)/60 ))m, $n molecules recorded)"
    COMPLETED+=("$name")
  else
    say "FAIL  $name (see $LOGS/$name.log)"
    FAILED+=("$name"); return 1
  fi
}

# reward_args <osim|peri> -- the assembled benchmark, via the guacamol passthrough
reward_args () {
  case "$1" in
    osim) echo "--reward guacamol --reward_smiles hard_osimertinib" ;;
    peri) echo "--reward guacamol --reward_smiles perindopril_rings" ;;
  esac
}

# ---------------------------- 0. sanity gate ---------------------------------
if [[ "$SKIP_SANITY" != "1" ]]; then
  say "sanity: proj scope on the dense nitrogen reward"
  run rtb-proj-nitrogen-sanity \
    --finetune_scope proj \
    --reward nitrogen_count \
    --reward_beta 10 --beta_start 1 --beta_anneal_epochs 0 \
    --bsz 64 --steps_per_epoch 100 --max_epochs 2 \
    --lr 1e-5

  if [[ ${#FAILED[@]} -gt 0 ]]; then
    say "ABORT: sanity run failed; not launching the benchmark runs."
    exit 1
  fi
  if [[ "$DRY" != "1" ]]; then
    say "check before trusting what follows: train/log_reward_mean should CLIMB"
    say "  grep log_reward_mean $LOGS/rtb-proj-nitrogen-sanity.log | tail -20"
  fi
fi

# --------------------------- 1. Osimertinib MPO ------------------------------
# beta anneals linearly from beta_start to reward_beta over beta_anneal_epochs.
if [[ "$WHICH" == "all" || "$WHICH" == "osim" ]]; then
  RA="$(reward_args osim)"

  # proj, full rank. The headline configuration: exact ratio, 98,304 parameters.
  # shellcheck disable=SC2086
  run rtb-proj-osim-b10 --finetune_scope proj $RA \
    --reward_beta 10 --beta_start 2 --beta_anneal_epochs 4 \
    --bsz 64 --steps_per_epoch 100 --max_epochs 10 --lr 1e-5

  # the one configuration that clears the dataset baseline by a real margin,
  # and does so while uniqueness falls from 0.997 to 0.854
  # shellcheck disable=SC2086
  run rtb-proj-osim-b20 --finetune_scope proj $RA \
    --reward_beta 20 --beta_start 5 --beta_anneal_epochs 4 \
    --bsz 64 --steps_per_epoch 100 --max_epochs 10 --lr 1e-5

  # LoRA on proj_logits: the capacity dial below full-rank proj
  for R in 4 16 64; do
    # shellcheck disable=SC2086
    run rtb-lora${R}-osim-b10 --finetune_scope proj \
      --lora_rank "$R" --lora_targets proj_logits --lora_alpha 16 $RA \
      --reward_beta 10 --beta_start 2 --beta_anneal_epochs 4 \
      --bsz 64 --steps_per_epoch 100 --max_epochs 6 --lr 1e-4
  done

  # rank-64 repeated from a different initialisation (Table 8, "alt. init")
  # shellcheck disable=SC2086
  run rtb-lora64-osim-b10-end --finetune_scope proj \
    --lora_rank 64 --lora_targets proj_logits --lora_alpha 16 $RA \
    --reward_beta 10 --beta_start 2 --beta_anneal_epochs 4 \
    --bsz 64 --steps_per_epoch 100 --max_epochs 6 --lr 1e-4

  # LoRA with the trunk unfrozen: same adapter, approximate ratio
  for R in 4 16 64; do
    # shellcheck disable=SC2086
    run rtb-atom-lora${R}-osim-b10 --finetune_scope atom \
      --lora_rank "$R" --lora_targets proj_logits --lora_alpha 16 $RA \
      --reward_beta 10 --beta_start 2 --beta_anneal_epochs 4 \
      --bsz 16 --steps_per_epoch 100 --max_epochs 6 --lr 1e-4
  done

  # head + trunk, full rank.
  #
  # DISCREPANCY: the recorded meta.json for the original rtb-atom-osim-b10 and
  # rtb-atom-peri-b10 runs says finetune_scope=full, not atom, although the run
  # names and Table 8 (43.1M trainable parameters) both say atom. The scope is
  # set to atom here, matching the name and the reported parameter count. If the
  # recorded value is the accurate one, these two rows belong with FULL and the
  # capacity axis needs re-running -- resolve before citing them as ATOM.
  # shellcheck disable=SC2086
  run rtb-atom-osim-b10 --finetune_scope atom $RA \
    --reward_beta 10 --beta_start 2 --beta_anneal_epochs 4 \
    --bsz 32 --steps_per_epoch 100 --max_epochs 6 \
    --lr 5e-6 --trunk_lr_mult 0.1 --logp_grad_frac 0.25

  # every weight. Batch size drops to fit the retained autograd graph, and
  # logp_grad_frac keeps a random quarter of the per-step graph -- unbiased
  # conditional on the trajectory, at the cost of variance. At this batch size
  # the run reaches 4,800 oracle calls rather than 38,400, which is why it is
  # reported separately from the capacity axis.
  # shellcheck disable=SC2086
  run rtb-full-osim-b10 --finetune_scope full $RA \
    --reward_beta 10 --beta_start 2 --beta_anneal_epochs 4 \
    --bsz 16 --steps_per_epoch 100 --max_epochs 12 \
    --lr 5e-6 --trunk_lr_mult 0.1 --logp_grad_frac 0.25
fi

# --------------------------- 2. Perindopril MPO ------------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "peri" ]]; then
  RA="$(reward_args peri)"

  # shellcheck disable=SC2086
  run rtb-proj-peri-b10 --finetune_scope proj $RA \
    --reward_beta 10 --beta_start 2 --beta_anneal_epochs 4 \
    --bsz 64 --steps_per_epoch 100 --max_epochs 6 --lr 1e-5

  for R in 4 16 64; do
    # shellcheck disable=SC2086
    run rtb-proj-lora${R}-peri-b10 --finetune_scope proj \
      --lora_rank "$R" --lora_targets proj_logits --lora_alpha 16 $RA \
      --reward_beta 10 --beta_start 2 --beta_anneal_epochs 4 \
      --bsz 64 --steps_per_epoch 100 --max_epochs 6 --lr 1e-4
  done

  for R in 4 16 64; do
    # shellcheck disable=SC2086
    run rtb-atom-lora${R}-peri-b10 --finetune_scope atom \
      --lora_rank "$R" --lora_targets proj_logits --lora_alpha 16 $RA \
      --reward_beta 10 --beta_start 2 --beta_anneal_epochs 4 \
      --bsz 16 --steps_per_epoch 100 --max_epochs 6 --lr 1e-4
  done

  # see the note on rtb-atom-osim-b10 above: the recorded scope was full
  # shellcheck disable=SC2086
  run rtb-atom-peri-b10 --finetune_scope atom $RA \
    --reward_beta 10 --beta_start 2 --beta_anneal_epochs 4 \
    --bsz 16 --steps_per_epoch 100 --max_epochs 6 \
    --lr 5e-6 --trunk_lr_mult 0.1 --logp_grad_frac 0.25

  # shellcheck disable=SC2086
  run rtb-full-peri-b10 --finetune_scope full $RA \
    --reward_beta 10 --beta_start 2 --beta_anneal_epochs 4 \
    --bsz 12 --steps_per_epoch 100 --max_epochs 10 \
    --lr 5e-6 --trunk_lr_mult 0.1 --logp_grad_frac 0.25
fi

# ------------------------------- summary -------------------------------------
hr
say "completed: ${#COMPLETED[@]}  ${COMPLETED[*]:-(none)}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  say "FAILED:    ${#FAILED[@]}  ${FAILED[*]}"
  say "re-run this script to retry only the failures"
  exit 1
fi
say "all runs complete. Score them with: bash scripts/08_analysis.sh harvest"
hr

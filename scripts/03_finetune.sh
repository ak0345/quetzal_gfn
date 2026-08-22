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
# SEEDS is opt-in and empty by default, which runs each configuration once with
# names exactly as before. Setting it runs each configuration once per seed,
# passing --seed and appending a -s<N> suffix to the run name and record
# directory. The paper's fine-tuning results are single-seed; this is what makes
# them repeatable.
#
# Usage:
#   bash scripts/03_finetune.sh                       # sanity + osim + peri
#   SEEDS="0 42 100" bash scripts/03_finetune.sh      # three seeds per config
#   bash scripts/03_finetune.sh osim
#   SKIP_SANITY=1 bash scripts/03_finetune.sh
#   ONLY=rtb-proj-osim-b10 bash scripts/03_finetune.sh
#   DRY=1 bash scripts/03_finetune.sh
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_prior

WHICH="${1:-all}"
[[ "$WHICH" != "all" ]] && REWARDS="$WHICH"

MOLROOT="${MOLROOT:-${RESULTS_ROOT}/oracle_gfn_mols}"
SEEDS="${SEEDS:-}"          # empty -> one unsuffixed run per configuration
REWARDS="${REWARDS:-osim peri zaleplon}"
LORA_RANKS="${LORA_RANKS:-4 16 64}"
BETA="${BETA:-10}"
BETA_START="${BETA_START:-2}"

# ~10,000 molecules per configuration, which is the harvest budget
BSZ="${BSZ:-64}";        STEPS_SMALL="${STEPS_SMALL:-40}";  EPOCHS_SMALL="${EPOCHS_SMALL:-4}"
BSZ_FULL="${BSZ_FULL:-12}"; STEPS_FULL="${STEPS_FULL:-140}"; EPOCHS_FULL="${EPOCHS_FULL:-6}"
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
  # SEED is set by the per-seed loop below; empty means the legacy single run.
  # The +"${a[@]}" guard is needed because bash 3.2 treats expanding an empty
  # array under `set -u` as an unbound variable.
  local seed_arg=()
  if [[ -n "${SEED:-}" ]]; then
    name="${name}-s${SEED}"
    seed_arg=(--seed "$SEED")
  fi
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
    --guard_stall_minutes "$GUARD_STALL_MINUTES"
    --guard_reward_timeout "$GUARD_REWARD_TIMEOUT"
    --max_train_hours "$MAX_TRAIN_HOURS"
    --record_dir "$MOLROOT/$name"
    ${seed_arg[@]+"${seed_arg[@]}"}
    "$@")

  if [[ "$DRY" == "1" ]]; then printf '  %q' "${cmd[@]}"; echo; return 0; fi

  say "START $name -> $MOLROOT/$name"
  local t0=$SECONDS
  # 17 (stall) and 18 (time limit) leave a sound checkpoint, so no .done marker
  # is written and the next attempt resumes rather than restarts.
  "${cmd[@]}" 2>&1 | tee -a "$LOGS/$name.log"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -eq 17 || $rc -eq 18 ]]; then
    say "PAUSE $name (exit $rc: $([[ $rc -eq 17 ]] && echo stall || echo "${MAX_TRAIN_HOURS}h limit")); a retry resumes it"
    return 0
  fi
  if [[ $rc -eq 0 ]]; then
    local n=0
    [[ -f "$MOLROOT/$name/molecules.jsonl" ]] && n=$(wc -l < "$MOLROOT/$name/molecules.jsonl")
    touch "$STATE/$name.done"
    say "DONE  $name ($(( (SECONDS-t0)/60 ))m, $n molecules recorded)"
    COMPLETED+=("$name")
  else
    say "FAIL  $name (exit $rc, see $LOGS/$name.log)"
    FAILED+=("$name"); return 1
  fi
}

# reward_args <osim|peri> -- the assembled benchmark, via the guacamol passthrough
reward_args () {
  case "$1" in
    osim)     echo "--reward guacamol --reward_smiles hard_osimertinib" ;;
    peri)     echo "--reward guacamol --reward_smiles perindopril_rings" ;;
    fexo) echo "--reward guacamol --reward_smiles hard_fexofenadine" ;;
    zaleplon) echo "--reward guacamol --reward_smiles zaleplon_with_other_formula" ;;
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

# "" is the no-seed case: one pass, names unchanged, --seed not passed.
for SEED in ${SEEDS:-""}; do
[[ -n "$SEED" ]] && say "===== training seed ${SEED} ====="

# ------------------------- 1. the benchmark runs -----------------------------
# One pass per reward. Every configuration is sized to collect ~10,000 molecules,
# which is the oracle budget stage 8 harvests at, so no run generates samples that
# the budget slice would throw away.
#
#   proj / atom : bsz 64, 4 epochs x 40 steps  = 160 steps -> 10,240 molecules
#   full        : bsz 12, 6 epochs x 140 steps = 840 steps -> 10,080 molecules
#
# beta anneals from BETA_START to BETA over the first half of the run, so each
# configuration spends its second half at the target tilt.
for REWARD in $REWARDS; do
  RA="$(reward_args "$REWARD")"
  [[ -z "$RA" ]] && { say "SKIP unknown reward $REWARD"; continue; }
  say "----- $REWARD -----"

  # ---- proj: head only, exact RTB ratio ----
  # shellcheck disable=SC2086
  run rtb-proj-${REWARD}-b${BETA} --finetune_scope proj $RA \
    --reward_beta "$BETA" --beta_start "$BETA_START" --beta_anneal_epochs 2 \
    --bsz "$BSZ" --steps_per_epoch "$STEPS_SMALL" --max_epochs "$EPOCHS_SMALL" \
    --lr 1e-5

  for R in $LORA_RANKS; do
    # shellcheck disable=SC2086
    run rtb-proj-lora${R}-${REWARD}-b${BETA} --finetune_scope proj \
      --lora_rank "$R" --lora_targets proj_logits --lora_alpha 16 $RA \
      --reward_beta "$BETA" --beta_start "$BETA_START" --beta_anneal_epochs 2 \
      --bsz "$BSZ" --steps_per_epoch "$STEPS_SMALL" --max_epochs "$EPOCHS_SMALL" \
      --lr 1e-4
  done

  # ---- atom: head plus the encode1 trunk, approximate ratio ----
  # shellcheck disable=SC2086
  run rtb-atom-${REWARD}-b${BETA} --finetune_scope atom $RA \
    --reward_beta "$BETA" --beta_start "$BETA_START" --beta_anneal_epochs 2 \
    --bsz "$BSZ" --steps_per_epoch "$STEPS_SMALL" --max_epochs "$EPOCHS_SMALL" \
    --lr 5e-6 --trunk_lr_mult 0.1

  for R in $LORA_RANKS; do
    # shellcheck disable=SC2086
    run rtb-atom-lora${R}-${REWARD}-b${BETA} --finetune_scope atom \
      --lora_rank "$R" --lora_targets proj_logits --lora_alpha 16 $RA \
      --reward_beta "$BETA" --beta_start "$BETA_START" --beta_anneal_epochs 2 \
      --bsz "$BSZ" --steps_per_epoch "$STEPS_SMALL" --max_epochs "$EPOCHS_SMALL" \
      --lr 1e-4 --trunk_lr_mult 0.1
  done

  # ---- full: every weight, no LoRA variants ----
  # The batch drops to fit the autograd graph retained over the rollout, and
  # logp_grad_frac keeps a random quarter of the per-step graph, which is
  # unbiased conditional on the trajectory at the cost of variance.
  # shellcheck disable=SC2086
  run rtb-full-${REWARD}-b${BETA} --finetune_scope full $RA \
    --reward_beta "$BETA" --beta_start "$BETA_START" --beta_anneal_epochs 3 \
    --bsz "$BSZ_FULL" --steps_per_epoch "$STEPS_FULL" --max_epochs "$EPOCHS_FULL" \
    --lr 5e-6 --trunk_lr_mult 0.1 --logp_grad_frac 0.25
done

done   # seed loop

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

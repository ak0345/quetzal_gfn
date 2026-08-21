#!/usr/bin/env bash
# =============================================================================
# run_study.sh -- the full study, supervised, in one invocation.
#
# THE GRID
#   guides (stage 1)   4 rewards {osim, peri, zaleplon, nitrogen}
#                      x 2 architectures {hidden, base}
#                      x 2 objectives {db, rtb}
#                      x 2 replay settings {on, off}
#                      x 3 betas {1, 10, 100}
#                      x 2 seeds  =  192 runs
#                      600 optimiser steps at batch 128 = 76,800 molecules each
#
#   fine-tunes (3)     3 rewards {osim, peri, zaleplon}
#                      x 9 configurations
#                        {proj, proj+LoRA 4/16/64, atom, atom+LoRA 4/16/64, full}
#                      x 2 seeds  =  54 runs, plus one nitrogen sanity run
#                      sized to ~10,000 molecules each, the harvest budget
#
# Neither stage runs eval or histograms during training. Scoring happens in
# stage 4 (dumps) and stage 8 (harvest), which is what keeps the training loop
# from stalling on CPU-side metric work.
#
# SUPERVISION
#   Every training process carries the in-process hang guard set to
#   GUARD_STALL_MINUTES (10 by default). A run whose batches stop progressing for
#   that long dumps its stacks, flushes its molecule log and exits 17.
#
#   Each stage is re-invoked up to MAX_RETRIES times. Every stage is resumable
#   (a run whose checkpoint or done-marker exists is skipped), so a retry picks
#   up exactly the runs that did not finish rather than repeating the stage.
#
# COST
#   The guide sweep alone is 192 x 76,800 = 14.7M molecules, each one rollout
#   plus one reward call. Measure your throughput before committing:
#
#     time MAX_EPOCHS=1 STEPS=100 SEEDS=0 REWARDS=osim GUIDES=hidden \
#       OBJECTIVES=db REPLAYS=off BETAS=10 bash scripts/01_train_guides.sh
#
#   That is 12,800 molecules. Divide the wall-clock by 12,800 and multiply by
#   15.2M for the whole study. At 75 ms per molecule it is on the order of two
#   weeks serial, or four to five days at MAX_PARALLEL=3.
#
# USAGE
#   DRY=1 bash scripts/run_study.sh              # print every command, run none
#   bash scripts/run_study.sh                    # the whole study
#   MAX_PARALLEL=3 bash scripts/run_study.sh     # 3 guide runs at a time
#   STAGES="4 8" bash scripts/run_study.sh       # re-score without retraining
#
#   nohup bash scripts/run_study.sh > study.log 2>&1 &
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------ the study grid -------------------------------
# Exported so the stage scripts and everything they call see the same values.
export SEEDS="${SEEDS:-0 42}"
export REWARDS="${REWARDS:-osim peri zaleplon nitrogen}"
export GUIDES="${GUIDES:-hidden base}"
export OBJECTIVES="${OBJECTIVES:-db rtb}"
export REPLAYS="${REPLAYS:-on off}"
export BETAS="${BETAS:-1 10 100}"
export MAX_EPOCHS="${MAX_EPOCHS:-6}"     # x STEPS = 600 optimiser steps
export STEPS="${STEPS:-100}"
export BSZ="${BSZ:-128}"

# the fine-tune half uses its own reward list, which has no nitrogen benchmark
export FT_REWARDS="${FT_REWARDS:-osim peri zaleplon}"

export GUARD_STALL_MINUTES="${GUARD_STALL_MINUTES:-10}"
export GUARD_REWARD_TIMEOUT="${GUARD_REWARD_TIMEOUT:-20}"
export MAX_PARALLEL="${MAX_PARALLEL:-1}"
export NUM_GPUS="${NUM_GPUS:-1}"
export DRY="${DRY:-0}"

MAX_RETRIES="${MAX_RETRIES:-3}"
STAGES="${STAGES:-1 3 4 6 8}"
BACKOFF="${BACKOFF:-30}"

source "$HERE/common.sh"

# --------------------------------- driver ------------------------------------
# Stages are resumable, so a retry is a no-op for everything that already
# finished. `attempt` therefore costs only what is left to do.
run_stage () {
  local label="$1"; shift
  local attempt=0
  while (( attempt < MAX_RETRIES )); do
    attempt=$((attempt+1))
    hr
    say "STAGE ${label}  (attempt ${attempt}/${MAX_RETRIES})"
    hr
    if "$@"; then
      say "STAGE ${label} finished"
      return 0
    fi
    say "STAGE ${label} exited non-zero on attempt ${attempt}"
    (( attempt < MAX_RETRIES )) && sleep "$BACKOFF"
  done
  say "STAGE ${label} still failing after ${MAX_RETRIES} attempts; continuing"
  return 1
}

START=$SECONDS
say "study grid: rewards='${REWARDS}' guides='${GUIDES}' objectives='${OBJECTIVES}'"
say "            betas='${BETAS}' replay='${REPLAYS}' seeds='${SEEDS}'"
say "            ${MAX_EPOCHS}x${STEPS} steps at batch ${BSZ}, guard ${GUARD_STALL_MINUTES} min,"
say "            up to ${MAX_RETRIES} attempts per stage, MAX_PARALLEL=${MAX_PARALLEL}"

for s in $STAGES; do
  case "$s" in
    1) run_stage "1 guides"      bash "$HERE/01_train_guides.sh" ;;
    # the fine-tune reward list differs from the guide one
    3) REWARDS="$FT_REWARDS" run_stage "3 fine-tune" bash "$HERE/03_finetune.sh" ;;
    4) run_stage "4 dumps"       bash "$HERE/04_dump_guides.sh" ;;
    6) run_stage "6 flips"       bash "$HERE/06_flip_diagnostics.sh" ;;
    8) for b in hard_osimertinib perindopril_rings zaleplon_with_other_formula; do
         BENCH="$b" run_stage "8 harvest ${b}" bash "$HERE/08_analysis.sh" harvest
       done
       run_stage "8 baselines"   bash "$HERE/08_analysis.sh" baseline ;;
    *) say "unknown stage '$s' (valid: 1 3 4 6 8)" ;;
  esac
done

hr
say "study finished in $(( (SECONDS-START)/3600 ))h $(( ((SECONDS-START)%3600)/60 ))m"
say "next: bash figures/make_all.sh"
hr

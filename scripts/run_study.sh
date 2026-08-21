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
# ORDER AND PARALLELISM
#   1 guides      GUIDE_PARALLEL concurrent runs (3 by default)
#   4 dumps       DUMP_PARALLEL concurrent runs (follows GUIDE_PARALLEL)
#   6 flips       serial, and cheap
#   7 ablations   serial, cheap, and needs stage 1's checkpoints
#   3 fine-tune   ONE AT A TIME, and last of the training stages
#   8 harvest     serial, reads the fine-tune streams so it must follow stage 3
#
# STAGES 2 AND 5 ARE NOT IN THE DEFAULT ORDER. Stage 2 trains the per-component
# guides and stage 5 composes them, and this study trains neither. Stage 7 runs
# only the sections that work from the sweep checkpoints: the margin binning
# (Figure 5), the residual-scale sweep (Figure 7) and the rollout diagnostics.
# Its `singles` section needs the component guides from stage 2 and its
# `tempgain` section needs temperature heads, which this study excludes, so both
# are skipped rather than run and left empty.
#
#   The guides parallelise because reward evaluation is CPU-serial while
#   generation is on the GPU, so concurrent processes overlap one run's scoring
#   with another's sampling. Fine-tuning does not: it holds two copies of the
#   model plus the autograd graph retained over the rollout, and at batch 128 in
#   the guide stage the GPU is already the constraint. Stage 3 therefore runs
#   each configuration synchronously, which is a property of the script rather
#   than a setting.
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
#   GUIDE_PARALLEL=2 bash scripts/run_study.sh   # ease off if VRAM is tight
#   STAGES="4 8" bash scripts/run_study.sh       # re-score without retraining
#   STAGES=3 bash scripts/run_study.sh           # only the fine-tunes
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
export NUM_GPUS="${NUM_GPUS:-1}"
export DRY="${DRY:-0}"

# Per-stage concurrency. Stage 3 is serial by construction and takes no setting.
GUIDE_PARALLEL="${GUIDE_PARALLEL:-3}"
DUMP_PARALLEL="${DUMP_PARALLEL:-$GUIDE_PARALLEL}"

# Stage 7 probes a small, representative set rather than all 192 runs: both
# architectures on one reward at one beta and one seed, which is what the margin
# and scale figures compare. Run names carry the -s<seed> suffix, so they have to
# be built from the grid rather than left to the script's un-seeded defaults.
ABL_REWARD="${ABL_REWARD:-osim}"
ABL_BETA="${ABL_BETA:-10}"
ABL_SEED="${ABL_SEED:-${SEEDS%% *}}"
ABL_SECTIONS="${ABL_SECTIONS:-ceiling guide rollout}"
if [[ -z "${ABL_RUNS:-}" ]]; then
  ABL_RUNS=""
  for g in $GUIDES; do
    ABL_RUNS="${ABL_RUNS} sweep-${ABL_REWARD}-${g}-db-replay_off-b${ABL_BETA}-s${ABL_SEED}"
  done
  ABL_RUNS="${ABL_RUNS# }"
fi

MAX_RETRIES="${MAX_RETRIES:-3}"
# Fine-tuning runs last of the training stages, and the harvest follows it
# because it reads the molecule streams stage 3 records.
STAGES="${STAGES:-1 4 6 7 3 8}"
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
say "            up to ${MAX_RETRIES} attempts per stage"
say "            stage order '${STAGES}' | guides x${GUIDE_PARALLEL}, dumps x${DUMP_PARALLEL}, fine-tune serial"
say "            ablations: sections '${ABL_SECTIONS}' over [${ABL_RUNS}]"

# `env` rather than a bare prefix so each stage gets exactly the concurrency it
# should have, with nothing leaking into the next one.
for s in $STAGES; do
  case "$s" in
    1) run_stage "1 guides" \
         env MAX_PARALLEL="$GUIDE_PARALLEL" bash "$HERE/01_train_guides.sh" ;;
    4) run_stage "4 dumps" \
         env MAX_PARALLEL="$DUMP_PARALLEL" bash "$HERE/04_dump_guides.sh" ;;
    6) run_stage "6 flips" \
         env MAX_PARALLEL=1 bash "$HERE/06_flip_diagnostics.sh" ;;
    7) for sec in $ABL_SECTIONS; do
         run_stage "7 ablations/${sec}" \
           env MAX_PARALLEL=1 RUNS="$ABL_RUNS" \
               SINGLE_CKPT="" bash "$HERE/07_ablations.sh" "$sec"
       done ;;
    # one configuration at a time, and the fine-tune reward list has no nitrogen
    3) run_stage "3 fine-tune" \
         env MAX_PARALLEL=1 REWARDS="$FT_REWARDS" bash "$HERE/03_finetune.sh" ;;
    8) for b in hard_osimertinib perindopril_rings zaleplon_with_other_formula; do
         run_stage "8 harvest ${b}" \
           env MAX_PARALLEL=1 BENCH="$b" bash "$HERE/08_analysis.sh" harvest
       done
       run_stage "8 baselines" \
         env MAX_PARALLEL=1 bash "$HERE/08_analysis.sh" baseline ;;
    *) say "unknown stage '$s' (valid: 1 3 4 6 7 8)" ;;
  esac
done

hr
say "study finished in $(( (SECONDS-START)/3600 ))h $(( ((SECONDS-START)%3600)/60 ))m"
say "next: bash figures/make_all.sh"
hr

#!/usr/bin/env bash
# =============================================================================
# run_study-all.sh -- the complete guide matrix, in one invocation.
#
# WHAT THIS IS FOR
#   run_study.sh, run_study2.sh and run_study-tempgain.sh each cover a slice of
#   the grid so the pieces can run on different machines. This runner is the
#   UNION of all three: every reward against every architecture. Point it at a
#   partly-finished tree and it trains only what is missing, which in practice
#   means the tempgain half plus whatever the other runners did not reach.
#
#   Nothing here is destructive. Stage 1 skips a run whose checkpoint exists and
#   stage 4 skips a (run, seed) whose dump_summary.json exists, so re-invoking
#   costs only what is left.
#
# THE GRID
#   4 rewards {osim, peri, fexo, nitrogen}
#     x 3 architectures {hidden, base, tempgain}
#     x 2 objectives {db, rtb}
#     x 2 replay settings {on, off}
#     x 3 betas {1, 10, 100}
#     x 2 seeds  =  288 runs
#   600 optimiser steps at batch 128 = 76,800 molecules each.
#
#   No fine-tuning and no harvest. Stage 3 needs a reward list that excludes
#   nitrogen and belongs with the runner that owns the fine-tunes; stage 8 reads
#   stage 3's molecule streams, so neither has anything to do here.
#
# STAGES  (default order: 1 2 4 5 6 7 9 3 8)
#   1 guides      the sweep, GUIDE_PARALLEL concurrent
#   2 components  per-component teachers; ONLY what stage 5 and stage 7's
#                 `singles` section need, and only for COMPONENT_BENCHES
#   4 dumps       the FULL dump: stage 4 globs every sweep-* and stability-*
#                 directory under CKPT_ROOT rather than filtering by this
#                 runner's reward list, so it picks up everything on disk
#   5 composed    composes the stage 2 teachers and scores the result
#   6 flips       coupled-flip probe, reads the architecture from each ckpt
#   7 ablations   mechanism probes, one invocation per section
#   9 occupancy   chemical-space occupancy against the prior, CPU only
#   3 fine-tune   ONE AT A TIME, and last of the training stages
#   8 harvest     reads the stage 3 molecule streams, so it must follow 3
#
# DEPENDENCIES THAT ACTUALLY BITE
#   1 -> 4 -> 9        occupancy compares dumps, so it needs stage 4
#   1 -> 6, 7
#   2 -> 5             composition needs the component teachers
#   2 -> 7 `singles`   likewise
#   3 -> 8             the harvest reads what fine-tuning recorded
#
# STAGE 7 RUNS EVERY SECTION HERE, unlike the sliced runners. `tempgain` works
# because this grid trains tempgain guides, and `singles` works because stage 2
# runs before it. Neither is true of run_study.sh or run_study2.sh.
#
# STAGES 2 AND 5 ARE OSIM-ONLY BY DEFAULT. The component decomposition is
# defined per benchmark and only osimertinib has been used that way in this
# repo; COMPONENT_BENCHES widens it if you have the others.
#
# CONCURRENCY
#   Guides parallelise because reward evaluation is CPU-serial while generation
#   is on the GPU. Dumps do NOT: two concurrent generations of 5,000 molecules
#   exhaust one A100, so DUMP_PARALLEL stays at 1 and the dump's own worker pool
#   (CONV_PROCS) uses the spare cores instead.
#
# USAGE
#   DRY=1 bash scripts/run_study-all.sh            # print every command, run none
#   bash scripts/run_study-all.sh                  # everything still missing
#   STAGES=1 bash scripts/run_study-all.sh         # train only
#   STAGES="4 9" bash scripts/run_study-all.sh     # dumps + occupancy
#   STAGES="3 8" bash scripts/run_study-all.sh     # fine-tune then harvest
#   STAGES=9 bash scripts/run_study-all.sh         # occupancy only, CPU
#   GUIDES=tempgain bash scripts/run_study-all.sh  # the tempgain half alone
#
#   nohup bash scripts/run_study-all.sh > study-all.log 2>&1 &
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------ the study grid -------------------------------
# Exported so the stage scripts and everything they call see the same values.
export SEEDS="${SEEDS:-0 42}"
export REWARDS="${REWARDS:-osim peri fexo}"
export GUIDES="${GUIDES:-hidden base tempgain}"
export OBJECTIVES="${OBJECTIVES:-db rtb}"
export REPLAYS="${REPLAYS:-on off}"
export BETAS="${BETAS:-1 10}"
export MAX_EPOCHS="${MAX_EPOCHS:-5}"     # x STEPS = 600 optimiser steps
export STEPS="${STEPS:-100}"
export BSZ="${BSZ:-128}"

export GUARD_STALL_MINUTES="${GUARD_STALL_MINUTES:-10}"
export GUARD_REWARD_TIMEOUT="${GUARD_REWARD_TIMEOUT:-20}"
export MAX_TRAIN_HOURS="${MAX_TRAIN_HOURS:-3}"
export NUM_GPUS="${NUM_GPUS:-1}"
export DRY="${DRY:-0}"

# Dump-side knobs. The 3D->SMILES conversion is the dump's bottleneck and runs
# in a worker pool; keep CONV_PROCS x DUMP_PARALLEL at or under the vCPU count.
export CONV_PROCS="${CONV_PROCS:-14}"
export CONV_TIMEOUT="${CONV_TIMEOUT:-90}"

GUIDE_PARALLEL="${GUIDE_PARALLEL:-3}"
DUMP_PARALLEL="${DUMP_PARALLEL:-1}"

# Stage 7 probes a small representative set rather than all 288 runs: every
# architecture on one reward at one beta and one seed, which is what the margin
# and scale figures compare. Run names carry the -s<seed> suffix, so they have
# to be built from the grid rather than left to the script's un-seeded defaults.
ABL_REWARD="${ABL_REWARD:-osim}"
ABL_BETA="${ABL_BETA:-10}"
ABL_SEED="${ABL_SEED:-${SEEDS%% *}}"
ABL_SECTIONS="${ABL_SECTIONS:-ceiling guide rollout tempgain singles}"

# Run names are DISCOVERED, not constructed. Two things make a constructed name
# wrong: runs trained before SEEDS existed carry no -s<N> suffix at all, and a
# given (reward, guide) may simply not have been trained at the preferred beta.
# A name that resolves to nothing leaves CKPTS empty, and then every section of
# stage 7 skips with "no guide checkpoints found".
#
# Preference order per architecture: the exact configuration with the seed
# suffix, the same without it, then any run for that (reward, guide) that has a
# checkpoint on disk.
pick_abl_run () {
  local g="$1" cand d
  for cand in \
      "sweep-${ABL_REWARD}-${g}-db-replay_off-b${ABL_BETA}-s${ABL_SEED}" \
      "sweep-${ABL_REWARD}-${g}-db-replay_off-b${ABL_BETA}"; do
    if compgen -G "${CKPT_ROOT}/${cand}/checkpoints/*.ckpt" > /dev/null; then
      echo "$cand"; return 0
    fi
  done
  while IFS= read -r d; do
    [[ -z "$d" ]] && continue
    if compgen -G "${d}/checkpoints/*.ckpt" > /dev/null; then
      basename "$d"; return 0
    fi
  done < <(ls -d "${CKPT_ROOT}/sweep-${ABL_REWARD}-${g}-"* 2>/dev/null | sort)
  return 1
}

# Resolved lazily, immediately before stage 7 runs: stage 1 may have just
# trained the very checkpoints this needs, so resolving at script start would
# miss them.
resolve_abl_runs () {
  if [[ -n "${ABL_RUNS:-}" ]]; then return 0; fi
  local g r missing=""
  ABL_RUNS=""
  for g in $GUIDES; do
    if r=$(pick_abl_run "$g"); then
      ABL_RUNS="${ABL_RUNS:+$ABL_RUNS }${r}"
    else
      missing="${missing:+$missing }${g}"
    fi
  done
  [[ -n "$missing" ]] && \
    say "[warn] stage 7: no ${ABL_REWARD} checkpoint for architecture(s): ${missing}"
  if [[ -z "$ABL_RUNS" ]]; then
    say "[warn] stage 7: nothing to probe under ${CKPT_ROOT}/sweep-${ABL_REWARD}-*"
    say "       set ABL_REWARD to a reward you have trained, or pass"
    say "       ABL_RUNS=\"name-a name-b\" explicitly"
    return 1
  fi
  return 0
}

# ---- reward -> GuacaMol names -----------------------------------------------
# Three different names for the same benchmark, and the stages disagree about
# which they want: stage 8 takes the standard_benchmarks function, stages 2 and
# 5 take the friendly component key. Nitrogen has neither, which is why it never
# appears in the fine-tune or harvest lists.
bench_fn () {            # for 08_analysis.sh
  case "$1" in
    osim)     echo "hard_osimertinib" ;;
    peri)     echo "perindopril_rings" ;;
    fexo)     echo "hard_fexofenadine" ;;
    zaleplon) echo "zaleplon_with_other_formula" ;;
    *)        echo "" ;;
  esac
}
bench_key () {           # for 02_train_components.sh / 05_dump_composed.sh
  case "$1" in
    osim)     echo "osimertinib" ;;
    peri)     echo "perindopril" ;;
    fexo)     echo "fexofenadine" ;;
    zaleplon) echo "zaleplon" ;;
    *)        echo "" ;;
  esac
}

# The fine-tune and harvest halves use their own reward list: nitrogen is a
# dense positive control for the guides, not a benchmark, and has no GuacaMol
# objective to fine-tune against or harvest.
export FT_REWARDS="${FT_REWARDS:-osim peri fexo}"
HARVEST_REWARDS="${HARVEST_REWARDS:-$FT_REWARDS}"

# Component teachers exist per benchmark; only osimertinib has been decomposed
# in this repo, so stages 2 and 5 default to it alone.
COMPONENT_BENCHES="${COMPONENT_BENCHES:-osim}"
COMPONENTS="${COMPONENTS:-0 2 3}"
COMPOSE_BETA="${COMPOSE_BETA:-10}"
OPERATORS="${OPERATORS:-linear,product,harmonic}"

# Stage 9 needs no configuration beyond the dumps; these are pass-throughs.
OCC_N="${OCC_N:-1000}"
OCC_CUTOFF="${OCC_CUTOFF:-0.25}"

MAX_RETRIES="${MAX_RETRIES:-3}"
STAGES="${STAGES:-1 4 6 7 9 8}"
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
say "            wall-clock ceiling ${MAX_TRAIN_HOURS}h, up to ${MAX_RETRIES} attempts per stage"
say "            stage order '${STAGES}' | guides x${GUIDE_PARALLEL}, dumps x${DUMP_PARALLEL}"
say "            dump conversion: ${CONV_PROCS} workers, ${CONV_TIMEOUT}s wedge timeout"
say "            ablations: sections '${ABL_SECTIONS}' (runs resolved at stage 7)"
say "            fine-tune/harvest rewards '${FT_REWARDS}' (no nitrogen: it has"
say "                       no GuacaMol objective to fine-tune against)"
say "            components/compose: [${COMPONENT_BENCHES}] components '${COMPONENTS}'"

# `env` rather than a bare prefix so each stage gets exactly the concurrency it
# should have, with nothing leaking into the next one.
for s in $STAGES; do
  case "$s" in
    1) run_stage "1 guides" \
         env MAX_PARALLEL="$GUIDE_PARALLEL" bash "$HERE/01_train_guides.sh" ;;
    # SUBSET=0 would widen the axes again; the grid above is already explicit,
    # so stage 4 is left to glob whatever stage 1 produced.
    2) for r in $COMPONENT_BENCHES; do
         bf="$(bench_fn "$r")"
         if [[ -z "$bf" ]]; then
           say "[warn] stage 2: no GuacaMol benchmark for '$r'; skipping"
           continue
         fi
         run_stage "2 components ${r}" \
           env MAX_PARALLEL="$GUIDE_PARALLEL" BENCH="$bf" BENCH_TAG="$r" \
               COMPONENTS="$COMPONENTS" BETA="$COMPOSE_BETA" \
               bash "$HERE/02_train_components.sh"
       done ;;
    4) run_stage "4 dumps" \
         env MAX_PARALLEL="$DUMP_PARALLEL" bash "$HERE/04_dump_guides.sh" ;;
    # composition needs the stage 2 teachers, so it is a no-op without them
    5) for r in $COMPONENT_BENCHES; do
         bk="$(bench_key "$r")"
         if [[ -z "$bk" ]]; then
           say "[warn] stage 5: no component key for '$r'; skipping"
           continue
         fi
         run_stage "5 composed ${r}" \
           env MAX_PARALLEL="$DUMP_PARALLEL" BENCH_KEY="$bk" BENCH_TAG="$r" \
               COMPONENTS="$COMPONENTS" BETA="$COMPOSE_BETA" \
               OPERATORS="$OPERATORS" bash "$HERE/05_dump_composed.sh"
       done ;;
    6) run_stage "6 flips" \
         env MAX_PARALLEL=1 bash "$HERE/06_flip_diagnostics.sh" ;;
    7) if resolve_abl_runs; then
         say "stage 7 will probe: ${ABL_RUNS}"
         for sec in $ABL_SECTIONS; do
           run_stage "7 ablations/${sec}" \
             env MAX_PARALLEL=1 RUNS="$ABL_RUNS" \
                 SINGLE_CKPT="" bash "$HERE/07_ablations.sh" "$sec"
         done
       else
         say "STAGE 7 skipped: no checkpoints to probe"
       fi ;;
    # CPU only, and it reads whatever stage 4 has produced so far, so it is
    # cheap to re-run as more dumps land
    9) run_stage "9 occupancy" \
         env MAX_PARALLEL=1 N="$OCC_N" CUTOFF="$OCC_CUTOFF" \
             bash "$HERE/09_cluster_occupancy.sh" ;;
    # serial by construction, and last of the training stages
    3) run_stage "3 fine-tune" \
         env MAX_PARALLEL=1 REWARDS="$FT_REWARDS" bash "$HERE/03_finetune.sh" ;;
    8) for r in $HARVEST_REWARDS; do
         bf="$(bench_fn "$r")"
         if [[ -z "$bf" ]]; then
           say "[warn] stage 8: no GuacaMol benchmark for '$r'; skipping"
           continue
         fi
         run_stage "8 harvest ${r}" \
           env MAX_PARALLEL=1 BENCH="$bf" bash "$HERE/08_analysis.sh" harvest
       done
       run_stage "8 baselines" \
         env MAX_PARALLEL=1 bash "$HERE/08_analysis.sh" baseline ;;
    *) say "unknown stage '$s' (this runner handles 1 2 3 4 5 6 7 8 9)" ;;
  esac
done

hr
say "study finished in $(( (SECONDS-START)/3600 ))h $(( ((SECONDS-START)%3600)/60 ))m"
say "next: bash figures/make_all.sh"
hr

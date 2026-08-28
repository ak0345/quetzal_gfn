#!/usr/bin/env bash
# =============================================================================
# run_study2.sh -- the fexo slice of the study, supervised, in one invocation.
#
# THE GRID
#   guides (stage 1)   1 reward {fexo}
#                      x 2 architectures {hidden, base}
#                      x 2 objectives {db, rtb}
#                      x 2 replay settings {on, off}
#                      x 3 betas {1, 10, 100}
#                      x 2 seeds  =  48 runs
#                      500 optimiser steps at batch 128 = 64,000 molecules each
#
#   fine-tunes (3)     3 rewards {fexo, zaleplon, peri}, off by default: stage 3
#                      is not in STAGES. Add it when the guides are done.
#
# Neither stage runs eval or histograms during training. Scoring happens in
# stage 4 (dumps) and stage 8 (harvest), which is what keeps the training loop
# from stalling on CPU-side metric work.
#
# ORDER AND PARALLELISM
#   1 guides      GUIDE_PARALLEL concurrent runs (3 by default)
#   4 dumps       DUMP_PARALLEL concurrent runs (1: two concurrent generations
#                 of 5,000 molecules exhaust one A100). The dump's own worker
#                 pool (CONV_PROCS) uses the spare cores instead.
#   6 flips       serial, and cheap
#   7 ablations   serial, cheap, and needs stage 1's checkpoints
#
# STAGES 2 AND 5 ARE NOT IN THE DEFAULT ORDER. Stage 2 trains the per-component
# guides and stage 5 composes them, and this study trains neither. Stage 7 runs
# only the sections that work from the sweep checkpoints: the margin binning
# (Figure 5), the residual-scale sweep (Figure 7) and the rollout diagnostics.
# Its `singles` section needs the component guides from stage 2, and `tempgain`
# needs temperature heads, which this slice does not train.
#
#   The guides parallelise because reward evaluation is CPU-serial while
#   generation is on the GPU, so concurrent processes overlap one run's scoring
#   with another's sampling.
#
# SUPERVISION
#   Every training process carries the in-process hang guard set to
#   GUARD_STALL_MINUTES. A run whose batches stop progressing for that long
#   dumps its stacks, flushes its molecule log and exits 17. MAX_TRAIN_HOURS
#   stops a run at a batch boundary and exits 18. Both leave the run resumable.
#
#   Each stage is re-invoked up to MAX_RETRIES times. Every stage is resumable
#   (a run whose checkpoint or done-marker exists is skipped), so a retry picks
#   up exactly the runs that did not finish rather than repeating the stage.
#
# USAGE
#   DRY=1 bash scripts/run_study2.sh              # print every command, run none
#   bash scripts/run_study2.sh                    # the whole slice
#   GUIDE_PARALLEL=2 bash scripts/run_study2.sh   # ease off if VRAM is tight
#   STAGES="4 6 7" bash scripts/run_study2.sh     # re-measure without training
#   STAGES=7 bash scripts/run_study2.sh           # ablations only
#
#   MAX_TRAIN_HOURS=3 N_REPORT_POS=256 N_TRAJ=1000 nohup bash scripts/run_study2.sh > study2.log 2>&1 &
#     
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------ the study grid -------------------------------
# Exported so the stage scripts and everything they call see the same values.
export SEEDS="${SEEDS:-0 42}"
export REWARDS="${REWARDS:-fexo}"
export GUIDES="${GUIDES:-hidden base}"
export OBJECTIVES="${OBJECTIVES:-db rtb}"
export REPLAYS="${REPLAYS:-on off}"
export BETAS="${BETAS:-1 10 100}"
export MAX_EPOCHS="${MAX_EPOCHS:-5}"     # x STEPS = 500 optimiser steps
export STEPS="${STEPS:-100}"
export BSZ="${BSZ:-128}"

# the fine-tune half uses its own reward list, which has no nitrogen benchmark
export FT_REWARDS="${FT_REWARDS:-fexo zaleplon peri}"

export GUARD_STALL_MINUTES="${GUARD_STALL_MINUTES:-60}"
export GUARD_REWARD_TIMEOUT="${GUARD_REWARD_TIMEOUT:-20}"
export NUM_GPUS="${NUM_GPUS:-1}"
export DRY="${DRY:-0}"

# Per-stage concurrency. Stage 3 is serial by construction and takes no setting.
GUIDE_PARALLEL="${GUIDE_PARALLEL:-3}"
DUMP_PARALLEL="${DUMP_PARALLEL:-1}"

# Dump-side knobs. The 3D->SMILES conversion is the dump's bottleneck and runs
# in a worker pool, so these MUST be exported: 04_dump_guides.sh reads them from
# the environment and otherwise falls back to its own smaller defaults.
export CONV_PROCS="${CONV_PROCS:-14}"
export CONV_TIMEOUT="${CONV_TIMEOUT:-90}"

# ------------------------------- stage 7 -------------------------------------
# Stage 7 probes a representative set rather than every run. It is invoked ONCE
# PER (reward, beta): 07_ablations.sh fills --train_betas with a single $BETA
# repeated per checkpoint and scores everything against a single benchmark, so a
# mixed-reward or mixed-beta invocation would mislabel its own inputs. Looping
# keeps each invocation internally consistent and gives it its own output dir.
ABL_REWARDS="${ABL_REWARDS:-${ABL_REWARD:-osim fexo peri zaleplon}}"
ABL_BETAS="${ABL_BETAS:-${ABL_BETA:-1 10 100}}"
ABL_SEED="${ABL_SEED:-${SEEDS%% *}}"
ABL_SECTIONS="${ABL_SECTIONS:-ceiling guide rollout}"

# NITROGEN IS NOT IN ABL_REWARDS BY DEFAULT. All three sections score their
# probes through --eval_rewards, and gflow_multi._build_eval_rewards understands
# qed / logp / tpsa / similarity / isomer / guacamol / gcomp -- there is no
# nitrogen spec. A nitrogen run passed here would be scored against whatever
# benchmark the default names, which is a wrong number rather than a missing
# one. Add it only alongside an ABL_EVAL entry that actually scores it.
abl_eval_spec () {
  case "$1" in
    osim)     echo "guacamol:hard_osimertinib=osim_MPO" ;;
    peri)     echo "guacamol:perindopril_rings=peri_MPO" ;;
    fexo)     echo "guacamol:hard_fexofenadine=fexo_MPO" ;;
    zaleplon) echo "guacamol:zaleplon_with_other_formula=zaleplon_MPO" ;;
    *)        echo "" ;;
  esac
}

# Run names are DISCOVERED, not constructed. Two things make a constructed name
# wrong: runs trained before SEEDS existed carry no -s<N> suffix at all, and a
# given (reward, guide, beta) may simply not have been trained. A name that
# resolves to nothing leaves CKPTS empty, and then every section of stage 7
# skips with "no guide checkpoints found".
#
# Preference per architecture: the exact configuration with the seed suffix, the
# same without it, then any run for that (reward, guide, beta) holding a .ckpt.
pick_abl_run () {          # $1 reward, $2 guide, $3 beta
  local r="$1" g="$2" b="$3" cand d
  for cand in \
      "sweep-${r}-${g}-db-replay_off-b${b}-s${ABL_SEED}" \
      "sweep-${r}-${g}-db-replay_off-b${b}"; do
    if compgen -G "${CKPT_ROOT}/${cand}/checkpoints/*.ckpt" > /dev/null; then
      echo "$cand"; return 0
    fi
  done
  # The beta must match EXACTLY. A glob of "*-b${b}*" lets -b1 match -b10 and
  # -b100, which would hand back a run at the wrong beta that 07_ablations.sh
  # then relabels with the requested one via --train_betas.
  while IFS= read -r d; do
    [[ -z "$d" ]] && continue
    cand="$(basename "$d")"
    case "$cand" in
      *-b"${b}"|*-b"${b}"-s*) ;;
      *) continue ;;
    esac
    if compgen -G "${d}/checkpoints/*.ckpt" > /dev/null; then
      echo "$cand"; return 0
    fi
  done < <(ls -d "${CKPT_ROOT}/sweep-${r}-${g}-"* 2>/dev/null | sort)
  return 1
}

MAX_RETRIES="${MAX_RETRIES:-3}"
STAGES="${STAGES:-1 4 6 7}"
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

# Resolved when stage 7 runs, not at script start: stage 1 may have just trained
# the very checkpoints this needs.
run_ablations () {
  local r b g runs missing eval_spec found_any=0
  for r in $ABL_REWARDS; do
    eval_spec="$(abl_eval_spec "$r")"
    if [[ -z "$eval_spec" ]]; then
      say "[warn] stage 7: no eval spec for reward '$r'; skipping it rather than"
      say "       scoring its guides against the wrong benchmark"
      continue
    fi
    for b in $ABL_BETAS; do
      runs=""; missing=""
      for g in $GUIDES; do
        if n=$(pick_abl_run "$r" "$g" "$b"); then
          runs="${runs:+$runs }${n}"
        else
          missing="${missing:+$missing }${g}"
        fi
      done
      [[ -n "$missing" ]] && \
        say "[warn] stage 7 ${r} b${b}: no checkpoint for architecture(s): ${missing}"
      if [[ -z "$runs" ]]; then
        say "[skip] stage 7 ${r} b${b}: nothing to probe"
        continue
      fi
      found_any=1
      say "stage 7 ${r} b${b} will probe: ${runs}"
      for sec in $ABL_SECTIONS; do
        run_stage "7 ablations/${sec} ${r} b${b}" \
          env MAX_PARALLEL=1 RUNS="$runs" SINGLE_CKPT="" \
              BETA="$b" EVAL_REWARDS="$eval_spec" \
              OUT_ROOT="${RESULTS_ROOT}/ablations/${r}-b${b}" \
              bash "$HERE/07_ablations.sh" "$sec"
      done
    done
  done
  (( found_any == 1 )) || say "STAGE 7 found no checkpoints to probe at all"
}

START=$SECONDS
say "study grid: rewards='${REWARDS}' guides='${GUIDES}' objectives='${OBJECTIVES}'"
say "            betas='${BETAS}' replay='${REPLAYS}' seeds='${SEEDS}'"
say "            ${MAX_EPOCHS}x${STEPS} steps at batch ${BSZ}, guard ${GUARD_STALL_MINUTES} min,"
say "            wall-clock ceiling ${MAX_TRAIN_HOURS}h, up to ${MAX_RETRIES} attempts per stage"
say "            stage order '${STAGES}' | guides x${GUIDE_PARALLEL}, dumps x${DUMP_PARALLEL}"
say "            dump conversion: ${CONV_PROCS} workers, ${CONV_TIMEOUT}s wedge timeout"
say "            ablations: sections '${ABL_SECTIONS}' over rewards '${ABL_REWARDS}'"
say "                       betas '${ABL_BETAS}' (runs resolved at stage 7)"

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
    7) run_ablations ;;
    # one configuration at a time, and the fine-tune reward list has no nitrogen
    3) run_stage "3 fine-tune" \
         env MAX_PARALLEL=1 REWARDS="$FT_REWARDS" bash "$HERE/03_finetune.sh" ;;
    8) for b in hard_osimertinib perindopril_rings hard_fexofenadine; do
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

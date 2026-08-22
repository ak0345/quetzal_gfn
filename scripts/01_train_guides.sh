#!/usr/bin/env bash
# =============================================================================
# Stage 1 -- the guide sweep.
#
# Trains a small guide on top of the frozen Quetzal prior across four axes:
#
#   seed      : training seed (opt-in, see below)
#   guide     : hidden | base                  (where the residual is injected)
#   objective : db | rtb                       (what it is trained to minimise)
#   replay    : on | off                       (db/rtb only; the KL branches
#                                               have no trajectory buffer)
#   beta      : reward exponent, 1 | 10 | 100
#   reward    : osim | peri | zaleplon         (GuacaMol MPO benchmarks)
#               nitrogen                       (dense positive control)
#
# Training only -- no eval, no dump. Scoring happens in stage 4, so a sweep can
# run to completion on one GPU and be measured afterwards. Each run is skipped
# if its checkpoint directory already holds a *.ckpt, so the sweep is resumable
# across restarts.
#
# SUBSET=1 (the default) runs the study grid: 4 rewards x 2 guides x 2 objectives
# x 2 replay settings x 3 betas = 96 configurations, or 192 runs at two seeds.
# SUBSET=0 restores the wider matrix including tempgain and the KL objectives.
#
# SEEDS is opt-in and empty by default, which trains one run per configuration
# under whatever the dataclass default is, with names exactly as before. Setting
# it trains one run per (configuration, seed), passing --seed and appending a
# -s<N> suffix to the run name so the runs neither collide nor resume into each
# other. The aggregators read that suffix back as a `train_seed` column.
#
# This is a different axis from the dump seeds in stage 4: those resample
# molecules from ONE trained checkpoint, and give sampling variance. Only SEEDS
# here gives variance across independently trained guides. Error bars over dump
# seeds alone understate run-to-run spread.
#
# Usage:
#   bash scripts/01_train_guides.sh
#   SEEDS="0 42 100" bash scripts/01_train_guides.sh  # three training seeds
#   SUBSET=0 bash scripts/01_train_guides.sh          # the entire matrix
#   REWARDS="nitrogen" bash scripts/01_train_guides.sh
#   MAX_PARALLEL=3 bash scripts/01_train_guides.sh    # 3 at a time on one GPU
#   DRY=1 bash scripts/01_train_guides.sh             # print commands only
# =============================================================================
set -uo pipefail    # no -e: one failed run must not kill the sweep
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_prior

SUBSET="${SUBSET:-1}"
SEEDS="${SEEDS:-}"          # empty -> one unsuffixed run per configuration
MAX_EPOCHS="${MAX_EPOCHS:-6}"     # 6 x 100 = 600 optimiser steps
STEPS="${STEPS:-100}"
BSZ="${BSZ:-128}"
LOGDIR="${LOGDIR:-${LOG_ROOT}/guides}"
mkdir -p "$LOGDIR"

# ------------------------------ axes ----------------------------------------
# Each axis is a space-separated string so it can be overridden from the
# environment, then split into an array.
if [[ "$SUBSET" == "1" ]]; then
  : "${GUIDES:=hidden base}"
  : "${OBJECTIVES:=db rtb}"
  : "${REPLAYS:=on off}"
  : "${BETAS:=1 10 100}"
  : "${REWARDS:=osim peri zaleplon nitrogen}"
else
  : "${GUIDES:=hidden tempgain base}"
  : "${OBJECTIVES:=db rtb revkl fwdkl}"
  : "${REPLAYS:=on off}"
  : "${BETAS:=1 2 4 10}"
  : "${REWARDS:=osim fexo peri nitrogen}"
fi
read -r -a GUIDES     <<< "$GUIDES"
read -r -a OBJECTIVES <<< "$OBJECTIVES"
read -r -a REPLAYS    <<< "$REPLAYS"
read -r -a BETAS      <<< "$BETAS"
read -r -a REWARDS    <<< "$REWARDS"

# ------------------------ reward -> flag translation ------------------------
# The assembled benchmark goes through the guacamol passthrough, which reads the
# standard_benchmarks function name from --reward_smiles. (--reward
# guacamol_component with --reward_component i would instead train against one
# leaf scorer of the MPO objective; that is stage 2.)
reward_flags () {
  case "$1" in
    osim)     echo "--reward guacamol --reward_smiles hard_osimertinib" ;;
    peri)     echo "--reward guacamol --reward_smiles perindopril_rings" ;;
    # GuacaMol's Zaleplon. PMO reimplements this benchmark differently, so PMO's
    # zaleplon_mpo numbers are not directly comparable to ours.
    zaleplon) echo "--reward guacamol --reward_smiles zaleplon_with_other_formula" ;;
    fexo)     echo "--reward guacamol --reward_smiles hard_fexofenadine" ;;
    nitrogen) echo "--reward nitrogen_count" ;;
    *) echo "UNKNOWN_REWARD_$1" ;;
  esac
}

# ------------------------ guide -> flag translation -------------------------
# argparse only generates the flag that flips a dataclass default, so a value
# already at its default is omitted rather than negated:
#   use_hidden_guide  defaults True  -> only --no_use_hidden_guide exists
#   use_prior_temp    defaults False -> only --use_prior_temp exists
#   use_residual_gain defaults False -> only --use_residual_gain exists
guide_flags () {
  case "$1" in
    hidden)   echo "" ;;
    tempgain) echo "--no_use_hidden_guide --use_prior_temp --use_residual_gain" ;;
    base)     echo "--no_use_hidden_guide" ;;
    *) echo "UNKNOWN_GUIDE_$1" ;;
  esac
}

replay_flags () {
  case "$1" in
    on)  echo "--use_replay --replay_fraction 0.25 --replay_strategy reward" ;;
    off) echo "" ;;
    *) echo "UNKNOWN_REPLAY_$1" ;;
  esac
}

# ------------------------------ driver --------------------------------------
COUNT=0; RAN=0; SKIPPED=0
START_TS=$(date +%s)

for reward in "${REWARDS[@]}"; do
  for guide in "${GUIDES[@]}"; do
    for obj in "${OBJECTIVES[@]}"; do
      if [[ " ${SKIP_OBJECTIVES:-} " == *" ${obj} "* ]]; then continue; fi
      for replay in "${REPLAYS[@]}"; do
        # replay is a trajectory buffer, meaningless for the KL objectives
        if [[ "$replay" == "on" && "$obj" != "db" && "$obj" != "rtb" ]]; then
          continue
        fi
        for beta in "${BETAS[@]}"; do
          # "" is the no-seed case: one run, name unchanged, --seed not passed
          for seed in ${SEEDS:-""}; do
            COUNT=$((COUNT+1))
            NAME="sweep-${reward}-${guide}-${obj}-replay_${replay}-b${beta}"
            SEED_ARG=""
            if [[ -n "$seed" ]]; then
              NAME="${NAME}-s${seed}"
              SEED_ARG="--seed ${seed}"
            fi

            # A run with any checkpoint is treated as done. That includes a run
            # stopped part-way by the hang guard or the wall-clock limit, which
            # will be skipped rather than resumed; delete its checkpoint
            # directory to force it to continue.
            if compgen -G "${CKPT_ROOT}/${NAME}/checkpoints/*.ckpt" > /dev/null; then
              echo "[skip $COUNT] $NAME (checkpoint exists)"
              SKIPPED=$((SKIPPED+1)); continue
            fi

            CMD="$PY gflow.py \
              --name ${NAME} \
              --quetzal_ckpt ${QUETZAL_CKPT} \
              --objective ${obj} \
              --reward_beta ${beta} \
              ${SEED_ARG} \
              --bsz ${BSZ} \
              ${GUARD_FLAGS} \
              $(reward_flags "$reward") \
              $(guide_flags "$guide") \
              $(replay_flags "$replay") \
              --max_epochs ${MAX_EPOCHS} \
              --steps_per_epoch ${STEPS} \
              --eval_n 0 \
              --final_n 0 \
              --hist_every_n_epochs 0 \
              --no_fcd_enabled \
              --no_eval_base"

            hr; echo "[run $COUNT] $NAME"; echo "$CMD"; hr
            [[ "$DRY" == "1" ]] && continue

            throttle
            (
              eval "$CMD" > "${LOGDIR}/${NAME}.log" 2>&1
              RC=$?
              case $RC in
                0)  ;;
                17) echo "[stall] $NAME hit the hang guard (see ${LOGDIR}/${NAME}.log)" ;;
                18) echo "[timelimit] $NAME stopped at ${MAX_TRAIN_HOURS}h (see ${LOGDIR}/${NAME}.log)" ;;
                *)  echo "[warn] $NAME exited $RC (see ${LOGDIR}/${NAME}.log)" ;;
              esac
            ) &
            echo "[launch $COUNT] $NAME (pid $!, active=$(( $(jobs -r -p | wc -l) )))"
            RAN=$((RAN+1))
            sleep 2   # stagger so parallel runs don't grab VRAM in lockstep
          done
        done
      done
    done
  done
done

wait
END_TS=$(date +%s)
hr
say "guide sweep done: enumerated=$COUNT ran=$RAN skipped=$SKIPPED"
say "elapsed $(( (END_TS-START_TS)/60 )) min | logs $LOGDIR | checkpoints ${CKPT_ROOT}/sweep-*"
say "next: scripts/04_dump_guides.sh"
hr

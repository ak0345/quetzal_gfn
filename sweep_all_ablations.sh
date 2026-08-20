#!/usr/bin/env bash
# ==========================================================================
# Full ablation sweep for the Quetzal GFlowNet-guidance project.
#
# Trains models ONLY (no eval, no final dump): each run does 8 epochs x 100
# steps, --eval_n 0, --final_n 0. Runs up to MAX_PARALLEL (default 3) at a time
# on a SINGLE shared GPU. Fresh --name per run (avoids resuming stale
# checkpoints, per the handoff gotcha). Skips a run if its checkpoint dir
# already has a *.ckpt (resumable).
#
# GPU NOTE: 3 concurrent trainings each hold their OWN copy of the frozen
# Quetzal + guide + optimizer state in VRAM. If you hit CUDA OOM, lower
# MAX_PARALLEL to 2 (or 1). There is no cross-process VRAM sharing here.
#
# AXES:
#   guide     : hidden | tempgain | base
#   objective : db | rtb | revkl | fwdkl
#   replay    : on | off        (only applied to db/rtb; kl branches ignore it)
#   beta      : 1 2 4 10
#   reward    : osim | fexo | peri  (guacamol_component)  +  nitrogen (easy, Ablation X)
#
# Full cross product (minus invalid replay+kl combos) = 288 runs. That is a LOT
# for sequential training. Use SUBSET=1 (default) for a decisive reduced grid;
# set SUBSET=0 to run the entire matrix.
#
# Requires the reward_fn.py nitrogen_count patch (reward_fn_nitrogen_patch.md)
# and the replay-buffer patch already applied to gflow.py.
# ==========================================================================

set -u  # error on unset vars; do NOT set -e (one failed run shouldn't kill the sweep)

# ----------------------------- knobs --------------------------------------
SUBSET="${SUBSET:-1}"                 # 1 = reduced decisive grid, 0 = full 288
QUETZAL_CKPT="${QUETZAL_CKPT:-geom.ckpt}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
STEPS="${STEPS:-100}"
LOGDIR="${LOGDIR:-sweep_logs}"
DRY_RUN="${DRY_RUN:-0}"               # 1 = print commands only, don't train
MAX_PARALLEL="${MAX_PARALLEL:-3}"    # concurrent runs on the single shared GPU
mkdir -p "$LOGDIR"

# ----------------------------- axes ---------------------------------------
if [[ "$SUBSET" == "1" ]]; then
  # Decisive reduced grid: the questions that actually matter.
  #  - hidden guide is the mechanistically strongest (fix B); base is the control.
  #  - db + rtb are the live objectives; kl is the loss-independence check.
  #  - beta 1 & 10: low (fittable target) vs the standard high value.
  #  - replay on/off on db only (the leading branch).
  GUIDES=(tempgain)
  OBJECTIVES=(db rtb)
  REPLAYS=(on off)
  BETAS=(1 10)
  REWARDS=(nitrogen osim peri)
else
  GUIDES=(hidden tempgain base)
  OBJECTIVES=(db rtb revkl fwdkl)
  REPLAYS=(on off)
  BETAS=(1 2 4 10)
  REWARDS=(osim fexo peri nitrogen)
fi

# --------------------- reward -> flag translation -------------------------
# Full assembled benchmark uses --reward guacamol --reward_smiles <sb_fn_name>
# (the guacamol passthrough reads cfg.reward_smiles). NOTE: --reward
# guacamol_component --reward_component 0 would train ONE sub-scorer of the MPO
# objective, not the benchmark itself -- so we use the passthrough here.
# nitrogen uses the easy --reward nitrogen_count (ignores smiles/benchmark).
reward_flags () {
  case "$1" in
    osim)     echo "--reward guacamol --reward_smiles hard_osimertinib" ;;
    fexo)     echo "--reward guacamol --reward_smiles hard_fexofenadine" ;;
    peri)     echo "--reward guacamol --reward_smiles perindopril_rings" ;;
    nitrogen) echo "--reward nitrogen_count" ;;
    *) echo "UNKNOWN_REWARD_$1" ;;
  esac
}

# --------------------- guide -> flag translation --------------------------
# NOTE argparse forms depend on the dataclass DEFAULTS in gflow.py:
#   use_hidden_guide  defaults TRUE  -> only --no_use_hidden_guide exists
#   use_prior_temp    defaults FALSE -> only --use_prior_temp exists
#   use_residual_gain defaults FALSE -> only --use_residual_gain exists
# So we OMIT (not negate) flags that are already at their default.
# hidden   : leave use_hidden_guide at its default True (pass nothing)
# tempgain : turn OFF hidden guide, turn ON prior-temp + residual-gain
# base     : turn OFF hidden guide, leave temp/gain OFF (their default)
guide_flags () {
  case "$1" in
    hidden)   echo "" ;;                                            # default is hidden
    tempgain) echo "--no_use_hidden_guide --use_prior_temp --use_residual_gain" ;;
    base)     echo "--no_use_hidden_guide" ;;                       # temp/gain already off
    *) echo "UNKNOWN_GUIDE_$1" ;;
  esac
}

# --------------------- replay -> flag translation -------------------------
# use_replay defaults FALSE -> only --use_replay exists; "off" passes nothing.
replay_flags () {
  case "$1" in
    on)  echo "--use_replay --replay_fraction 0.25 --replay_strategy reward" ;;
    off) echo "" ;;                                                 # default is off
    *) echo "UNKNOWN_REPLAY_$1" ;;
  esac
}

# ----------------------------- driver -------------------------------------
COUNT=0
RAN=0
SKIPPED=0
START_TS=$(date +%s)

# Block until the number of running background jobs drops below MAX_PARALLEL.
throttle () {
  # `jobs -r -p` lists PIDs of still-running background jobs of THIS shell.
  while [[ "$(jobs -r -p | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    # wait -n returns when ANY one job finishes (bash >= 4.3); fall back to a
    # short sleep-poll on older bash where wait -n is unavailable.
    wait -n 2>/dev/null || sleep 5
  done
}

for reward in "${REWARDS[@]}"; do
  for guide in "${GUIDES[@]}"; do
    for obj in "${OBJECTIVES[@]}"; do
      # optional: skip whole objectives, e.g. SKIP_OBJECTIVES="rtb" or "rtb fwdkl"
      if [[ " ${SKIP_OBJECTIVES:-} " == *" ${obj} "* ]]; then
        continue
      fi
      for replay in "${REPLAYS[@]}"; do
        # replay only meaningful for db/rtb; skip replay=on for kl objectives
        if [[ "$replay" == "on" && "$obj" != "db" && "$obj" != "rtb" ]]; then
          continue
        fi
        for beta in "${BETAS[@]}"; do
          COUNT=$((COUNT+1))

          NAME="sweep-${reward}-${guide}-${obj}-replay_${replay}-b${beta}"
          CKPT_DIR="logs/quetzal-gfn/${NAME}/checkpoints"

          # skip if already trained (resumable sweep)
          if compgen -G "${CKPT_DIR}/*.ckpt" > /dev/null; then
            echo "[skip $COUNT] $NAME (checkpoint exists)"
            SKIPPED=$((SKIPPED+1))
            continue
          fi

          RFLAGS=$(reward_flags "$reward")
          GFLAGS=$(guide_flags "$guide")
          PFLAGS=$(replay_flags "$replay")

          CMD="python gflow.py \
            --name ${NAME} \
            --quetzal_ckpt ${QUETZAL_CKPT} \
            --objective ${obj} \
            --reward_beta ${beta} \
            ${RFLAGS} \
            ${GFLAGS} \
            ${PFLAGS} \
            --max_epochs ${MAX_EPOCHS} \
            --steps_per_epoch ${STEPS} \
            --eval_n 0 \
            --final_n 0 \
            --hist_every_n_epochs 0 \
            --no_fcd_enabled \
            --no_eval_base"

          echo "==================================================================="
          echo "[run $COUNT] $NAME"
          echo "$CMD"
          echo "==================================================================="

          if [[ "$DRY_RUN" == "1" ]]; then
            continue
          fi

          # gate: wait until a slot frees up, then launch this run in the
          # background so up to MAX_PARALLEL train concurrently on the shared GPU.
          throttle
          (
            eval "$CMD" > "${LOGDIR}/${NAME}.log" 2>&1
            RC=$?
            if [[ $RC -ne 0 ]]; then
              echo "[warn] $NAME exited with code $RC (see ${LOGDIR}/${NAME}.log)"
            fi
          ) &
          echo "[launch $COUNT] $NAME (pid $!, active=$(( $(jobs -r -p | wc -l) )))"
          RAN=$((RAN+1))
          sleep 2   # small stagger so 3 processes don't grab VRAM in lockstep
        done
      done
    done
  done
done

# wait for the final in-flight batch to finish before summarizing
wait

END_TS=$(date +%s)
echo ""
echo "==================================================================="
echo "sweep done: enumerated=$COUNT  ran=$RAN  skipped=$SKIPPED"
echo "elapsed: $(( (END_TS-START_TS)/60 )) min"
echo "logs in: ${LOGDIR}/   checkpoints in: logs/quetzal-gfn/sweep-*/"
echo "==================================================================="
#!/usr/bin/env bash
# ==========================================================================
# Atom-stability guidance on GEOM (Quetzal's home benchmark).
#
# Trains hidden-guide DB models toward the EDM atom-stability reward, at
# beta 1 and 10, for BOTH the hidden guide and the base guide (control).
# Also runs a base-prior reference so you can see guided-vs-base stability.
#
# atom_stability is DENSE + prior-aligned + atom-type-driven -- the axis where
# guidance SHOULD work, unlike sparse osimertinib. The question is whether it
# raises stability above the frozen prior (positive result) or the prior is
# already near-ceiling (no headroom).
#
# Requires the atom_stability reward patch (reward_fn_atom_stability_patch.md).
# IMPORTANT: wire the repo metrics path in that patch first, or the numbers
# use the EDM fallback (fine for training signal, not for paper-matching).

# ==========================================================================

set -u

QUETZAL_CKPT="${QUETZAL_CKPT:-geom.ckpt}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
STEPS="${STEPS:-100}"
LOGDIR="${LOGDIR:-stability_logs}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
NUM_GPUS="${NUM_GPUS:-1}"
mkdir -p "$LOGDIR"

RAN=0
SKIPPED=0
START_TS=$(date +%s)

throttle () {
  while [[ "$(jobs -r -p | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    wait -n 2>/dev/null || sleep 5
  done
}

# guide flags: hidden = default (nothing); base = --no_use_hidden_guide
guide_flags () {
  case "$1" in
    hidden) echo "" ;;
    base)   echo "--no_use_hidden_guide" ;;
  esac
}

# NOTE eval_base ON here (unlike the training-only sweep): we WANT the base-vs
# guided stability delta, since headroom-vs-ceiling is the whole question.
for guide in hidden base; do
  for beta in 1 10; do
    NAME="stability-geom-${guide}-db-b${beta}"
    CKPT_DIR="logs/quetzal-gfn/${NAME}/checkpoints"
    if compgen -G "${CKPT_DIR}/*.ckpt" > /dev/null; then
      echo "[skip] $NAME (checkpoint exists)"; SKIPPED=$((SKIPPED+1)); continue
    fi
    GFLAGS=$(guide_flags "$guide")
    CMD="python gflow.py \
      --name ${NAME} \
      --quetzal_ckpt ${QUETZAL_CKPT} \
      --objective db \
      --reward atom_stability \
      --reward_beta ${beta} \
      ${GFLAGS} \
      --dataset geom \
      --max_epochs ${MAX_EPOCHS} \
      --steps_per_epoch ${STEPS} \
      --eval_n 500 \
      --final_n 0 \
      --hist_every_n_epochs 0 \
      --no_fcd_enabled"

    echo "==================================================================="
    echo "[run] $NAME"
    echo "$CMD"
    echo "==================================================================="
    if [[ "$DRY_RUN" == "1" ]]; then continue; fi

    throttle
    GPU=$(( RAN % NUM_GPUS ))
    (
      CUDA_VISIBLE_DEVICES="$GPU" eval "$CMD" > "${LOGDIR}/${NAME}.log" 2>&1
      RC=$?
      [[ $RC -ne 0 ]] && echo "[warn] $NAME exited $RC (see ${LOGDIR}/${NAME}.log)"
    ) &
    echo "[launch] $NAME (pid $!, gpu $GPU, active=$(( $(jobs -r -p | wc -l) )))"
    RAN=$((RAN+1)); sleep 2
  done
done

wait
END_TS=$(date +%s)
echo ""
echo "==================================================================="
echo "atom-stability runs done: ran=$RAN skipped=$SKIPPED  elapsed=$(( (END_TS-START_TS)/60 )) min"
echo "Read eval/log_reward_mean (guided) vs eval_base/log_reward_mean (base):"
echo "  guided > base  -> guidance raises atom stability (positive result)"
echo "  guided ~ base near 0 -> prior already near-ceiling (no headroom)"
echo "First: confirm base eval matches Quetzal's reported GEOM atom-stability."
echo "==================================================================="
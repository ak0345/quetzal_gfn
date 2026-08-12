#!/usr/bin/env bash
# ==========================================================================
# Per-COMPONENT guides for osimertinib, for the composition track.
#
# Trains ONE hidden-guide DB model per component of the assembled
# hard_osimertinib MPO objective, at beta=10. These are the per-component
# teachers you compose downstream (compose.py / Composer). Each uses
#   --reward guacamol_component --reward_benchmark hard_osimertinib
#   --reward_component <i>
# which reaches INTO the benchmark and trains on leaf scorer i only (NOT the
# assembled benchmark -- that's --reward guacamol, used elsewhere).
#
# osimertinib has 4 components (indices 0..3). Per the handoff, c0/c2/c3 carry
# real variance while c1 is a DEAD axis (ECFP6, std_logr=0): the guide gets no
# gradient there, so its curve will be flat by construction -- expected, not a
# bug. The composition step drops c1 and composes the other 3.
#
# Trains only (no eval / no final dump). Runs up to MAX_PARALLEL at a time,
# round-robin over NUM_GPUS. Fresh --name per run; skips if a checkpoint exists.
#
# Requires the replay-buffer patch already applied to gflow.py.
# ==========================================================================

set -u

# ----------------------------- knobs --------------------------------------
QUETZAL_CKPT="${QUETZAL_CKPT:-geom.ckpt}"
BETA="${BETA:-10}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
STEPS="${STEPS:-100}"
BENCH="${BENCH:-hard_osimertinib}"
COMPONENTS="${COMPONENTS:-0 1 2 3}"   # override to e.g. "0 2 3" to skip dead c1
LOGDIR="${LOGDIR:-compose_logs}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"     # concurrent runs across all gpus
NUM_GPUS="${NUM_GPUS:-1}"            # physical gpus for round-robin pinning
mkdir -p "$LOGDIR"

# ----------------------------- driver -------------------------------------
RAN=0
SKIPPED=0
START_TS=$(date +%s)

# Block until running background jobs drop below MAX_PARALLEL.
throttle () {
  while [[ "$(jobs -r -p | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    wait -n 2>/dev/null || sleep 5
  done
}

for comp in $COMPONENTS; do
  NAME="compose-osim-c${comp}-hidden-db-b${BETA}"
  CKPT_DIR="logs/quetzal-gfn/${NAME}/checkpoints"

  if compgen -G "${CKPT_DIR}/*.ckpt" > /dev/null; then
    echo "[skip] $NAME (checkpoint exists)"
    SKIPPED=$((SKIPPED+1))
    continue
  fi

  # hidden guide is the default (use_hidden_guide defaults True) -> no guide flag.
  CMD="python gflow.py \
    --name ${NAME} \
    --quetzal_ckpt ${QUETZAL_CKPT} \
    --objective db \
    --reward guacamol_component \
    --reward_benchmark ${BENCH} \
    --reward_component ${comp} \
    --reward_beta ${BETA} \
    --max_epochs ${MAX_EPOCHS} \
    --steps_per_epoch ${STEPS} \
    --eval_n 0 \
    --final_n 0 \
    --hist_every_n_epochs 0 \
    --no_fcd_enabled \
    --no_eval_base"

  echo "==================================================================="
  echo "[run c${comp}] $NAME"
  echo "$CMD"
  echo "==================================================================="

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  throttle
  GPU=$(( RAN % NUM_GPUS ))
  (
    CUDA_VISIBLE_DEVICES="$GPU" eval "$CMD" > "${LOGDIR}/${NAME}.log" 2>&1
    RC=$?
    if [[ $RC -ne 0 ]]; then
      echo "[warn] $NAME exited with code $RC (see ${LOGDIR}/${NAME}.log)"
    fi
  ) &
  echo "[launch c${comp}] $NAME (pid $!, gpu $GPU, active=$(( $(jobs -r -p | wc -l) )))"
  RAN=$((RAN+1))
  sleep 2
done

wait

END_TS=$(date +%s)
echo ""
echo "==================================================================="
echo "component training done: ran=$RAN  skipped=$SKIPPED"
echo "elapsed: $(( (END_TS-START_TS)/60 )) min"
echo "checkpoints in: logs/quetzal-gfn/compose-osim-c*/"
echo ""
echo "NEXT: compose the per-component guides (drop dead axis c1), e.g."
echo "  python compose.py \\"
echo "    --guide_ckpts <c0.ckpt> <c2.ckpt> <c3.ckpt> \\"
echo "    --weights 1 1 1 --train_betas ${BETA} ${BETA} ${BETA} \\"
echo "    --guide_labels c0 c2 c3 --eval_rewards <...> \\"
echo "    --reward_benchmark ${BENCH}"
echo "(EVERY per-guide list arg must be 3-wide once c1 is dropped.)"
echo "==================================================================="
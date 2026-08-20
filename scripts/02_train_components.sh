#!/usr/bin/env bash
# =============================================================================
# Stage 2 -- per-component guides, and the atom-stability control.
#
# Two families of guide that sit outside the main sweep of stage 1:
#
#   components  One guide per leaf scorer of an assembled MPO objective, via
#               --reward guacamol_component --reward_component <i>. These are
#               the teachers the composition track mixes in stage 5. Osimertinib
#               has four (0..3); component 1 (ECFP6 similarity) has zero variance
#               over reachable molecules, so its guide receives no gradient and
#               its curve is flat by construction. COMPONENTS defaults to the
#               three live axes.
#
#   stability   Guides trained against EDM atom stability rather than a GuacaMol
#               objective. Dense and atom-type-driven, so it is the axis where a
#               logit-level guide should work if anything does. Reported as an
#               excluded run: molecular stability sits below 0.05 under the prior
#               for 80-100 heavy-atom molecules, so the objective is close to
#               saturated and carries little gradient.
#
# Unlike stage 1 this runs with eval_base on for the stability family, since the
# guided-minus-prior delta is the whole question there.
#
# Usage:
#   bash scripts/02_train_components.sh              # both families
#   bash scripts/02_train_components.sh components
#   bash scripts/02_train_components.sh stability
#   COMPONENTS="0 1 2 3" bash scripts/02_train_components.sh components
#   DRY=1 bash scripts/02_train_components.sh
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_prior

WHICH="${1:-all}"

BENCH="${BENCH:-hard_osimertinib}"
BENCH_TAG="${BENCH_TAG:-osim}"
COMPONENTS="${COMPONENTS:-0 2 3}"
BETA="${BETA:-10}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
STEPS="${STEPS:-100}"
LOGDIR="${LOGDIR:-${LOG_ROOT}/components}"
mkdir -p "$LOGDIR"

RAN=0; SKIPPED=0
START_TS=$(date +%s)

launch () {
  local name="$1" cmd="$2"
  hr; echo "[run] $name"; echo "$cmd"; hr
  [[ "$DRY" == "1" ]] && return 0
  throttle
  local gpu=$(( RAN % NUM_GPUS ))
  (
    CUDA_VISIBLE_DEVICES="$gpu" eval "$cmd" > "${LOGDIR}/${name}.log" 2>&1
    RC=$?
    [[ $RC -ne 0 ]] && echo "[warn] $name exited $RC (see ${LOGDIR}/${name}.log)"
  ) &
  echo "[launch] $name (pid $!, gpu $gpu, active=$(( $(jobs -r -p | wc -l) )))"
  RAN=$((RAN+1)); sleep 2
}

# ------------------------- per-component guides ------------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "components" ]]; then
  say "component guides: bench=$BENCH components='$COMPONENTS' beta=$BETA"
  for comp in $COMPONENTS; do
    NAME="compose-${BENCH_TAG}-c${comp}-hidden-db-b${BETA}"
    if compgen -G "${CKPT_ROOT}/${NAME}/checkpoints/*.ckpt" > /dev/null; then
      echo "[skip] $NAME (checkpoint exists)"; SKIPPED=$((SKIPPED+1)); continue
    fi
    # the hidden guide is the dataclass default, so no guide flag is needed
    launch "$NAME" "$PY gflow.py \
      --name ${NAME} \
      --quetzal_ckpt ${QUETZAL_CKPT} \
      --objective db \
      --reward guacamol_component \
      --reward_benchmark ${BENCH} \
      --reward_component ${comp} \
      --reward_beta ${BETA} \
      --max_epochs ${MAX_EPOCHS} \
      --steps_per_epoch ${STEPS} \
      --eval_n 0 --final_n 0 \
      --hist_every_n_epochs 0 \
      --no_fcd_enabled --no_eval_base"
  done
fi

# --------------------------- atom-stability ----------------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "stability" ]]; then
  say "atom-stability guides (hidden vs base control), beta 1 and 10"
  for guide in hidden base; do
    for beta in 1 10; do
      NAME="stability-geom-${guide}-db-b${beta}"
      if compgen -G "${CKPT_ROOT}/${NAME}/checkpoints/*.ckpt" > /dev/null; then
        echo "[skip] $NAME (checkpoint exists)"; SKIPPED=$((SKIPPED+1)); continue
      fi
      GFLAGS=""
      [[ "$guide" == "base" ]] && GFLAGS="--no_use_hidden_guide"
      launch "$NAME" "$PY gflow.py \
        --name ${NAME} \
        --quetzal_ckpt ${QUETZAL_CKPT} \
        --objective db \
        --reward atom_stability \
        --reward_beta ${beta} \
        ${GFLAGS} \
        --dataset geom \
        --max_epochs ${MAX_EPOCHS} \
        --steps_per_epoch ${STEPS} \
        --eval_n 500 --final_n 0 \
        --hist_every_n_epochs 0 \
        --no_fcd_enabled"
    done
  done
fi

wait
END_TS=$(date +%s)
hr
say "stage 2 done: ran=$RAN skipped=$SKIPPED elapsed $(( (END_TS-START_TS)/60 )) min"
say "components feed scripts/05_dump_composed.sh; stability feeds scripts/04_dump_guides.sh"
hr

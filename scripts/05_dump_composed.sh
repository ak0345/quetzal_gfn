#!/usr/bin/env bash
# =============================================================================
# Stage 5 -- the composition track.
#
# Mixes the per-component guides from stage 2 into one sampler and scores it on
# the assembled MPO objective the components never saw, under three operators:
#
#   linear    p_M proportional to sum_i W_i          (exact at beta = 1)
#   product   product-of-experts, "all components high"   (approximate)
#   harmonic  the harmonic-mean product operator          (approximate)
#
# where W_i = omega_i * Z_i * u_i(s_t) weights each component by its learned
# partition function and its running probability of having reached the current
# state along the shared trajectory.
#
# One process handles all operators for a given seed (it loops over them
# internally, reusing the loaded checkpoints), so parallelism is over seeds.
#
# Each run emits both compose.py's own reporting -- reward histograms, KDE,
# ternary plot, hypervolume, per-component summary, under <out>/compose_native/
# -- and a dump_summary.json in the same schema as stage 4, so composed rows
# join the master table alongside the single-guide sweep.
#
# Note on routes: --route policy applies the composed residual to the logits.
# An earlier set of runs used a flow-based routing path in which the residual
# was computed but never applied; its rollout diagnostics show a residual norm
# of exactly 0.000 at every state and a flip rate identically zero at every
# position. Those runs are excluded from all reported results as a delivery
# failure rather than a bound.
#
# Usage:
#   bash scripts/05_dump_composed.sh
#   COMPONENTS="0 1 2 3" bash scripts/05_dump_composed.sh   # include the dead axis
#   OPERATORS=harmonic SEEDS=42 bash scripts/05_dump_composed.sh
#   DRY=1 bash scripts/05_dump_composed.sh
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
warn_missing_ref

OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/dumps_composed}"
SINGLE_ROOT="${SINGLE_ROOT:-${RESULTS_ROOT}/dumps}"
N="${N:-5000}"
SEEDS="${SEEDS:-42}"
DIFF_STEPS="${DIFF_STEPS:-18}"
REF_LIMIT="${REF_LIMIT:-$N}"
DATASET="${DATASET:-geom}"
BENCH_KEY="${BENCH_KEY:-osimertinib}"
BENCH_TAG="${BENCH_TAG:-osim}"
OPERATORS="${OPERATORS:-linear,product,harmonic}"
BETA="${BETA:-10}"
GUIDE_SOURCE="${GUIDE_SOURCE:-ema}"
MAIN_LOG="${MAIN_LOG:-${LOG_ROOT}/compose_dump_master.log}"
: > "$MAIN_LOG"

# Component 1 of the osimertinib objective (ECFP6 similarity) has zero variance
# over reachable molecules, so its guide has nothing to learn; the default
# composes the three live axes with equal weight.
COMPONENTS="${COMPONENTS:-0 2 3}"

CKPTS=""; LABELS=""; WEIGHTS=""; BETAS=""
NCOMP=$(echo "$COMPONENTS" | wc -w | tr -d ' ')
W=$(awk -v n="$NCOMP" 'BEGIN{printf "%.3f", 1.0/n}')
for c in $COMPONENTS; do
  name="compose-${BENCH_TAG}-c${c}-hidden-db-b${BETA}"
  ckpt=$(resolve_ckpt "$name") || {
    echo "[MISSING] component c${c}: no checkpoint for $name" | tee -a "$MAIN_LOG"; continue; }
  CKPTS="${CKPTS:+$CKPTS,}${ckpt}"
  LABELS="${LABELS:+$LABELS,}c${c}"
  WEIGHTS="${WEIGHTS:+$WEIGHTS,}${W}"
  BETAS="${BETAS:+$BETAS,}${BETA}"
done

if [[ -z "$CKPTS" ]]; then
  echo "[fatal] no component checkpoints found; run scripts/02_train_components.sh first" >&2
  exit 1
fi
say "composing ${NCOMP} components: ${LABELS}" | tee -a "$MAIN_LOG"

# Eval spec: the full MPO objective first (it is the primary metric and the one
# the aggregator reads), then each component present.
EVAL="guacamol:hard_${BENCH_KEY}=${BENCH_TAG}_MPO"
for c in $COMPONENTS; do
  EVAL="${EVAL},gcomp:${BENCH_KEY}:${c}=c${c}"
done

REF_ARG=""
[[ -f "$REF_SMILES" ]] && REF_ARG="--ref_smiles ${REF_SMILES} --ref_limit ${REF_LIMIT}"

RAN=0; START_TS=$(date +%s)
OUT="${OUT_ROOT}/${BENCH_KEY}"

for seed in $SEEDS; do
  done_all=1
  for op in ${OPERATORS//,/ }; do
    [[ -f "${OUT}/${op}/seed${seed}/dump_summary.json" ]] || done_all=0
  done
  if [[ "$done_all" == "1" ]]; then
    echo "[skip] all operators, seed$seed" | tee -a "$MAIN_LOG"; continue
  fi

  CMD="$PY final_dump_composed.py \
    --guide_ckpts ${CKPTS} \
    --guide_labels ${LABELS} \
    --weights ${WEIGHTS} \
    --train_betas ${BETAS} \
    --eval_rewards ${EVAL} \
    --bench_key ${BENCH_KEY} \
    --operators ${OPERATORS} \
    --n ${N} --seed ${seed} \
    --diff_steps ${DIFF_STEPS} --dataset ${DATASET} \
    --guide_source ${GUIDE_SOURCE} --progress \
    ${REF_ARG} \
    --out_dir ${OUT}"

  say "compose seed$seed (operators: ${OPERATORS})" | tee -a "$MAIN_LOG"
  [[ "$DRY" == "1" ]] && { echo "$CMD"; continue; }

  throttle
  GPU=$(( RAN % NUM_GPUS ))
  mkdir -p "$OUT"
  TAG="compose|s${seed}"
  (
    CUDA_VISIBLE_DEVICES="$GPU" stdbuf -oL -eL bash -c "$CMD" 2>&1 \
      | stdbuf -oL tee "${OUT}/compose_seed${seed}.log" \
      | stdbuf -oL tr '\r' '\n' \
      | stdbuf -oL sed "s#^#[${TAG}] #" >> "$MAIN_LOG"
  ) &
  echo "[launch] ${TAG} (pid $!)" | tee -a "$MAIN_LOG"
  RAN=$((RAN+1)); sleep 2
done

wait
END_TS=$(date +%s)
say "composed dumps done: ran=$RAN elapsed $(( (END_TS-START_TS)/60 )) min" | tee -a "$MAIN_LOG"

# Join composed rows into the single-guide master table.
if [[ "$DRY" != "1" ]]; then
  say "aggregating composed + single-guide dumps" | tee -a "$MAIN_LOG"
  $PY aggregate_dumps.py \
    --dumps_root "$SINGLE_ROOT" \
    --extra_roots "$OUT_ROOT" \
    --out_dir "${SINGLE_ROOT}/_aggregate" 2>&1 \
    | stdbuf -oL sed 's#^#[aggregate] #' | tee -a "$MAIN_LOG"
fi
hr

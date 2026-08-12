#!/usr/bin/env bash
# ==========================================================================
# Dump the COMPOSED sampler (compose.py) across operators {linear, product,
# harmonic} and 3 seeds, with compose.py's native plots AND an aggregator-
# compatible dump_summary.json per (operator, seed).
#
# Composes the 4 osimertinib component guides (trained by train_osim_components.sh)
# and scores on the FULL osimertinib MPO the components never saw. Runs the
# aggregator at the end with --extra_roots so composed rows join the master table.
#
# Requires: final_dump_composed.py, compose.py, final_dump2.py, gflow.py,
# edm_metrics.py on PYTHONPATH, the 4 component checkpoints, and a GEOM ref .smi.
# ==========================================================================

set -u

# ----------------------------- knobs --------------------------------------
CKPT_ROOT="${CKPT_ROOT:-logs/quetzal-gfn}"
OUT_ROOT="${OUT_ROOT:-dumps_composed}"
SINGLE_ROOT="${SINGLE_ROOT:-dumps}"        # single-guide dumps, for joint aggregate
N="${N:-5000}"
SEEDS="${SEEDS:-42}"
DIFF_STEPS="${DIFF_STEPS:-18}"
REF_SMILES="${REF_SMILES:-results/geom_drugs_smiles.txt}"
REF_LIMIT="${REF_LIMIT:-$N}"
DATASET="${DATASET:-geom}"
BENCH_KEY="${BENCH_KEY:-osimertinib}"
OPERATORS="${OPERATORS:-linear,product,harmonic}"
BETA="${BETA:-10}"                          # the beta the component guides were trained with
GUIDE_SOURCE="${GUIDE_SOURCE:-ema}"
CKPT_FILE="${CKPT_FILE:-last.ckpt}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
MAIN_LOG="${MAIN_LOG:-compose_dump_master.log}"
: > "$MAIN_LOG"

# ---- the 4 osimertinib component checkpoints (from train_osim_components.sh) ----
# component c1 (ECFP6) is the dead axis per the handoff; included here for the
# 4-way compose, but you can drop it and compose 3 (see COMPONENTS override).
COMPONENTS="${COMPONENTS:-0 2 3}"
CKPTS=""; LABELS=""; WEIGHTS=""; BETAS=""
for c in $COMPONENTS; do
  name="compose-osim-c${c}-hidden-db-b${BETA}"
  ckpt=$(ls -t "${CKPT_ROOT}/${name}/checkpoints/"*.ckpt 2>/dev/null | head -1)
  if [[ -z "$ckpt" ]]; then
    echo "[MISSING] component c${c}: no ckpt at ${CKPT_ROOT}/${name}/checkpoints/" | tee -a "$MAIN_LOG"
    continue
  fi
  CKPTS="${CKPTS:+$CKPTS,}${ckpt}"
  LABELS="${LABELS:+$LABELS,}c${c}"
  WEIGHTS="${WEIGHTS:+$WEIGHTS,}0.333"
  BETAS="${BETAS:+$BETAS,}${BETA}"
done
NCOMP=$(echo $COMPONENTS | wc -w)
echo "[compose] composing ${NCOMP} components: ${LABELS}" | tee -a "$MAIN_LOG"
echo "[compose] ckpts: ${CKPTS}" | tee -a "$MAIN_LOG"

if [[ -z "$CKPTS" ]]; then
  echo "[FATAL] no component checkpoints found; run train_osim_components.sh first" | tee -a "$MAIN_LOG"
  exit 1
fi

# eval spec: FULL MPO first (primary), then the components present.
EVAL="guacamol:hard_${BENCH_KEY}=osim_MPO"
for c in $COMPONENTS; do
  EVAL="${EVAL},gcomp:${BENCH_KEY}:${c}=c${c}"
done

[[ -f "$REF_SMILES" ]] || echo "[warn] REF_SMILES=$REF_SMILES not found; FCD/descriptor-vs-GEOM skipped" | tee -a "$MAIN_LOG"
REF_ARG=""
[[ -f "$REF_SMILES" ]] && REF_ARG="--ref_smiles ${REF_SMILES} --ref_limit ${REF_LIMIT}"

# ----------------------------- driver -------------------------------------
RAN=0; START_TS=$(date +%s)
throttle () { while [[ "$(jobs -r -p | wc -l)" -ge "$MAX_PARALLEL" ]]; do wait -n 2>/dev/null || sleep 5; done; }

# one final_dump_composed process handles ALL operators for a given seed
# (it loops operators internally), so we parallelize over SEEDS.
for seed in $SEEDS; do
  OUT="${OUT_ROOT}/${BENCH_KEY}"
  # skip only if ALL operator summaries for this seed already exist
  done_all=1
  for op in ${OPERATORS//,/ }; do
    [[ -f "${OUT}/${op}/seed${seed}/dump_summary.json" ]] || done_all=0
  done
  if [[ "$done_all" == "1" ]]; then
    echo "[skip] all operators seed$seed exist" | tee -a "$MAIN_LOG"; continue
  fi

  CMD="python final_dump_composed.py \
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

  echo "[run] compose seed$seed (operators: ${OPERATORS})" | tee -a "$MAIN_LOG"
  [[ "$DRY_RUN" == "1" ]] && { echo "$CMD"; continue; }

  throttle
  GPU=$(( RAN % NUM_GPUS ))
  mkdir -p "${OUT}"
  TAG="compose|s${seed}|gpu${GPU}"
  (
    CUDA_VISIBLE_DEVICES="$GPU" stdbuf -oL -eL bash -c "$CMD" 2>&1 \
      | stdbuf -oL tee "${OUT}/compose_seed${seed}.log" \
      | stdbuf -oL tr '\r' '\n' \
      | stdbuf -oL sed "s#^#[${TAG}] #" \
      >> "$MAIN_LOG"
  ) &
  echo "[launch] ${TAG} (pid $!, active=$(( $(jobs -r -p | wc -l) )))" | tee -a "$MAIN_LOG"
  RAN=$((RAN+1)); sleep 2
done

wait
END_TS=$(date +%s)
echo "[compose] done: ran=$RAN elapsed=$(( (END_TS-START_TS)/60 )) min" | tee -a "$MAIN_LOG"

# ----------------------------- aggregate ----------------------------------

#!/usr/bin/env bash
# =============================================================================
# Stage 4 -- score every trained guide.
#
# Dumps N molecules per (checkpoint, seed) with the full metric suite: reward
# histogram against the frozen prior, FCD and descriptor distances to
# GEOM-Drugs, EDM atom/mol stability, and a per-molecule CSV. Then aggregates
# every dump_summary.json into one master table with seed error bars.
#
# Two phases, because the frozen prior's samples do not depend on which guide
# is being evaluated:
#
#   PHASE 1  one base dump per (reward family, seed), written to
#            <out>/_base/<family>/seed<k>. All guides on the same reward reuse it.
#   PHASE 2  every checkpoint, guided-only, pointing --base_from at its family's
#            base dump. This is what makes the sweep affordable: FCD is a neural
#            embedding and EDM stability is quadratic in atom count, so
#            regenerating the base for each of ~60 checkpoints would dominate.
#
# Resumable: a (checkpoint, seed) whose dump_summary.json exists is skipped.
#
# diff_steps matches training at 18. Lowering it changes the geometry and
# therefore every bond-perception-derived number, so do not.
#
# The reference sample is capped at REF_LIMIT (default N) so the GEOM comparison
# set is the same size as the generated set -- FCD is biased by sample size, and
# loading all ~292k reference SMILES for every run is both slow and unfair.
#
# Usage:
#   bash scripts/04_dump_guides.sh
#   N=1000 SEEDS=0 bash scripts/04_dump_guides.sh     # quick pass
#   DRY=1 bash scripts/04_dump_guides.sh
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
warn_missing_ref

OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/dumps}"
N="${N:-5000}"
SEEDS_="${SEEDS_:-0}"
DIFF_STEPS="${DIFF_STEPS:-18}"
REF_LIMIT="${REF_LIMIT:-$N}"
DATASET="${DATASET:-geom}"
GUIDE_SOURCE="${GUIDE_SOURCE:-ema}"
# Worker processes for the 3D->SMILES conversion, which is the dump's
# bottleneck and is CPU-bound and single-threaded without this. Keep
# CONV_PROCS x MAX_PARALLEL at or under the vCPU count. CONV_TIMEOUT is the
# silence interval after which the remaining molecules are written off to a
# worker wedged inside rdDetermineBonds, where no Python timeout can reach.
CONV_PROCS="${CONV_PROCS:-8}"
CONV_TIMEOUT="${CONV_TIMEOUT:-90}"
CONV_ARGS="--conv_procs ${CONV_PROCS} --conv_timeout ${CONV_TIMEOUT}"
MAIN_LOG="${MAIN_LOG:-${LOG_ROOT}/dump_master.log}"
: > "$MAIN_LOG"

# Every run directory the guide stages produce. Override CHECKPOINTS to dump a
# subset.
CKPT_NAMES=()
if [[ -n "${CHECKPOINTS:-}" ]]; then
  read -r -a CKPT_NAMES <<< "$CHECKPOINTS"
else
  while IFS= read -r _n; do
    [[ -n "$_n" ]] && CKPT_NAMES+=("$_n")
  done < <(ls "$CKPT_ROOT" 2>/dev/null | grep -E '^(sweep|stability)-' | sort)
fi
if [[ ${#CKPT_NAMES[@]} -eq 0 ]]; then
  echo "[fatal] no sweep-* or stability-* runs under $CKPT_ROOT" >&2
  echo "        run scripts/01_train_guides.sh first" >&2
  exit 1
fi
say "${#CKPT_NAMES[@]} checkpoints x $(echo "$SEEDS_" | wc -w) seeds -> $OUT_ROOT" | tee -a "$MAIN_LOG"

RAN=0; SKIPPED=0; MISSING=0
START_TS=$(date +%s)

# The base sampler is the frozen prior, so it is shared within a reward family;
# only the reward config differs between families.
reward_family () {
  case "$1" in
    *-osim-*)     echo "osim" ;;
    *-zaleplon-*) echo "zaleplon" ;;
    *-fexo-*)     echo "fexo" ;;
    *-peri-*)     echo "peri" ;;
    *-nitrogen-*) echo "nitrogen" ;;
    stability-*)  echo "stability" ;;
    *)            echo "unknown" ;;
  esac
}

launch () {
  local tag="$1" out="$2" cmd="$3"
  throttle
  (
    stdbuf -oL -eL bash -c "$cmd" 2>&1 \
      | stdbuf -oL tee "${out}/dump.log" \
      | stdbuf -oL tr '\r' '\n' \
      | stdbuf -oL sed "s#^#[${tag}] #" >> "$MAIN_LOG"
    RC=${PIPESTATUS[0]}
    [[ $RC -ne 0 ]] && echo "[warn] ${tag} exited $RC (see ${out}/dump.log)" | tee -a "$MAIN_LOG"
  ) &
  echo "[launch] ${tag} (pid $!, active=$(( $(jobs -r -p | wc -l) )))" | tee -a "$MAIN_LOG"
  RAN=$((RAN+1)); sleep 2
}

REF_ARG=""
[[ -f "$REF_SMILES" ]] && REF_ARG="--ref_smiles ${REF_SMILES} --ref_limit ${REF_LIMIT}"

# ============================ PHASE 1: BASE ==================================
say "PHASE 1 -- frozen prior, one dump per (reward family, seed)" | tee -a "$MAIN_LOG"

# The base dump path is a pure function of (family, seed); SEEN just prevents
# scheduling the same one twice while walking the checkpoint list.
base_dir_for () { echo "${OUT_ROOT}/_base/${1}/seed${2}"; }
SEEN=" "

for name in "${CKPT_NAMES[@]}"; do
  fam=$(reward_family "$name")
  for seed in $SEEDS_; do
    key="${fam},${seed}"
    [[ "$SEEN" == *" ${key} "* ]] && continue
    SEEN="${SEEN}${key} "
    OUT="$(base_dir_for "$fam" "$seed")"
    if [[ -f "${OUT}/dump_summary.json" ]]; then
      echo "[skip-base] ${fam} seed${seed}"; SKIPPED=$((SKIPPED+1)); continue
    fi
    CKPT=$(resolve_ckpt "$name") || { echo "[MISSING] $name"; MISSING=$((MISSING+1)); continue; }
    mkdir -p "$OUT"
    GPU=$(( RAN % NUM_GPUS ))
    CMD="CUDA_VISIBLE_DEVICES=${GPU} $PY final_dump.py \
      --ckpt ${CKPT} --n ${N} --seed ${seed} --skip_guided \
      --diff_steps ${DIFF_STEPS} --dataset ${DATASET} --progress ${CONV_ARGS} \
      ${REF_ARG} --out_dir ${OUT}"
    [[ "$DRY" == "1" ]] && { echo "[base] ${fam} s${seed}: $CMD"; continue; }
    launch "BASE|${fam}|s${seed}" "$OUT" "$CMD"
  done
done
wait   # every base dump must exist before a guided run can --base_from it
say "PHASE 1 done" | tee -a "$MAIN_LOG"

# ============================ PHASE 2: GUIDED ================================
say "PHASE 2 -- every checkpoint, reusing the matching base dump" | tee -a "$MAIN_LOG"
for name in "${CKPT_NAMES[@]}"; do
  CKPT=$(resolve_ckpt "$name") || { echo "[MISSING] $name"; MISSING=$((MISSING+1)); continue; }
  fam=$(reward_family "$name")
  for seed in $SEEDS_; do
    OUT="${OUT_ROOT}/${name}/seed${seed}"
    if [[ -f "${OUT}/dump_summary.json" ]]; then
      echo "[skip] $name seed$seed"; SKIPPED=$((SKIPPED+1)); continue
    fi
    BASE="$(base_dir_for "$fam" "$seed")"
    BASE_ARG=""
    if [[ -f "${BASE}/dump_summary.json" ]]; then
      BASE_ARG="--base_from ${BASE}"
    else
      echo "[warn] no base dump for ${fam} seed${seed}; ${name} regenerates its own" \
        | tee -a "$MAIN_LOG"
    fi
    mkdir -p "$OUT"
    GPU=$(( RAN % NUM_GPUS ))
    CMD="CUDA_VISIBLE_DEVICES=${GPU} $PY final_dump.py \
      --ckpt ${CKPT} --n ${N} --seed ${seed} --guide_source ${GUIDE_SOURCE} \
      --diff_steps ${DIFF_STEPS} --dataset ${DATASET} --progress ${CONV_ARGS} \
      ${BASE_ARG} ${REF_ARG} --out_dir ${OUT}"
    [[ "$DRY" == "1" ]] && { echo "[run] $name s${seed}: $CMD"; continue; }
    launch "${name}|s${seed}" "$OUT" "$CMD"
  done
done
wait

END_TS=$(date +%s)
hr | tee -a "$MAIN_LOG"
say "dumps done: ran=$RAN skipped=$SKIPPED missing=$MISSING elapsed $(( (END_TS-START_TS)/60 )) min" \
  | tee -a "$MAIN_LOG"

# ------------------------------ aggregate ------------------------------------
if [[ "$DRY" != "1" ]]; then
  say "aggregating -> ${OUT_ROOT}/_aggregate" | tee -a "$MAIN_LOG"
  $PY aggregate_dumps.py \
    --dumps_root "$OUT_ROOT" \
    --out_dir "${OUT_ROOT}/_aggregate" 2>&1 \
    | stdbuf -oL sed 's#^#[aggregate] #' | tee -a "$MAIN_LOG"
  say "master table: ${OUT_ROOT}/_aggregate/master_table.csv"
fi
hr

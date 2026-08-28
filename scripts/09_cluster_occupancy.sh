#!/usr/bin/env bash
# =============================================================================
# Stage 9 -- chemical-space occupancy: new regions, or reweighted modes?
#
# Pools equal-sized samples from the frozen prior and from every guide in a
# reward family, measures where each one sits relative to the prior, and reports
# it against a null computed between two independent draws from the prior
# itself. See ablations/cluster_occupancy.py for what each statistic means and
# why the bin-based ones carry a health warning.
#
# READS   results/dumps/_base/<family>/seed<k>/base_smiles.txt   (the prior)
#         results/dumps/sweep-<family>-*/seed<k>/guided_smiles.txt
# WRITES  results/occupancy/occupancy_<family>_s<k>.{json,csv}
#         results/occupancy/occupancy_<family>_s<k>_assignments.csv
#
# NEEDS STAGE 4 FIRST, and specifically needs the PRIOR dump at two seeds. The
# null is what makes every other number readable, and it is the difference
# between "the guides reach 9% novel clusters" and "the guides reach 9% novel
# clusters where resampling the prior reaches 10%, so they reach none".
#
# CPU only, no GPU, no training. Cheap enough to re-run whenever new dumps land.
#
# Resumable: a (family, seed) whose .json exists is skipped. SKIP_EXISTING=0
# forces a redo.
#
# Usage:
#   bash scripts/09_cluster_occupancy.sh
#   FAMILIES="osim peri" bash scripts/09_cluster_occupancy.sh
#   N=3000 CUTOFF=0.3 bash scripts/09_cluster_occupancy.sh
#   SKIP_EXISTING=0 bash scripts/09_cluster_occupancy.sh
#   DRY=1 bash scripts/09_cluster_occupancy.sh
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SCRIPT="ablations/cluster_occupancy.py"
[[ -f "$SCRIPT" ]] || { echo "[fatal] $SCRIPT not found" >&2; exit 2; }

DUMPS_ROOT="${DUMPS_ROOT:-${RESULTS_ROOT}/dumps}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/occupancy}"
LOGDIR="${LOGDIR:-${LOG_ROOT}/occupancy}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Sample size per source. The nearest-neighbour step is O(n^2) per source, and
# 1500 already pins the mean to three decimals, so raising this buys precision
# nobody reads at a cost that grows quadratically.
N="${N:-1500}"
# Deliberately coarse. On this prior, a 0.35 cutoff leaves 76% of molecules
# alone in their own cluster and 0.55 leaves 97%; occupancy over a partition
# like that measures sample size rather than chemistry.
CUTOFF="${CUTOFF:-0.25}"
MAX_LEADERS="${MAX_LEADERS:-1500}"
REF_POOL="${REF_POOL:-4000}"
MAX_GUIDES="${MAX_GUIDES:-0}"       # 0 = every guide in the family
NULL_SEED="${NULL_SEED:-42}"
EXTRA="${EXTRA:-}"

mkdir -p "$OUT_DIR" "$LOGDIR"

# Families are discovered from the prior dumps rather than hardcoded: a family
# with no prior dump has no reference to compare against, so there is nothing to
# do for it. `stability` is excluded because it is a control on the frozen prior
# rather than a reward-steered family.
if [[ -z "${FAMILIES:-}" ]]; then
  FAMILIES=""
  while IFS= read -r d; do
    f="$(basename "$d")"
    [[ "$f" == "stability" ]] && continue
    FAMILIES="${FAMILIES:+$FAMILIES }${f}"
  done < <(ls -d "${DUMPS_ROOT}/_base/"*/ 2>/dev/null | sort)
fi
if [[ -z "$FAMILIES" ]]; then
  echo "[fatal] no prior dumps under ${DUMPS_ROOT}/_base/" >&2
  echo "        run scripts/04_dump_guides.sh first" >&2
  exit 1
fi

# Dump seeds come from whatever the prior was actually sampled at, which is not
# necessarily the training seed list.
if [[ -z "${DUMP_SEEDS:-}" ]]; then
  first="$(echo "$FAMILIES" | awk '{print $1}')"
  DUMP_SEEDS=""
  while IFS= read -r d; do
    s="$(basename "$d")"; s="${s#seed}"
    DUMP_SEEDS="${DUMP_SEEDS:+$DUMP_SEEDS }${s}"
  done < <(ls -d "${DUMPS_ROOT}/_base/${first}/seed"*/ 2>/dev/null | sort)
fi

say "families: [${FAMILIES}] | dump seeds: [${DUMP_SEEDS}] | n=${N} cutoff=${CUTOFF}"
OK=0; FAIL=0; SKIP=0; FAILED=()

for fam in $FAMILIES; do
  for seed in $DUMP_SEEDS; do
    out="${OUT_DIR}/occupancy_${fam}_s${seed}.json"
    if [[ "$SKIP_EXISTING" == "1" && -s "$out" ]]; then
      echo "[skip] ${fam} seed${seed}"; SKIP=$((SKIP+1)); continue
    fi
    if [[ ! -f "${DUMPS_ROOT}/_base/${fam}/seed${seed}/base_smiles.txt" ]]; then
      echo "[warn] ${fam} seed${seed}: no prior dump, skipping"; continue
    fi
    # the null needs a SECOND prior seed; without it every number loses its floor
    if [[ ! -f "${DUMPS_ROOT}/_base/${fam}/seed${NULL_SEED}/base_smiles.txt" ]]; then
      echo "[warn] ${fam}: no prior dump at the null seed ${NULL_SEED}; the"
      echo "       sampling floor cannot be computed and novel_mass will have"
      echo "       nothing to be compared against"
    fi

    CMD="$PY $SCRIPT --dumps_root ${DUMPS_ROOT} --family ${fam} \
      --dump_seed ${seed} --null_seed ${NULL_SEED} --n ${N} \
      --cutoff ${CUTOFF} --max_leaders ${MAX_LEADERS} --ref_pool ${REF_POOL} \
      --max_guides ${MAX_GUIDES} --out_dir ${OUT_DIR} ${EXTRA}"

    hr; echo "[run] ${fam} seed${seed}"; echo "$CMD"
    [[ "$DRY" == "1" ]] && continue

    if eval "$CMD" > "${LOGDIR}/${fam}_s${seed}.log" 2>&1; then
      echo "[ ok ] ${fam} seed${seed} -> ${out}"
      grep -E "^\[null\]|^\[warn\]" "${LOGDIR}/${fam}_s${seed}.log" | sed 's/^/       /'
      OK=$((OK+1))
    else
      echo "[FAIL] ${fam} seed${seed} (see ${LOGDIR}/${fam}_s${seed}.log)" >&2
      tail -n 12 "${LOGDIR}/${fam}_s${seed}.log" | sed 's/^/       | /' >&2
      FAIL=$((FAIL+1)); FAILED+=("${fam}_s${seed}")
    fi
  done
done

hr
say "occupancy: ok=$OK failed=$FAIL skipped=$SKIP -> $OUT_DIR"
[[ $FAIL -gt 0 ]] && echo "[warn] failed: ${FAILED[*]}" >&2
say "read the [null] lines first: no occupancy number means anything above it"
hr

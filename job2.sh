#!/usr/bin/env bash
#
# run_component_hists.sh -- Per-COMPONENT log-reward histograms.
#
# For each GuacaMol MPO benchmark, this decomposes the objective into its
# individual sub-scorers (via reward_fn's guacamol_component path) and plots
# each component's reward distribution separately on the same SMILES file.
# This is the companion to run_reward_hists.sh, which does the FULL (aggregate)
# objectives -- here we look inside each one.
#
# Component layout (enumerated from the installed guacamol; geometric mean each):
#   osimertinib  [0] sim-FCFP4     [1] sim-ECFP6   [2] TPSA   [3] logP
#   ranolazine   [0] sim-AP        [1] logP        [2] fluorine-count  [3] TPSA
#   sitagliptin  [0] sim-ECFP4     [1] logP        [2] TPSA   [3] isomer
#   fexofenadine [0] sim-AP        [1] TPSA        [2] logP
#   perindopril  [0] sim-ECFP4     [1] aromatic-rings
#
# If your installed guacamol version differs, re-check with:
#   python -c "import reward_fn as R; print(R.guacamol_components('osimertinib'))"
# and adjust the label arrays below (indices are what actually drive selection).
#
# Usage:
#   bash run_component_hists.sh
#
# Requires: rdkit, guacamol, matplotlib, numpy; reward_fn.py + reward_hist.py here.

set -euo pipefail

# ----------------------------------------------------------------- settings
SMILES="data/geom_drugs_smiles.txt"
N=10000
SEED=0
BINS=60
SCRIPT="smiles_hist.py"
OUTDIR="reward_plots_components"
PY="${PYTHON:-python}"

mkdir -p "$OUTDIR"

if [[ ! -f "$SMILES" ]]; then
  echo "ERROR: SMILES file not found: $SMILES" >&2; exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: $SCRIPT not found in $(pwd)" >&2; exit 1
fi

# Per-benchmark component labels, index-aligned. The label is only used to build
# a readable output filename; the integer index is what selects the component.
declare -A LABELS
LABELS[osimertinib]="sim-FCFP4 sim-ECFP6 tpsa logp"
LABELS[ranolazine]="sim-AP logp fluorine-count tpsa"
LABELS[sitagliptin]="sim-ECFP4 logp tpsa isomer"
LABELS[fexofenadine]="sim-AP tpsa logp"
LABELS[perindopril]="sim-ECFP4 aromatic-rings"

# Order in which benchmarks are processed.
BENCHES="osimertinib ranolazine sitagliptin fexofenadine perindopril"

echo "=================================================================="
echo " per-component reward histograms | file=$SMILES | n=$N | seed=$SEED"
echo "=================================================================="

run_job () {
  local label="$1"; shift
  echo
  echo "------------------------------------------------------------------"
  echo ">>> $label"
  echo "------------------------------------------------------------------"
  if "$PY" "$SCRIPT" "$@"; then
    echo "    [done] $label"
  else
    echo "    [FAILED] $label (continuing)" >&2
  fi
}

job_num=0
for bench in $BENCHES; do
  # shellcheck disable=SC2206
  labels=(${LABELS[$bench]})
  ncomp=${#labels[@]}
  for (( idx=0; idx<ncomp; idx++ )); do
    job_num=$((job_num + 1))
    lbl="${labels[$idx]}"
    tag=$(printf "%02d_%s_c%d_%s" "$job_num" "$bench" "$idx" "$lbl")
    run_job "$bench component [$idx] $lbl" \
      --smiles "$SMILES" --n "$N" --seed "$SEED" --bins "$BINS" \
      --reward guacamol_component --benchmark "$bench" --component "$idx" \
      --out "$OUTDIR/${tag}.png"
  done
done

echo
echo "=================================================================="
echo " all $job_num component jobs finished. plots in: $OUTDIR/"
echo "=================================================================="
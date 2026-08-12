#!/usr/bin/env bash
#
# run_reward_hists.sh -- Generate log-reward histograms for one SMILES file
# across 8 reward objectives, one after another.
#
# Each job samples the SAME 10,000 molecules (fixed --seed 0) so the reward
# distributions are directly comparable across objectives. Change SEED per
# block if you want an independent draw per objective instead.
#
# Usage:
#   bash run_reward_hists.sh
#   # or make it executable once:  chmod +x run_reward_hists.sh && ./run_reward_hists.sh
#
# Requires: python env with rdkit, guacamol, matplotlib, numpy; reward_fn.py
# and reward_hist.py in the same directory (or adjust SCRIPT below).

set -euo pipefail          # stop on first error, treat unset vars as errors

# ----------------------------------------------------------------- settings
SMILES="data/geom_drugs_smiles.txt"
SMILES_DB_osim="logs/quetzal-gfn/db-osim-compose/compose/osim_compose_db_composed_smiles.txt"
SMILES_DB_peri="logs/quetzal-gfn/db-peri-compose/compose/peri_compose_db_composed_smiles.txt"
SMILES_RTB_osim="logs/quetzal-gfn/osim-compose/compose-harmonic/osim_compose_composed_smiles.txt"
SMILES_BASE="logs/quetzal-gfn/db-peri-compose/compose/peri_compose_db_base_smiles.txt"
N=10000
CMP_N=2000
SEED=0
BINS=60
SCRIPT="smiles_hist.py"
OUTDIR="reward_plots"
PY="${PYTHON:-python}"     # override interpreter with:  PYTHON=python3 bash run_reward_hists.sh

mkdir -p "$OUTDIR"

# Fail early with a clear message if inputs are missing.
if [[ ! -f "$SMILES" ]]; then
  echo "ERROR: SMILES file not found: $SMILES" >&2
  exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: $SCRIPT not found in $(pwd)" >&2
  exit 1
fi

echo "=================================================================="
echo " reward histograms | file=$SMILES | n=$N | seed=$SEED"
echo "=================================================================="

# Small helper: print a header, time each job, keep going even if one fails
# (remove '|| ...' if you want the whole batch to abort on any single failure).
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

# ----------------------------------------------------------------- 1-5: full MPO benchmarks
# The full-benchmark passthrough (--reward guacamol) takes the raw guacamol
# standard_benchmarks function name via --reward-smiles.

#run_job "1) Osimertinib MPO (full)" \
#  --smiles "$SMILES" --n "$N" --seed "$SEED" --bins "$BINS" \
#  --reward guacamol --reward-smiles hard_osimertinib \
#  --out "$OUTDIR/01_osimertinib_mpo.png" --invalid-logr -15

#run_job "2) Ranolazine MPO (full)" \
#  --smiles "$SMILES" --n "$N" --seed "$SEED" --bins "$BINS" \
#  --reward guacamol --reward-smiles ranolazine_mpo \
#  --out "$OUTDIR/02_ranolazine_mpo.png"

#run_job "3) Sitagliptin MPO (full)" \
#  --smiles "$SMILES" --n "$N" --seed "$SEED" --bins "$BINS" \
#  --reward guacamol --reward-smiles sitagliptin_replacement \
#  --out "$OUTDIR/03_sitagliptin_mpo.png"

#run_job "4) Fexofenadine MPO (full)" \
#  --smiles "$SMILES" --n "$N" --seed "$SEED" --bins "$BINS" \
#  --reward guacamol --reward-smiles hard_fexofenadine \
#  --out "$OUTDIR/04_fexofenadine_mpo.png"

#run_job "5) Perindopril MPO (full)" \
#  --smiles "$SMILES" --n "$N" --seed "$SEED" --bins "$BINS" \
#  --reward guacamol --reward-smiles perindopril_rings \
#  --out "$OUTDIR/05_perindopril_mpo.png" --invalid-logr -15

# ----------------------------------------------------------------- 6-8: single-property rewards

#run_job "6) QED" \
#  --smiles "$SMILES" --n "$N" --seed "$SEED" --bins "$BINS" \
#  --reward qed \
#  --out "$OUTDIR/06_qed.png"

#run_job "7) logP low (target 0, sigma 1)" \
#  --smiles "$SMILES" --n "$N" --seed "$SEED" --bins "$BINS" \
#  --reward logp --target 0 --sigma 1 \
#  --out "$OUTDIR/07_logp_low.png"

#run_job "8) TPSA (target 80, sigma 20)" \
#  --smiles "$SMILES" --n "$N" --seed "$SEED" --bins "$BINS" \
#  --reward tpsa --target 80 --sigma 20 \
#  --out "$OUTDIR/08_tpsa.png"

#run_job "9) Compare Osimertinib with Base Quetzal" \
#  --smiles "$SMILES" --smiles2 "$SMILES_BASE" --n "$CMP_N" --seed "$SEED" --bins "$BINS" \
#  --reward guacamol --reward-smiles hard_osimertinib \
#  --out "$OUTDIR/01_osimertinib_mpo_cmp_base.png" --invalid-logr -15

#run_job "10) Compare Osimertinib with composed DB Quetzal" \
#  --smiles "$SMILES" --smiles2 "$SMILES_DB_osim" --n "$CMP_N" --seed "$SEED" --bins "$BINS" \
#  --reward guacamol --reward-smiles hard_osimertinib \
#  --out "$OUTDIR/01_osimertinib_mpo_cmp_DB.png" --invalid-logr -15

#run_job "11) Compare Osimertinib with composed RTB Quetzal" \
#  --smiles "$SMILES" --smiles2 "$SMILES_RTB_osim" --n "$CMP_N" --seed "$SEED" --bins "$BINS" \
#  --reward guacamol --reward-smiles hard_osimertinib \
#  --out "$OUTDIR/01_osimertinib_mpo_cmp_RTB.png" --invalid-logr -15

#run_job "12) Ranolazine MPO (full)" \
#  --smiles "$SMILES" --smiles2 "$SMILES_BASE" --n "$CMP_N" --seed "$SEED" --bins "$BINS" \
#  --reward guacamol --reward-smiles ranolazine_mpo \
#  --out "$OUTDIR/12_ranolazine_mpo_cmp_base.png" --invalid-logr -12

#run_job "13) Sitagliptin MPO (full)" \
#  --smiles "$SMILES" --smiles2 "$SMILES_BASE" --n "$CMP_N" --seed "$SEED" --bins "$BINS" \
#  --reward guacamol --reward-smiles sitagliptin_replacement \
#  --out "$OUTDIR/13_sitagliptin_mpo_cmp_base.png" --invalid-logr -80

#run_job "14) Fexofenadine MPO (full)" \
#  --smiles "$SMILES" --smiles2 "$SMILES_BASE" --n "$CMP_N" --seed "$SEED" --bins "$BINS" \
#  --reward guacamol --reward-smiles hard_fexofenadine \
#  --out "$OUTDIR/14_fexofenadine_mpo_cmp_base.png" --invalid-logr -10

#run_job "15) Perindopril MPO (full)" \
#  --smiles "$SMILES" --smiles2 "$SMILES_BASE" --n "$CMP_N" --seed "$SEED" --bins "$BINS" \
#  --reward guacamol --reward-smiles perindopril_rings \
#  --out "$OUTDIR/15_perindopril_mpo_cmp_base.png" --invalid-logr -15

run_job "16) Compare Osimertinib Base Quetzal with composed RTB Quetzal" \
  --smiles "$SMILES_BASE" --smiles2 "$SMILES_DB_peri" --n "$CMP_N" --seed "$SEED" --bins "$BINS" \
  --reward guacamol --reward-smiles perindopril_rings \
  --out "$OUTDIR/17_peri_rings_BASE_cmp_DB.png"  --invalid-logr -5

echo
echo "=================================================================="
echo " all jobs finished. plots in: $OUTDIR/"
echo "=================================================================="
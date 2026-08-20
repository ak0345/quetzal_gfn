#!/usr/bin/env bash
# =============================================================================
# The whole pipeline, in order. Every stage is individually resumable, so this
# can be interrupted and re-run.
#
# On one A100 the full sequence is on the order of days: stage 4 dominates
# (FCD is a neural embedding and EDM stability is quadratic in atom count, over
# ~60 checkpoints x 3 seeds x 5,000 molecules), and stage 3 holds two copies of
# the model plus a retained autograd graph over the rollout.
#
# Start with a smoke test before committing to that:
#   DRY=1 bash scripts/run_all.sh
#   SUBSET=1 REWARDS=nitrogen MAX_EPOCHS=1 N=200 SEEDS=0 bash scripts/run_all.sh
#
# Usage:
#   bash scripts/run_all.sh              # stages 1-8
#   STAGES="4 8" bash scripts/run_all.sh # only these
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

STAGES="${STAGES:-1 2 3 4 5 6 7 8}"

stage_script () {
  case "$1" in
    1) echo "01_train_guides.sh" ;;
    2) echo "02_train_components.sh" ;;
    3) echo "03_finetune.sh" ;;
    4) echo "04_dump_guides.sh" ;;
    5) echo "05_dump_composed.sh" ;;
    6) echo "06_flip_diagnostics.sh" ;;
    7) echo "07_ablations.sh" ;;
    8) echo "08_analysis.sh" ;;
  esac
}

for s in $STAGES; do
  script="$(stage_script "$s")"
  if [[ -z "$script" ]]; then
    echo "[warn] unknown stage '$s' (valid: 1-8)" >&2
    continue
  fi
  hr
  say "STAGE $s -- $script"
  hr
  bash "$HERE/$script" || say "STAGE $s exited non-zero; continuing"
done

hr
say "pipeline finished. Results under ${RESULTS_ROOT}/"
hr

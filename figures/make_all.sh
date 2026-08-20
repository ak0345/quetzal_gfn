#!/usr/bin/env bash
# =============================================================================
# Regenerate every paper figure from the committed artifacts under results/.
#
# No GPU and no checkpoints: every script reads a CSV or JSON that the pipeline
# already wrote. A figure whose input is missing prints the command that
# produces it and is counted as skipped, so a partial run still tells you
# exactly what is outstanding.
#
# Usage:
#   bash figures/make_all.sh                 # -> figures/out/
#   OUT_DIR=paper/figs bash figures/make_all.sh
#   FORMAT=png bash figures/make_all.sh
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OUT_DIR="${OUT_DIR:-out}"
FORMAT="${FORMAT:-pdf}"
PY="${PYTHON:-python3}"
mkdir -p "$OUT_DIR"

ok=0; skipped=0; failed=0
SKIPPED_FIGS=()

run () {
  local fig="$1"; shift
  local script="$1"; shift
  printf '%-6s %-34s ' "$fig" "$script"
  local log
  log="$("$PY" "$script" --out "${OUT_DIR}/${fig}.${FORMAT}" "$@" 2>&1)"
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "ok"; ok=$((ok+1))
  elif [[ $rc -eq 2 ]]; then
    echo "SKIPPED (input missing)"; skipped=$((skipped+1))
    SKIPPED_FIGS+=("$fig")
    echo "$log" | sed 's/^/         /'
  else
    echo "FAILED"; failed=$((failed+1))
    echo "$log" | tail -5 | sed 's/^/         /'
  fi
}

echo "=== paper figures -> ${OUT_DIR}/ ==="
run fig01 make_fig01_landscape.py
run fig02 make_fig02_positional.py
run fig03 make_fig03_capacity.py
# Figure 4 in the paper is the same plot as Figure 8
run fig05 make_fig05_ceiling_bins.py
run fig06 make_fig06_flip_position_raw.py
run fig07 make_fig07_scale_sweep.py
run fig08 make_fig08_tempgain.py
run fig09 make_fig09_singles_weights.py
# Figure 10 needs an explicitly named flow-route report; see its --help
run fig11 make_fig11_collapse_panel.py
run fig12 make_fig12_mpo_components.py
run fig13 make_fig13_sample_efficiency.py
run fig14 make_fig14_quality_vs_score.py

echo
echo "=== $ok written, $skipped skipped, $failed failed ==="
if (( skipped )); then
  echo "skipped: ${SKIPPED_FIGS[*]}"
  echo "each printed the command that produces its input, above."
fi
(( failed == 0 ))

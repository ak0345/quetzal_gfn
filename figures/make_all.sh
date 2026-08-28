#!/usr/bin/env bash
# =============================================================================
# Regenerate every paper figure from the committed artifacts under results/.
#
# Every script reads a CSV or JSON the pipeline already wrote. A figure whose
# input is missing prints the command that produces it and is counted as
# skipped, so a partial run still tells you exactly what is outstanding.
#
# SCOPE. The current figure set covers the GUIDE side only. The fine-tuning and
# composition results are held back, so the figures that are made entirely of
# them -- 3, 9, 11, 12, 13, 14 and the PMO table -- are off by default. Set
# WITH_FINETUNE=1 and/or WITH_COMPOSED=1 to bring them back.
#
# ZALEPLON is a negative control: only three guides were trained, to show the
# objective cannot be guided well. It is therefore kept out of the pooled main
# figures and drawn separately as fig02_zaleplon / fig06_zaleplon.
#
# Usage:
#   bash figures/make_all.sh                 # -> figures/out/
#   OUT_DIR=paper/figs bash figures/make_all.sh
#   FORMAT=png bash figures/make_all.sh
#   WITH_FINETUNE=1 WITH_COMPOSED=1 bash figures/make_all.sh
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OUT_DIR="${OUT_DIR:-out}"
FORMAT="${FORMAT:-pdf}"
PY="${PYTHON:-python3}"
mkdir -p "$OUT_DIR"

WITH_FINETUNE="${WITH_FINETUNE:-0}"
WITH_COMPOSED="${WITH_COMPOSED:-0}"

# Guide runs kept out of the pooled main figures. Zaleplon is the negative
# control; it gets its own panels below instead of being averaged in.
MAIN_EXCLUDE="${MAIN_EXCLUDE:-zaleplon}"
# Both recorded flip temperatures are drawn on every flip figure.
TEMPS="${TEMPS:-1.0,0.3}"

ok=0; skipped=0; failed=0
SKIPPED_FIGS=()

run () {
  local fig="$1"; shift
  local script="$1"; shift
  printf '%-16s %-34s ' "$fig" "$script"
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

note_off () {   # a figure deliberately not drawn, not a missing input
  printf '%-16s %-34s %s\n' "$1" "$2" "off ($3)"
}

echo "=== paper figures -> ${OUT_DIR}/ ==="

# ----------------------------------------------------------------- guide side
FIG01_ARGS=()
[[ "$WITH_FINETUNE" == "1" ]] || FIG01_ARGS+=(--no_finetune)
[[ "$WITH_COMPOSED" == "1" ]] || FIG01_ARGS+=(--no_composed)
run fig01 make_fig01_landscape.py --exclude "$MAIN_EXCLUDE" "${FIG01_ARGS[@]}"

run fig02 make_fig02_positional.py --temps "$TEMPS" --exclude "$MAIN_EXCLUDE"
run fig05 make_fig05_ceiling_bins.py
run fig06 make_fig06_flip_position_raw.py --temps "$TEMPS" --exclude "$MAIN_EXCLUDE"
run fig07 make_fig07_scale_sweep.py
# Figure 4 in the paper is the same plot as Figure 8
run fig08 make_fig08_tempgain.py

# Zaleplon on its own: the negative control, same two figures restricted to it
run fig02_zaleplon make_fig02_positional.py --temps "$TEMPS" --rewards zaleplon
run fig06_zaleplon make_fig06_flip_position_raw.py --temps "$TEMPS" --rewards zaleplon

# Figure 10 is the excluded flow-route run, which is not part of the standard
# pipeline, so it has no default input. Draw it when the reports exist.
FLOW_ROOT="${FLOW_ROOT:-../results/ablations/flow}"
FLOW_REPORTS=()
for t in ${TEMPS//,/ }; do
  [[ -f "$FLOW_ROOT/flip-t${t}/flip_report.json" ]] && \
    FLOW_REPORTS+=("$FLOW_ROOT/flip-t${t}/flip_report.json")
done
if (( ${#FLOW_REPORTS[@]} )); then
  # fig10 reads each report's own config.flip_temp, so the temperatures come
  # from which reports are passed rather than from a --temps flag
  run fig10 make_fig10_excluded_flow.py --report "${FLOW_REPORTS[@]}"
else
  printf '%-16s %-34s ' "fig10" "make_fig10_excluded_flow.py"
  echo "SKIPPED (input missing)"; skipped=$((skipped+1)); SKIPPED_FIGS+=("fig10")
  echo "         no flow-route flip report under $FLOW_ROOT"
  echo "         ROUTE=flow OUT_ROOT=results/ablations/flow RUNS=\"...\" bash scripts/07_ablations.sh flip"
fi

# one chemical-space panel per objective family; `run` takes the output stem
# first, so each family needs its own stem or they overwrite one another
for fam in ${FAMILIES:-osim peri fexo nitrogen zaleplon}; do
  run "fig15_${fam}" make_fig15_chemspace.py --family "$fam"
done

# every family in ONE space, with GEOM and the prior, plus the quantitative
# question fig15 cannot ask: is the objective recoverable from the molecule?
run fig16 make_fig16_joint_chemspace.py --families "${FAMILIES_JOINT:-osim,peri,fexo,zaleplon,nitrogen}"

# the island in that space is a 3D->SMILES failure, not a chemotype. This also
# emits the measurements the chemspace table needs.
CHEMSPACE_JSON="${OUT_DIR}/chemspace.json"
run fig17 make_fig17_decoder_artifact.py \
  --families "${FAMILIES_JOINT:-osim,peri,fexo,zaleplon,nitrogen}" \
  --emit_json "$CHEMSPACE_JSON"

# ------------------------------------------------- composition and fine-tuning
if [[ "$WITH_COMPOSED" == "1" ]]; then
  run fig09 make_fig09_singles_weights.py
else
  note_off fig09 make_fig09_singles_weights.py "composition held back"
fi

if [[ "$WITH_FINETUNE" == "1" ]]; then
  run fig03 make_fig03_capacity.py
  run fig11 make_fig11_collapse_panel.py
  run fig12 make_fig12_mpo_components.py
  run fig13 make_fig13_sample_efficiency.py
  run fig14 make_fig14_quality_vs_score.py

  # the PMO comparison writes a LaTeX table as well as a figure
  printf '%-16s %-34s ' "pmo" "make_table_pmo.py"
  if "$PY" make_table_pmo.py --out "${OUT_DIR}/tab_pmo.tex" \
       --plot "${OUT_DIR}/fig_pmo.${FORMAT}" >/dev/null 2>&1; then
    echo "ok"; ok=$((ok+1))
  else
    echo "SKIPPED (input missing)"; skipped=$((skipped+1)); SKIPPED_FIGS+=("pmo")
  fi
else
  for f in fig03 fig11 fig12 fig13 fig14 pmo; do
    note_off "$f" "-" "fine-tuning held back"
  done
fi

# ------------------------------------------------------------------- tables
# LaTeX tables from the same artifacts. Each skips itself with a note when its
# input is absent, so this is one call rather than one guard per table.
printf '%-16s %-34s ' "tables" "make_tables.py"
if "$PY" make_tables.py --out_dir "$OUT_DIR" --exclude "$MAIN_EXCLUDE" \
     --temps "$TEMPS" --sep_json "$CHEMSPACE_JSON" > /tmp/tables.$$.log 2>&1; then
  echo "ok"; ok=$((ok+1))
  grep '^\[tab\]' /tmp/tables.$$.log | sed 's/^/         /'
else
  echo "FAILED"; failed=$((failed+1))
  tail -5 /tmp/tables.$$.log | sed 's/^/         /'
fi
rm -f /tmp/tables.$$.log

echo
echo "=== $ok written, $skipped skipped, $failed failed ==="
if (( skipped )); then
  echo "skipped: ${SKIPPED_FIGS[*]}"
  echo "each printed the command that produces its input, above."
fi
(( failed == 0 ))

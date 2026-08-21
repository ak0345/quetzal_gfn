#!/usr/bin/env bash
# =============================================================================
# Stage 8 -- scoring, baselines and figures. No GPU training happens here.
#
# Sections (pass one as $1; default runs harvest + baseline):
#
#   harvest   Score every fine-tuning run as a goal-directed benchmark at a
#             fixed oracle budget, from the recorded generation stream. Two
#             conventions are emitted because they are not comparable:
#               unbounded  top-k over every molecule ever generated. The
#                          original GuacaMol leaderboard imposed no oracle
#                          budget, so this is the leaderboard-parity number.
#               budgeted   top-k over the first --budget calls, plus AUC-top-10
#                          over the budget, matching PMO. Since top-k is
#                          non-decreasing in n, AUC_k(B) <= f_k(B); the gap
#                          measures how quickly a method reaches its best
#                          molecules.
#             --extended adds novelty and nearest-neighbour similarity against
#             the training corpus, internal and scaffold diversity, quality-filter
#             pass rate, and the distribution-health panels.
#
#   baseline  Best-of-N curves. A flat benchmark table is compatible with two
#             different states of the world: the high-reward region is absent
#             from the prior, or it is present but rare and steering failed to
#             concentrate mass on it. Best-of-N separates them -- if the prior's
#             curve keeps climbing through the dataset line as N grows, the
#             honest claim is about search rather than support. This also
#             produces the matched-budget dataset baseline itself: the best of
#             10,000 molecules drawn from GEOM-Drugs.
#
#   hists     Log-reward histograms over a SMILES file, for the full MPO
#             objectives and for each leaf scorer separately. The per-component
#             pass is how a dead axis is identified: a component whose
#             log-reward has zero variance over reachable molecules offers no
#             gradient, so a guide trained against it is flat by construction.
#
# Usage:
#   bash scripts/08_analysis.sh
#   bash scripts/08_analysis.sh harvest
#   BENCH=perindopril_rings bash scripts/08_analysis.sh harvest
#   bash scripts/08_analysis.sh baseline
#   bash scripts/08_analysis.sh hists
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

WHICH="${1:-default}"

MOLROOT="${MOLROOT:-${RESULTS_ROOT}/oracle_gfn_mols}"
BENCH="${BENCH:-hard_osimertinib}"
BUDGET="${BUDGET:-10000}"
OUT_DIR="${OUT_DIR:-${MOLROOT}/_results}"

run_job () {
  local label="$1"; shift
  echo; hr; echo ">>> $label"; hr
  if [[ "$DRY" == "1" ]]; then printf '    %s' "$PY"; printf ' %q' "$@"; echo; return 0; fi
  if "$PY" "$@"; then echo "    [done] $label"
  else echo "    [FAILED] $label (continuing)" >&2; fi
}

# ------------------------------- harvest -------------------------------------
if [[ "$WHICH" == "default" || "$WHICH" == "all" || "$WHICH" == "harvest" ]]; then
  mkdir -p "$OUT_DIR"

  # Match the runs trained against this benchmark. The nitrogen sanity run is
  # excluded from the benchmark table -- it was never optimised for an MPO
  # objective and its score is a floor, not a result.
  case "$BENCH" in
    *osimertinib*) MATCH="${MATCH:-osim}" ;;
    *perindopril*) MATCH="${MATCH:-peri}" ;;
    *zaleplon*)    MATCH="${MATCH:-zaleplon}" ;;
    *)             MATCH="${MATCH:-}" ;;
  esac

  DIRS=()
  for d in "$MOLROOT"/rtb-*; do
    [[ -f "$d/molecules.jsonl" ]] || continue
    [[ -n "$MATCH" && "$d" != *"$MATCH"* ]] && continue
    DIRS+=("$d")
  done

  if [[ ${#DIRS[@]} -eq 0 ]]; then
    echo "[skip] no recorded runs under $MOLROOT matching '${MATCH}'"
  else
    say "harvesting ${#DIRS[@]} run(s) at budget=$BUDGET against $BENCH"
    run_job "harvest $BENCH @ $BUDGET" harvest_eval.py \
      --record_dir "${DIRS[@]}" \
      --bench "$BENCH" \
      --budget "$BUDGET" \
      --topk 1,10,100 \
      --out "$OUT_DIR/${BENCH}_budget${BUDGET}.json" \
      --csv "$OUT_DIR/${BENCH}_budget${BUDGET}.csv" \
      --plots_dir "$OUT_DIR/plots" \
      --extended \
      --ref_smiles "$REF_SMILES" \
      --ref_limit "$BUDGET" \
      --dump_best "$OUT_DIR/best"
    say "results -> $OUT_DIR/${BENCH}_budget${BUDGET}.{json,csv}"
  fi
fi

# ------------------------------ baselines ------------------------------------
if [[ "$WHICH" == "default" || "$WHICH" == "all" || "$WHICH" == "baseline" ]]; then
  BON_DIR="${BON_DIR:-${RESULTS_ROOT}/best_of_n}"
  BON_N="${BON_N:-20000}"
  mkdir -p "$BON_DIR"
  warn_missing_ref

  # --generate rolls molecules from the frozen prior and scores them; each
  # --smiles_file adds an already-scored pool as another curve. The dataset
  # baseline is the GEOM-Drugs curve read at n = 10,000.
  for pair in "hard_osimertinib:bon_osim" "perindopril_rings:bon_peri" \
              "zaleplon_with_other_formula:bon_zaleplon"; do
    bench="${pair%%:*}"; tag="${pair##*:}"
    [[ -f "${BON_DIR}/${tag}.json" ]] && { echo "[skip] $tag exists"; continue; }
    run_job "best-of-N | $bench" best_of_n_curve.py \
      --generate "$BON_N" \
      --quetzal_ckpt "$QUETZAL_CKPT" \
      --bench "$bench" \
      --smiles_file "$REF_SMILES" \
      --also_top10 \
      --out "${BON_DIR}/${tag}.json" \
      --plot "${BON_DIR}/${tag}.png"
  done
fi

# ------------------------------ histograms -----------------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "hists" ]]; then
  HIST_DIR="${HIST_DIR:-${RESULTS_ROOT}/reward_hists}"
  HIST_SMILES="${HIST_SMILES:-$REF_SMILES}"
  HIST_N="${HIST_N:-10000}"
  SEED="${SEED:-0}"
  BINS="${BINS:-60}"
  mkdir -p "$HIST_DIR" "$HIST_DIR/components"

  [[ -f "$HIST_SMILES" ]] || { echo "[fatal] SMILES file not found: $HIST_SMILES" >&2; exit 1; }

  # The same fixed seed for every objective, so the reward distributions are
  # drawn on the identical molecules and are directly comparable.
  say "full MPO objectives | file=$HIST_SMILES n=$HIST_N seed=$SEED"
  for pair in "hard_osimertinib:osimertinib" \
              "hard_fexofenadine:fexofenadine" \
              "perindopril_rings:perindopril" \
              "ranolazine_mpo:ranolazine" \
              "sitagliptin_replacement:sitagliptin"; do
    fn="${pair%%:*}"; tag="${pair##*:}"
    run_job "MPO | $tag" smiles_hist.py \
      --smiles "$HIST_SMILES" --n "$HIST_N" --seed "$SEED" --bins "$BINS" \
      --reward guacamol --reward-smiles "$fn" \
      --out "$HIST_DIR/${tag}.png"
  done

  # Leaf scorers, index-aligned with the installed guacamol. Verify with:
  #   python -c "import reward_fn as R; print(R.guacamol_components('osimertinib'))"
  component_labels () {
    case "$1" in
      osimertinib)  echo "sim-FCFP4 sim-ECFP6 tpsa logp" ;;
      ranolazine)   echo "sim-AP logp fluorine-count tpsa" ;;
      sitagliptin)  echo "sim-ECFP4 logp tpsa isomer" ;;
      fexofenadine) echo "sim-AP tpsa logp" ;;
      perindopril)  echo "sim-ECFP4 aromatic-rings" ;;
    esac
  }

  say "per-component objectives"
  for bench in osimertinib ranolazine sitagliptin fexofenadine perindopril; do
    read -r -a labels <<< "$(component_labels "$bench")"
    for (( idx=0; idx<${#labels[@]}; idx++ )); do
      lbl="${labels[$idx]}"
      run_job "$bench component [$idx] $lbl" smiles_hist.py \
        --smiles "$HIST_SMILES" --n "$HIST_N" --seed "$SEED" --bins "$BINS" \
        --reward guacamol_component --benchmark "$bench" --component "$idx" \
        --out "$HIST_DIR/components/${bench}_c${idx}_${lbl}.png"
    done
  done
fi

hr
say "analysis complete"
hr

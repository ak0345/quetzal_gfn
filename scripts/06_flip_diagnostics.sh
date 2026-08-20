#!/usr/bin/env bash
# =============================================================================
# Stage 6 -- the coupled-flip diagnostic, over every checkpoint.
#
# Aggregate reward differences cannot tell a guide that is inert from one that
# acts but is mis-directed. This measures the guide's effect on the decision
# itself: trajectories are rolled by the FROZEN PRIOR, and at every state the
# prior's next-atom distribution is compared with the guided one on the
# identical state, using a shared uniform draw so the two samplers are coupled.
# Because the trajectories belong to the prior, every architecture is compared
# on the same state distribution.
#
# What comes back, per checkpoint:
#   delivered_frac       the guided logits differ from the prior's at all. ~0 is
#                        a wiring bug (residual computed but never applied), not
#                        a bound -- the two are distinguishable only because
#                        delivery is measured separately from the flip rate.
#   argmax_flip_rate     the guided argmax differs from the prior's
#   sample_flip_rate     the coupled draw selects a different atom. The real
#                        "did the decision change" number.
#   mean_total_variation / mean_KL    how much probability mass moved
#   mean_prior_top1_gap  how dominant the prior's top-1 logit is, i.e. how large
#                        a residual would have to be to change the decision
#   flip_rate_by_position  where along the sequence flips happen. Positions no
#                        trajectory ever reached come back null rather than 0.0:
#                        "no molecule was ever this long" is not "the guide never
#                        flipped here", and averaging them together drags the
#                        tail of the curve to a false zero.
#
# Much cheaper than a dump -- no molecules are scored, ~400 trajectories -- so
# it runs over the whole sweep quickly.
#
# Both temperatures run in one process via --also_temp, which avoids reloading
# the checkpoint for the second.
#
# Usage:
#   bash scripts/06_flip_diagnostics.sh
#   PREFIXES="sweep rtb" OUT_DIR=results/flips-guide bash scripts/06_flip_diagnostics.sh
#   TEMPS=1.0 N_TRAJ=2000 bash scripts/06_flip_diagnostics.sh
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/flips-guide}"
PROBE="ablations/single_flip_ablation.py"
AGG="ablations/single_flip_agg.py"

TEMPS="${TEMPS:-1.0 0.3}"
PREFIXES="${PREFIXES:-sweep stability rtb compose}"
CKPT_NAME="${CKPT_NAME:-last.ckpt}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

[[ -f "$PROBE" ]] || { echo "[fatal] $PROBE not found" >&2; exit 2; }
[[ -d "$CKPT_ROOT" ]] || { echo "[fatal] $CKPT_ROOT is not a directory" >&2; exit 2; }

# extra flags after -- are forwarded verbatim to the probe
EXTRA=()
if [[ "${1:-}" == "--" ]]; then shift; EXTRA=( "$@" ); fi
[[ -n "${N_REPORT_POS:-}" ]] && EXTRA+=( --n_report_pos "$N_REPORT_POS" )
[[ -n "${N_TRAJ:-}" ]]       && EXTRA+=( --n_traj "$N_TRAJ" )
[[ -n "${GUIDE_SOURCE:-}" ]] && EXTRA+=( --guide_source "$GUIDE_SOURCE" )

read -r -a TEMP_ARR <<< "$TEMPS"
if [[ ${#TEMP_ARR[@]} -eq 0 || ${#TEMP_ARR[@]} -gt 2 ]]; then
  echo "[fatal] TEMPS must name 1 or 2 temperatures (--flip_temp, --also_temp); got: $TEMPS" >&2
  exit 2
fi
TEMP_ARGS=( --flip_temp "${TEMP_ARR[0]}" )
[[ ${#TEMP_ARR[@]} -eq 2 ]] && TEMP_ARGS+=( --also_temp "${TEMP_ARR[1]}" )

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Only run directories whose name starts with one of $PREFIXES are picked up, so
# wandb/, tensorboard/ and unrelated runs in the same log folder are left alone.
prefix_args=()
for p in $PREFIXES; do
  [[ ${#prefix_args[@]} -gt 0 ]] && prefix_args+=( -o )
  prefix_args+=( -name "${p}*" )
done
RUN_DIRS=()
while IFS= read -r _d; do
  [[ -n "$_d" ]] && RUN_DIRS+=("$_d")
done < <(find "$CKPT_ROOT" -mindepth 1 -maxdepth 1 -type d \
           \( "${prefix_args[@]}" \) | sort)
if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "[fatal] no run dirs under $CKPT_ROOT starting with: $PREFIXES" >&2
  exit 1
fi

# Runs are labelled after their top-level directory, not the checkpoint's parent
# -- every run has a directory called "checkpoints", so that would collide.
LABELS=(); CKPTS=()
for d in "${RUN_DIRS[@]}"; do
  label="$(basename "$d")"
  ckpt="$(find "$d" -maxdepth 3 -type f -name "$CKPT_NAME" | sort | head -n1)"
  if [[ -z "$ckpt" ]]; then
    ckpt="$(find "$d" -maxdepth 3 -type f -name '*.ckpt' -exec ls -t {} + 2>/dev/null | head -n1)"
    [[ -n "$ckpt" ]] && echo "[warn] $label: no $CKPT_NAME, using $(basename "$ckpt")" >&2
  fi
  [[ -z "$ckpt" ]] && { echo "[warn] $label: no .ckpt found, skipping" >&2; continue; }
  LABELS+=( "$label" ); CKPTS+=( "$ckpt" )
done
[[ ${#CKPTS[@]} -eq 0 ]] && { echo "[fatal] matched run dirs but none held a .ckpt" >&2; exit 1; }

say "${#CKPTS[@]} run(s) matching [$PREFIXES], temps: $TEMPS -> $OUT_DIR"
OK=0; FAIL=0; SKIP=0; FAILED_RUNS=()

for i in "${!CKPTS[@]}"; do
  label="${LABELS[$i]}"
  ckpt="${CKPTS[$i]}"
  # printf, not echo: a trailing newline would become a stray underscore
  safe="$(printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_')"
  json="$OUT_DIR/flip_report_${safe}.json"

  if [[ "$SKIP_EXISTING" == "1" && -s "$json" ]]; then
    echo "[skip] $label"; SKIP=$((SKIP+1)); continue
  fi

  echo "[run ] $label  <- $ckpt"
  if [[ "$DRY" == "1" ]]; then continue; fi

  # keep going: one broken checkpoint should not abort the sweep
  if "$PY" "$PROBE" \
        --ckpt "$ckpt" --label "$label" --report_tag "$safe" \
        --out_dir "$OUT_DIR" "${TEMP_ARGS[@]}" "${EXTRA[@]}" \
        > "$LOG_DIR/${safe}.log" 2>&1; then
    echo "[ ok ] $label -> $json"; OK=$((OK+1))
  else
    echo "[FAIL] $label (see $LOG_DIR/${safe}.log)" >&2
    tail -n 15 "$LOG_DIR/${safe}.log" | sed 's/^/       | /' >&2
    FAIL=$((FAIL+1)); FAILED_RUNS+=("$label")
  fi
done

hr
say "flip probes: ok=$OK failed=$FAIL skipped=$SKIP -> $OUT_DIR"
[[ $FAIL -gt 0 ]] && echo "[warn] failed: ${FAILED_RUNS[*]}" >&2

# ------------------------------ aggregate ------------------------------------
# Group summaries are pooled from the raw counts each report carries, summing
# numerators and denominators and dividing once at the end. Averaging per-run
# rates would weight a run of 400 states the same as one of 40,000, which at
# deep sequence positions -- where only a few long molecules survive -- is a
# large difference.
#
# Two groupings are produced at each temperature. Pooling by guide is the
# architecture comparison; pooling by reward shows the same decay holds across
# objectives. The per-position curves are what Figure 2 averages.
if [[ "$DRY" != "1" ]]; then
  AGGS="${OUT_DIR}/_aggs"
  mkdir -p "$AGGS"
  for t in "${TEMP_ARR[@]}"; do
    for grouping in guide reward; do
      say "aggregating: temp=$t grouped by $grouping"
      $PY "$AGG" \
        --flips_root "$OUT_DIR" \
        --temp "$t" \
        --group_by "$grouping" \
        --out "${AGGS}/flip_table_t${t}_by_${grouping}.csv" \
        --pos_out "${AGGS}/flip_by_position_t${t}_by_${grouping}.csv" \
        --plot
    done
  done
  say "tables and per-position curves: ${AGGS}/"
fi
hr

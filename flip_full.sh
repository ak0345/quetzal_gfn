#!/usr/bin/env bash
# run_all_ckpts.sh -- run single_flip_ablation.py once per checkpoint directory.
#
#   ./run_all_ckpts.sh CKPT_ROOT [OUT_DIR] [-- extra flags for the probe...]
#
# CKPT_ROOT is a log folder of run directories. Only runs whose NAME starts with
# one of $PREFIXES (default: sweep, rtb) are picked up, so unrelated dirs in the
# same log folder are left alone:
#   logs/quetzal-gfn/legit/
#     sweep-fexo-base-db-replay_on-b10/checkpoints/last.ckpt   <- taken
#     rtb-qm9-seed1/checkpoints/epoch=4-step=489.ckpt          <- taken
#     wandb/ , tensorboard/ , old-debug/                       <- ignored
#
# Each run is labelled after its top-level dir, NOT the checkpoint's parent --
# every run has a dir called "checkpoints", so that would collide for all of them.
# One report per run lands in OUT_DIR as flip_report_<run>.json, each carrying
# raw per-position counts so they can be pooled into a combined graph later.
#
# examples:
#   ./run_all_ckpts.sh logs/quetzal-gfn/legit flips
#   ./run_all_ckpts.sh logs/quetzal-gfn/legit flips -- --n_traj 2000 --n_report_pos 64
#   TEMPS="1.0 0.3" ./run_all_ckpts.sh logs/quetzal-gfn/legit flips -- --progress
#   N_REPORT_POS=256 ./run_all_ckpts.sh logs/quetzal-gfn/legit flips
#
# env knobs:
#   PREFIXES      run-name prefixes to include (default "sweep rtb")
#   CKPT_NAME     preferred checkpoint filename in a run (default last.ckpt)
#   TEMPS         1 or 2 --flip_temp values (default "1.0"); a 2nd is passed as
#                 --also_temp so both run in ONE process (one checkpoint load)
#   N_REPORT_POS  positions to report (default: the probe's own default, 64)
#   N_TRAJ        trajectories per run (default: the probe's own default)
#   GUIDE_SOURCE  ema | policy (default: the probe's own default, ema)
#   PYTHON        interpreter (default python)
#   SKIP_EXISTING=1  don't redo a run whose JSON is already there
set -uo pipefail

CKPT_ROOT="${1:-}"
OUT_DIR="${2:-flip_results}"
if [[ -z "$CKPT_ROOT" ]]; then
  echo "usage: $0 CKPT_ROOT [OUT_DIR] [-- extra probe flags]" >&2
  exit 2
fi
if [[ ! -d "$CKPT_ROOT" ]]; then
  echo "[err] not a directory: $CKPT_ROOT" >&2
  exit 2
fi
shift $(( $# > 2 ? 2 : $# ))
[[ "${1:-}" == "--" ]] && shift
EXTRA=( "$@" )                      # forwarded verbatim to the probe

PYTHON="${PYTHON:-python}"
TEMPS="${TEMPS:-1.0 0.3}"
PREFIXES="${PREFIXES:-sweep rtb}"
CKPT_NAME="${CKPT_NAME:-last.ckpt}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="$HERE/single_flip_ablation.py"
[[ -f "$PROBE" ]] || { echo "[err] cannot find $PROBE" >&2; exit 2; }

# optional env knobs -> probe flags (only when set, so the probe's own defaults
# stay authoritative otherwise)
[[ -n "${N_REPORT_POS:-}" ]] && EXTRA+=( --n_report_pos "$N_REPORT_POS" )
[[ -n "${N_TRAJ:-}" ]]       && EXTRA+=( --n_traj "$N_TRAJ" )
[[ -n "${GUIDE_SOURCE:-}" ]] && EXTRA+=( --guide_source "$GUIDE_SOURCE" )

# temperatures: the probe does both in one process via --also_temp, which saves
# reloading the checkpoint for the second one
read -r -a TEMP_ARR <<< "$TEMPS"
if [[ ${#TEMP_ARR[@]} -eq 0 ]]; then
  echo "[err] TEMPS is empty" >&2; exit 2
elif [[ ${#TEMP_ARR[@]} -gt 2 ]]; then
  echo "[err] the probe takes at most 2 temperatures (--flip_temp, --also_temp);"\
       "got: $TEMPS" >&2
  exit 2
fi
TEMP_ARGS=( --flip_temp "${TEMP_ARR[0]}" )
[[ ${#TEMP_ARR[@]} -eq 2 ]] && TEMP_ARGS+=( --also_temp "${TEMP_ARR[1]}" )

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs"; mkdir -p "$LOG_DIR"

# collect run dirs whose name starts with one of $PREFIXES, sorted for a stable
# order across sweeps
prefix_args=()
for p in $PREFIXES; do
  [[ ${#prefix_args[@]} -gt 0 ]] && prefix_args+=( -o )
  prefix_args+=( -name "${p}*" )
done
mapfile -t RUN_DIRS < <(find "$CKPT_ROOT" -mindepth 1 -maxdepth 1 -type d \
                          \( "${prefix_args[@]}" \) | sort)
if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "[err] no run dirs under $CKPT_ROOT starting with: $PREFIXES" >&2
  exit 1
fi

# resolve one checkpoint per run: prefer $CKPT_NAME, else the newest .ckpt found
# anywhere under the run (epoch=N-step=M.ckpt layouts vary between sweeps)
LABELS=(); CKPTS=()
for d in "${RUN_DIRS[@]}"; do
  label="$(basename "$d")"
  ckpt="$(find "$d" -maxdepth 3 -type f -name "$CKPT_NAME" | sort | head -n1)"
  if [[ -z "$ckpt" ]]; then
    ckpt="$(find "$d" -maxdepth 3 -type f -name '*.ckpt' -printf '%T@ %p\n' \
             | sort -rn | head -n1 | cut -d" " -f2-)"
    [[ -n "$ckpt" ]] && echo "[warn] $label: no $CKPT_NAME, using $(basename "$ckpt")" >&2
  fi
  if [[ -z "$ckpt" ]]; then
    echo "[warn] $label: no .ckpt found, skipping" >&2
    continue
  fi
  LABELS+=( "$label" ); CKPTS+=( "$ckpt" )
done
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "[err] matched ${#RUN_DIRS[@]} run dir(s) but none contained a .ckpt" >&2
  exit 1
fi

echo "[sweep] ${#CKPTS[@]} run(s) matching [$PREFIXES], temps: $TEMPS"
OK=0; FAIL=0; SKIP=0; FAILED_RUNS=()

for i in "${!CKPTS[@]}"; do
  label="${LABELS[$i]}"
  ckpt="${CKPTS[$i]}"
  # run names go in filenames; printf (not echo) so the trailing newline doesn't
  # get translated into a stray underscore
  safe="$(printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_')"
  json="$OUT_DIR/flip_report_${safe}.json"

  if [[ "$SKIP_EXISTING" == "1" && -s "$json" ]]; then
    echo "[skip] $label (already have $json)"
    SKIP=$((SKIP+1)); continue
  fi

  echo "[run ] $label  <- $ckpt"
  if "$PYTHON" "$PROBE" \
        --ckpt "$ckpt" \
        --label "$label" \
        --report_tag "$safe" \
        --out_dir "$OUT_DIR" \
        "${TEMP_ARGS[@]}" \
        "${EXTRA[@]}" > "$LOG_DIR/${safe}.log" 2>&1; then
    echo "[ ok ] $label -> $json"
    OK=$((OK+1))
  else
    # keep going: one broken checkpoint shouldn't abort the sweep
    echo "[FAIL] $label (see $LOG_DIR/${safe}.log)" >&2
    tail -n 15 "$LOG_DIR/${safe}.log" | sed 's/^/       | /' >&2
    FAIL=$((FAIL+1)); FAILED_RUNS+=("$label")
  fi
done

echo
echo "[sweep] ok=$OK failed=$FAIL skipped=$SKIP -> $OUT_DIR"
if [[ $FAIL -gt 0 ]]; then
  echo "[sweep] failed: ${FAILED_RUNS[*]}" >&2
  exit 1
fi
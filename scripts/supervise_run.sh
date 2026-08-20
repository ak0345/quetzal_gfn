#!/usr/bin/env bash
# =============================================================================
# supervise_run.sh -- run one long training job to completion, restarting it if
# it stalls or dies.
#
#   bash scripts/supervise_run.sh NAME -- python rtb_finetune.py --name NAME ...
#
# Pairs with hang_guard.py, which is wired into gflow.py and rtb_finetune.py:
# the in-process watchdog exits 17 when no batch has completed in
# --guard_stall_minutes, and this script restarts from the last checkpoint.
#
# RESTART SAFETY. Both trainers resume from the newest checkpoint in their run
# directory automatically, so a restart needs no extra flags.
#
# For rtb_finetune the molecule log is checkpoint-aware: on_save_checkpoint
# stores record_count, and on_load_checkpoint calls recorder.rollback_to(),
# dropping every record at or past that index. Lightning replays the batches
# between the last checkpoint and the crash, so without that rollback those
# molecules would appear twice under different indices and any budget slice on
# `i` would spend oracle calls on duplicates. gflow.py records no molecule
# stream at all, so restart is unconditionally safe there.
#
# exit 17 = the in-process stall watchdog fired (hang_guard.StallGuard)
# exit 0  = training finished; the supervisor stops
# anything else = crash; retried up to MAX_RETRIES with a short backoff
#
# env knobs:
#   MAX_RETRIES   attempts before giving up (default 20)
#   BACKOFF       seconds between attempts (default 60)
#   MIN_RUNTIME   an attempt shorter than this counts as a fast-failure loop
#                 rather than a transient stall (default 120s)
#   LOG_DIR       where per-attempt logs go (default logs/supervisor/NAME)
#
# Examples:
#   bash scripts/supervise_run.sh rtb-proj-osim-b10 -- \
#     python rtb_finetune.py --name rtb-proj-osim-b10 --finetune_scope proj \
#       --reward guacamol --reward_smiles hard_osimertinib --reward_beta 10
#
#   # supervise a whole stage by pointing it at the driver
#   bash scripts/supervise_run.sh guides -- bash scripts/01_train_guides.sh
# =============================================================================
set -uo pipefail

NAME="${1:-}"
if [[ -z "$NAME" || "${2:-}" != "--" ]]; then
  echo "usage: $0 NAME -- <command to run>" >&2
  exit 2
fi
shift 2
CMD=( "$@" )

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

MAX_RETRIES="${MAX_RETRIES:-20}"
BACKOFF="${BACKOFF:-60}"
MIN_RUNTIME="${MIN_RUNTIME:-120}"
LOG_DIR="${LOG_DIR:-logs/supervisor/$NAME}"
mkdir -p "$LOG_DIR"

echo "[sup] $NAME | up to $MAX_RETRIES attempts | logs in $LOG_DIR"
echo "[sup] command: ${CMD[*]}"

attempt=0
fast_fails=0
while (( attempt < MAX_RETRIES )); do
  attempt=$((attempt+1))
  log="$LOG_DIR/attempt-$(printf '%03d' "$attempt").log"
  echo "[sup] attempt $attempt/$MAX_RETRIES -> $log"
  start=$SECONDS
  "${CMD[@]}" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  dur=$((SECONDS - start))

  if [[ $rc -eq 0 ]]; then
    echo "[sup] finished cleanly after $attempt attempt(s)"
    exit 0
  fi

  if [[ $rc -eq 17 ]]; then
    echo "[sup] STALL detected after ${dur}s; restarting from last checkpoint" >&2
    grep -n "STALL" "$log" | tail -2 >&2
  else
    echo "[sup] exit $rc after ${dur}s" >&2
    tail -n 20 "$log" | sed 's/^/     | /' >&2
  fi

  # a run that dies within MIN_RUNTIME is almost certainly a config error, and
  # retrying it 20 times just fills the disk with identical tracebacks
  if (( dur < MIN_RUNTIME )); then
    fast_fails=$((fast_fails+1))
    if (( fast_fails >= 3 )); then
      echo "[sup] 3 failures inside ${MIN_RUNTIME}s each -- this is not a transient" >&2
      echo "[sup] stall. Giving up; read $log" >&2
      exit 1
    fi
  else
    fast_fails=0
  fi

  # free the GPU before the next attempt: a killed process can leave memory
  # held for a few seconds
  sleep "$BACKOFF"
done

echo "[sup] exhausted $MAX_RETRIES attempts" >&2
exit 1

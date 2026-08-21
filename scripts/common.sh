#!/usr/bin/env bash
# Shared settings and helpers sourced by every driver in scripts/.
#
# Every driver resolves paths relative to the repo root, so they can be invoked
# from anywhere. Python modules live flat at the repo root, so PYTHONPATH must
# include it for anything under ablations/ to import gflow, reward_fn, etc.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$REPO_ROOT"

PY="${PYTHON:-python}"

# The frozen prior. Everything in this project trains on top of it.
QUETZAL_CKPT="${QUETZAL_CKPT:-checkpoints/geom.ckpt}"

# Lightning writes run directories here as <CKPT_ROOT>/<run name>/checkpoints/.
CKPT_ROOT="${CKPT_ROOT:-logs/quetzal-gfn}"

# GEOM-Drugs SMILES, used as the FCD/descriptor reference and as the
# best-of-10k dataset baseline.
REF_SMILES="${REF_SMILES:-reference/geom_drugs_smiles.txt}"

RESULTS_ROOT="${RESULTS_ROOT:-results}"
LOG_ROOT="${LOG_ROOT:-logs/drivers}"

# Hang guard (hang_guard.py), passed to every training entry point. A run whose
# batches stop progressing for GUARD_STALL_MINUTES dumps its stacks, flushes the
# molecule log and exits 17, which the retry loop treats as recoverable.
GUARD_STALL_MINUTES="${GUARD_STALL_MINUTES:-10}"
GUARD_REWARD_TIMEOUT="${GUARD_REWARD_TIMEOUT:-20}"
GUARD_FLAGS="--guard_stall_minutes ${GUARD_STALL_MINUTES} --guard_reward_timeout ${GUARD_REWARD_TIMEOUT}"

DRY="${DRY:-0}"                       # 1 -> print commands, run nothing
MAX_PARALLEL="${MAX_PARALLEL:-1}"     # concurrent jobs on the shared GPU
NUM_GPUS="${NUM_GPUS:-1}"             # for round-robin CUDA_VISIBLE_DEVICES

mkdir -p "$LOG_ROOT"

say () { echo "[$(date '+%H:%M:%S')] $*"; }

hr () { echo "==================================================================="; }

# Block until running background jobs drop below MAX_PARALLEL. Each concurrent
# training holds its own copy of the frozen prior in VRAM, so raising this past
# what the GPU fits will OOM rather than run faster.
throttle () {
  while [[ "$(jobs -r -p | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    wait -n 2>/dev/null || sleep 5
  done
}

# Resolve a run's checkpoint: prefer last.ckpt, else the most recent *.ckpt.
resolve_ckpt () {
  local name="$1"
  local c="${CKPT_ROOT}/${name}/checkpoints/${CKPT_FILE:-last.ckpt}"
  if [[ -f "$c" ]]; then echo "$c"; return 0; fi
  local alt
  alt=$(ls -t "${CKPT_ROOT}/${name}/checkpoints/"*.ckpt 2>/dev/null | head -1)
  [[ -n "$alt" ]] && { echo "$alt"; return 0; }
  return 1
}

require_prior () {
  if [[ ! -f "$QUETZAL_CKPT" ]]; then
    echo "[fatal] frozen prior not found at $QUETZAL_CKPT" >&2
    echo "        download it, or set QUETZAL_CKPT=/path/to/geom.ckpt" >&2
    exit 1
  fi
}

warn_missing_ref () {
  if [[ ! -f "$REF_SMILES" ]]; then
    echo "[warn] REF_SMILES=$REF_SMILES not found; FCD and descriptor comparisons"
    echo "       against GEOM-Drugs will be skipped. Run: python data_smiles.py"
  fi
}

#!/usr/bin/env bash
#
# sweep_hidden_lowbeta.sh -- both RTB and DB hidden-guide runs plateau at beta=10
# (loss stuck in the hundreds, guided ~= base). The loss floor in the hundreds is
# the "beta*logR target too large to fit" signature. This sweeps LOW beta values
# to test whether a fittable target lets the hidden guide actually steer.
#
# For each beta in BETAS, trains a hidden-guide osim run (db by default) and dumps
# the steering delta. Watch: does the loss drop NEAR ZERO (not plateau in hundreds)
# and does top10_delta go positive?
#
# Usage:
#   bash sweep_hidden_lowbeta.sh
#   BETAS="1 2 4" OBJ=db bash sweep_hidden_lowbeta.sh
#   OBJ=rtb bash sweep_hidden_lowbeta.sh
#   DRY=1 bash sweep_hidden_lowbeta.sh
#
set -uo pipefail

SCRIPT="${SCRIPT:-gflow.py}"
PY="${PYTHON:-python}"
QUETZAL_CKPT="${QUETZAL_CKPT:-geom.ckpt}"
REWARD_SMILES="${REWARD_SMILES:-hard_osimertinib}"

BETAS="${BETAS:-1 2 4}"
OBJ="${OBJ:-db}"
SAMPLE_TEMP="${SAMPLE_TEMP:-2.0}"
RAND_EPS="${RAND_EPS:-0.2}"
LR="${LR:-1e-4}"
LOGZ_LR="${LOGZ_LR:-1e-2}"
MAX_EPOCHS="${MAX_EPOCHS:-14}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
EVAL_N="${EVAL_N:-200}"
HIST_EVERY="${HIST_EVERY:-7}"
DRY="${DRY:-0}"

for beta in $BETAS; do
  name="gfn-quetzal-osim-hidden-${OBJ}-beta${beta}"
  echo ">>> $name (hidden guide, beta=$beta, $OBJ)"
  args=(
    --name "$name"
    --quetzal_ckpt "$QUETZAL_CKPT"
    --objective "$OBJ"
    --reward guacamol --reward_smiles "$REWARD_SMILES"
    --reward_beta "$beta"
    --sample_temp "$SAMPLE_TEMP" --rand_eps "$RAND_EPS"
    --lr "$LR" --logz_lr "$LOGZ_LR"
    --max_epochs "$MAX_EPOCHS" --steps_per_epoch "$STEPS_PER_EPOCH"
    --use_hidden_guide
    --eval_n "$EVAL_N" --hist_every_n_epochs "$HIST_EVERY" --final_n 0
    --devices 1
  )
  if [[ "$DRY" == "1" ]]; then
    printf '    %s %s' "$PY" "$SCRIPT"; printf ' %q' "${args[@]}"; printf '\n'
    continue
  fi
  "$PY" "$SCRIPT" "${args[@]}"
  # quick steering check after each
  ckpt="logs/quetzal-gfn/${name}/checkpoints/last.ckpt"
  if [[ -f "$ckpt" ]]; then
    echo "    [ablate] $name"
    "$PY" ablate_hidden_guide.py --ckpt "$ckpt" --out_dir "dumps/ablate_${name}" || true
  fi
done

echo "=============================================================="
echo " Compare across beta:"
echo "   * loss: does it drop NEAR ZERO at low beta (vs plateau ~100s at beta=10)?"
echo "   * A4 top10_delta: does it go POSITIVE at low beta?"
echo "   * A1 delta_norm: is the delta actually growing?"
echo " If low beta converges + steers -> beta was the whole problem."
echo " If low beta STILL plateaus/flat -> the gradient isn't reaching the delta;"
echo "   the problem is deeper than beta (check A1/A3 in the ablation)."
echo "=============================================================="
#!/usr/bin/env bash
# =============================================================================
# Stage 7 -- mechanism ablations.
#
# Stage 6 measures whether a guide changes a decision. These probes ask why it
# does not, and localise the answer along the causal chain
#
#     residual magnitude -> logit change -> atom flip -> reward movement
#
# Sections (pass one as $1; default runs them all):
#
#   ceiling   Bin states by the prior's top-1 logit margin and report the
#             sampled-flip rate within each bin. This is the direct ceiling
#             measurement: the flip rate falls to zero above a margin of roughly
#             4, while ~72% of decisions sit above 8. The same run scores prior
#             samples on each leaf component of the MPO objective, separating a
#             DEAD AXIS (near-zero variance over reachable molecules, so no
#             guide can steer it and a flat curve is expected) from a genuinely
#             under-trained one.
#
#   guide     Effect size against the levers that control it: residual norm
#             relative to the prior's logit norm, per-step KL, and what happens
#             when the trained residual is multiplied by a constant factor at
#             sampling time. Effect size rises to a maximum near 4x and then
#             falls, with the mean reward shift turning negative -- scaling past
#             that point leaves the valid region faster than it gains reward.
#
#   flip      The same coupled-flip chain as stage 6, run on a composed sampler
#             rather than a single guide.
#
#   singles   Each component guide sampled alone under a direct policy, then the
#             composed sampler under different weight vectors, from equal weights
#             to all mass on one component. Concentrating weight changes the
#             effect size without lifting it out of the reported band.
#
#   tempgain  Read T(h) and g(h) directly off trained checkpoints, binned by the
#             prior's margin. Because the forward pass applies clamp(T, min=1),
#             a learned T below 1 has no effect and no gradient; the values
#             observed lie between 0.73 and 0.80 everywhere, so the temperature
#             mechanism was inactive and TEMPGAIN runs report a gain-scaled
#             residual guide rather than a test of prior softening.
#
#   rollout   Re-runs guided rollouts with instrumentation: the reward spread
#             over the molecules each policy actually samples (a component can
#             vary over GEOM yet be flat over the training distribution), the
#             residual norm against the logit headroom needed to flip the median
#             decision, and the reward reached at training versus eval sampling
#             settings.
#
#   logs      Training curves pulled from W&B -- terminal-loss trajectory,
#             fraction of valid terminals, and the interior-to-terminal loss
#             ratio -- to separate "the flow head never learned the reward
#             direction" from "training converged and the weak steering is the
#             prior". Needs WANDB_ENTITY.
#
#   single    Two probes that take one checkpoint rather than a component list:
#             the hidden-guide causal chain, and whether the reward lives in the
#             atom types a guide can change or in the frozen coordinates it
#             cannot.
#
# Usage:
#   bash scripts/07_ablations.sh
#   bash scripts/07_ablations.sh ceiling
#   FROM_TAG=db-beta20-t1.0-e0.05 bash scripts/07_ablations.sh
#   WANDB_ENTITY=you bash scripts/07_ablations.sh logs
#   SINGLE_CKPT=logs/quetzal-gfn/sweep-osim-hidden-db-replay_off-b10/checkpoints/last.ckpt \
#     bash scripts/07_ablations.sh single
#   DRY=1 bash scripts/07_ablations.sh
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

WHICH="${1:-all}"
A="ablations"

OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/ablations}"
BENCH="${BENCH:-osimertinib}"
COMPONENTS="${COMPONENTS:-0 2 3}"
BETA="${BETA:-10}"
ROUTE="${ROUTE:-policy}"
PRODUCT_KIND="${PRODUCT_KIND:-harmonic}"

NSAMPLES="${NSAMPLES:-1000}"
NTRAJ="${NTRAJ:-400}"
NSCORE="${NSCORE:-1500}"
SCALES="${SCALES:-0.5,1,2,4,8}"
WEIGHT_SETS="${WEIGHT_SETS:-0.333,0.333,0.334;1,0,0;0,0,1;0,1,0;0.2,0.2,0.6;0.4,0,0.6}"

# The sampling settings the guides were trained with, for the rollout probe's
# train-versus-eval comparison.
TRAIN_TEMP="${TRAIN_TEMP:-2.0}"
TRAIN_EPS="${TRAIN_EPS:-0.2}"

# W&B is only needed by the `logs` section; no default entity is assumed.
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-quetzal-gfn}"

# Which component runs to probe. Defaults to the composition guides from
# stage 2; override RUN_PREFIX to point at a warm-started or retrained set.
RUN_PREFIX="${RUN_PREFIX:-compose-osim-c}"
RUN_SUFFIX="${RUN_SUFFIX:--hidden-db-b${BETA}}"

# ---------------------- resolve the guide checkpoints ------------------------
# Two ways in.
#
#   RUNS="name1 name2 ..."  an explicit list of run names under $CKPT_ROOT, or
#                           GUIDE_CKPTS="/path/a.ckpt,/path/b.ckpt" for paths
#                           outside it. Use this to probe the SWEEP guides,
#                           which are not indexed by component. The margin,
#                           residual-scale and temperature figures all work on
#                           any set of guides.
#
#   COMPONENTS + RUN_PREFIX/RUN_SUFFIX   the per-component guides from stage 2,
#                           which is what the composition figure needs, since
#                           "each component alone" and "composition weights" are
#                           only meaningful for a set of component teachers.
CKPTS="${GUIDE_CKPTS:-}"; LABELS="${GUIDE_LABELS:-}"
BETAS=""; WEIGHTS=""; EVAL_COMPONENTS=""

if [[ -n "${RUNS:-}" ]]; then
  for name in $RUNS; do
    ckpt=$(resolve_ckpt "$name") || { echo "[warn] no checkpoint for $name"; continue; }
    CKPTS="${CKPTS:+$CKPTS,}${ckpt}"
    LABELS="${LABELS:+$LABELS,}${name}"
  done
elif [[ -z "$CKPTS" ]]; then
  for c in $COMPONENTS; do
    name="${RUN_PREFIX}${c}${RUN_SUFFIX}"
    ckpt=$(resolve_ckpt "$name") || { echo "[warn] no checkpoint for $name"; continue; }
    CKPTS="${CKPTS:+$CKPTS,}${ckpt}"
    LABELS="${LABELS:+$LABELS,}c${c}"
    EVAL_COMPONENTS="${EVAL_COMPONENTS:+$EVAL_COMPONENTS,}gcomp:${BENCH}:${c}=c${c}"
  done
fi

# betas and weights are per-guide lists and must match the checkpoint count
NG=$(awk -F, '{print NF}' <<< "${CKPTS:-}")
[[ -z "$CKPTS" ]] && NG=0
if (( NG > 0 )); then
  W=$(awk -v n="$NG" 'BEGIN{printf "%.3f", 1.0/n}')
  for ((i=0; i<NG; i++)); do
    BETAS="${BETAS:+$BETAS,}${BETA}"
    WEIGHTS="${WEIGHTS:+$WEIGHTS,}${W}"
  done
fi
# the per-component eval spec is only defined for component guides; fall back to
# the assembled objective so the probes still have something to score against
if [[ -z "$EVAL_COMPONENTS" ]]; then
  EVAL_COMPONENTS="${EVAL_REWARDS:-guacamol:hard_${BENCH}=${BENCH}_MPO}"
fi

need_components () {
  if [[ -z "$CKPTS" ]]; then
    echo "[skip] no guide checkpoints found."
    echo "       point at existing runs:  RUNS=\"run-a run-b\" bash scripts/07_ablations.sh $WHICH"
    echo "       or paths:                GUIDE_CKPTS=/a.ckpt,/b.ckpt GUIDE_LABELS=a,b ..."
    echo "       or train components:     bash scripts/02_train_components.sh"
    return 1
  fi
  return 0
}

runp () {   # runp <label> <script> <args...>
  local label="$1"; shift
  local script="$1"; shift
  echo ">>> [$label] route=$ROUTE -> ${*: -1}"
  if [[ "$DRY" == "1" ]]; then
    printf '    %s %s' "$PY" "$script"; printf ' %q' "$@"; printf '\n'
    return 0
  fi
  "$PY" "$script" "$@" || echo "    [FAILED] $label (continuing)" >&2
}

# ------------------------------- ceiling -------------------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "ceiling" ]] && need_components; then
  hr; echo "ablate_ceiling -- margin binning + component variance"; hr
  # --weights is not optional even though this probe does not sweep them: the
  # Composer asserts one weight per guide, and MultiConfig's default carries the
  # four component teachers, so any other guide count fails without it.
  runp ceiling "$A/ablate_ceiling.py" \
    --guide_ckpts "$CKPTS" --guide_labels "$LABELS" \
    --eval_rewards "$EVAL_COMPONENTS" --train_betas "$BETAS" \
    --weights "$WEIGHTS" \
    --route "$ROUTE" --n_traj "$NTRAJ" --n_score "$NSCORE" \
    --out_dir "${OUT_ROOT}/ceiling"
fi

# -------------------------------- guide --------------------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "guide" ]] && need_components; then
  hr; echo "ablate_guide -- residual magnitude and the residual-scale sweep"; hr
  for kind in harmonic poe; do
    runp "guide/$kind" "$A/ablate_guide.py" \
      --guide_ckpts "$CKPTS" --guide_labels "$LABELS" \
      --route "$ROUTE" --operator product --product_kind "$kind" \
      --weights "$WEIGHTS" --train_betas "$BETAS" \
      --n_samples "$NSAMPLES" --residual_scales "$SCALES" --do_singles \
      --out_dir "${OUT_ROOT}/guide-${kind}"
  done
fi

# --------------------------------- flip --------------------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "flip" ]] && need_components; then
  hr; echo "ablate_logit_flip -- coupled-flip chain on the composed sampler"; hr
  for temp in 1.0 0.3; do
    runp "flip/t$temp" "$A/ablate_logit_flip_compose.py" \
      --guide_ckpts "$CKPTS" --guide_labels "$LABELS" \
      --train_betas "$BETAS" --n_samples "$NSAMPLES" --weights "$WEIGHTS" \
      --route "$ROUTE" --n_traj "$NTRAJ" --flip_temp "$temp" \
      --out_dir "${OUT_ROOT}/flip-t${temp}"
  done
fi

# ------------------------------- singles -------------------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "singles" ]] && need_components; then
  hr; echo "ablate_singles_weights -- per-component effect and weight skew"; hr
  for kind in harmonic poe; do
    runp "singles/$kind" "$A/ablate_singles_weights.py" \
      --guide_ckpts "$CKPTS" --guide_labels "$LABELS" \
      --eval_rewards "$EVAL_COMPONENTS" \
      --route "$ROUTE" --product_kind "$kind" --train_betas "$BETAS" \
      --n_samples "$NSAMPLES" --weight_sets "$WEIGHT_SETS" \
      --out_dir "${OUT_ROOT}/singles-${kind}"
  done
fi

# ------------------------------- tempgain ------------------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "tempgain" ]] && need_components; then
  hr; echo "probe_tempgain -- what T(h) and g(h) actually learned"; hr
  runp tempgain "$A/probe_tempgain.py" \
    --guide_ckpts "$CKPTS" --guide_labels "$LABELS" \
    --eval_rewards "$EVAL_COMPONENTS" --train_betas "$BETAS" \
    --weights "$WEIGHTS" \
    --route "$ROUTE" --n_traj "$NTRAJ" \
    --out_dir "${OUT_ROOT}/tempgain"
fi

# ------------------------------- rollout -------------------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "rollout" ]] && need_components; then
  hr; echo "diag_rollout -- reward spread, residual headroom, train-vs-eval settings"; hr
  runp rollout "$A/diag_rollout.py" \
    --guide_ckpts "$CKPTS" --guide_labels "$LABELS" \
    --eval_rewards "$EVAL_COMPONENTS" --train_betas "$BETAS" \
    --route "$ROUTE" --n_samples "$NSAMPLES" \
    --train_temp "$TRAIN_TEMP" --train_eps "$TRAIN_EPS" \
    --out_dir "${OUT_ROOT}/rollout"
fi

# --------------------------------- logs --------------------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "logs" ]]; then
  hr; echo "diag_training_logs -- training curves from W&B"; hr
  if [[ -z "$WANDB_ENTITY" ]]; then
    echo "[skip] set WANDB_ENTITY=<your entity> to pull training curves"
  else
    names=""
    for c in $COMPONENTS; do
      names="${names:+$names,}${RUN_PREFIX}${c}${RUN_SUFFIX}"
    done
    runp logs "$A/diag_training_logs.py" \
      --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" \
      --runs "$names" --out_dir "${OUT_ROOT}/training-logs"
  fi
fi

# -------------------------- single-checkpoint probes -------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "single" ]]; then
  hr; echo "single-checkpoint probes"; hr
  SINGLE_CKPT="${SINGLE_CKPT:-$(resolve_ckpt "sweep-osim-hidden-db-replay_off-b10" || true)}"
  if [[ -z "$SINGLE_CKPT" || ! -f "$SINGLE_CKPT" ]]; then
    echo "[skip] set SINGLE_CKPT=/path/to/last.ckpt for the single-checkpoint probes"
  else
    echo "    using $SINGLE_CKPT"
    runp hidden_guide "$A/ablate_hidden_guide.py" \
      --ckpt "$SINGLE_CKPT" --guide_source ema \
      --out_dir "${OUT_ROOT}/hidden-guide"
    runp atoms_vs_coords "$A/diag_atoms_vs_coords.py" \
      --ckpt "$SINGLE_CKPT" --guide_source ema \
      --n 300 --coord_rerolls 5 --diff_steps 18 \
      --out_dir "${OUT_ROOT}/atoms-vs-coords"
  fi
fi

hr
say "ablations written to $OUT_ROOT"
hr

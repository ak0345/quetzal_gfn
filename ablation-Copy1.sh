#!/usr/bin/env bash
#
# run_ablations.sh -- all guide ablations in one place.
#
# The checkpoint lists are BUILT from a loop rather than pasted, so there is
# exactly one quoted string per call and no way to split a --guide_ckpts string
# across a paste boundary (the "EOF at random points" bug). Straight ASCII quotes
# only.
#
# Usage:
#   bash run_ablations.sh            # run everything
#   bash run_ablations.sh guide      # only ablate_guide.py runs
#   bash run_ablations.sh flip       # only ablate_logit_flip.py runs
#   bash run_ablations.sh ceiling    # only ablate_ceiling.py (#2 flip ceiling + #5 dead-axis)
#   bash run_ablations.sh singles    # only ablate_singles_weights.py (#3 fixed singles + #4 weight skew)
#   bash run_ablations.sh diag_rollout  # WHY not learned: reward flatness/residual/temp (mechs 2,3,4)
#   WANDB_ENTITY=you bash run_ablations.sh diag_logs  # WHY not learned: training curves (mechs 1,5)
#   DRY=1 bash run_ablations.sh      # print the commands, don't execute
#
set -uo pipefail

CKPT_ROOT="logs/quetzal-gfn"
PY="${PYTHON:-python}"
LABELS="c0,c1,c2,c3"
NSAMPLES="${NSAMPLES:-1000}"
SCALES="${SCALES:-0.5,1,2,4,8}"
NTRAJ="${NTRAJ:-400}"
NSCORE="${NSCORE:-1500}"
# Per-component eval rewards for the dead-axis test (#5). The benchmark key here
# is 'osimertinib'; change if you point this at a different benchmark's guides.
BENCH="${BENCH:-osimertinib}"
EVAL_COMPONENTS="gcomp:${BENCH}:0=c0,gcomp:${BENCH}:1=c1,gcomp:${BENCH}:2=c2,gcomp:${BENCH}:3=c3"
# Weight vectors for the #4 skew sweep (semicolon-separated).
WEIGHT_SETS="${WEIGHT_SETS:-0.25,0.25,0.25,0.25;1,0,0,0;0,0,0,1;0.1,0.1,0.1,0.7;0.4,0,0,0.6}"
# diag_rollout: the sampling settings the guides were TRAINED with (mechanism 4).
TRAIN_TEMP="${TRAIN_TEMP:-2.0}"
TRAIN_EPS="${TRAIN_EPS:-0.2}"
# diag_training_logs (wandb): entity/project and how run names are formed.
WANDB_ENTITY="${WANDB_ENTITY:-mathis-weil-university-college-london-ucl-}"
WANDB_PROJECT="${WANDB_PROJECT:-quetzal-gfn}"
DRY="${DRY:-0}"

# Build the comma-joined 4-component ckpt list for a given run suffix,
# e.g. suffix "beta50" -> .../gfn-geom-osim-comp{0..3}--beta50/checkpoints/last.ckpt
ckpts_for () {
  local suf="$1" ck=""
  for c in 0 1 2 3; do
    ck+="${CKPT_ROOT}/gfn-geom-osim-comp${c}--${suf}/checkpoints/last.ckpt,"
  done
  printf '%s' "${ck%,}"      # strip trailing comma
}

# ------------------------------------------------------------------ ablate_guide
# args: suffix  betas  route  product_kind  out_dir
run_guide () {
  local suf="$1" betas="$2" route="$3" kind="$4" out="$5"
  local ck; ck="$(ckpts_for "$suf")"
  echo ">>> [guide] suf=$suf route=$route kind=$kind -> $out"
  if [[ "$DRY" == "1" ]]; then
    printf '    %s ablate_guide.py --route %s --product_kind %s --out_dir %s\n' \
           "$PY" "$route" "$kind" "$out"
    return 0
  fi
  "$PY" ablate_guide.py \
    --guide_ckpts "$ck" \
    --guide_labels "$LABELS" \
    --route "$route" --operator product --product_kind "$kind" \
    --weights "0.25,0.25,0.25,0.25" --train_betas "$betas" \
    --n_samples "$NSAMPLES" --residual_scales "$SCALES" --do_singles \
    --out_dir "$out"
}

# ------------------------------------------------------------- ablate_logit_flip
# args: suffix  route  flip_temp  out_dir
run_flip () {
  local suf="$1" route="$2" temp="$3" out="$4"
  local ck; ck="$(ckpts_for "$suf")"
  echo ">>> [flip]  suf=$suf route=$route temp=$temp -> $out"
  if [[ "$DRY" == "1" ]]; then
    printf '    %s ablate_logit_flip.py --route %s --flip_temp %s --out_dir %s\n' \
           "$PY" "$route" "$temp" "$out"
    return 0
  fi
  "$PY" ablate_logit_flip.py \
    --guide_ckpts "$ck" \
    --guide_labels "$LABELS" \
    --n_samples "$NSAMPLES" \
    --route "$route" --n_traj "$NTRAJ" --flip_temp "$temp" \
    --out_dir "$out"
}

# ---------------------------------------------------------------- ablate_ceiling
# args: suffix  route  betas  out_dir   (#2 flip ceiling + #5 dead-axis variance)
run_ceiling () {
  local suf="$1" route="$2" betas="$3" out="$4"
  local ck; ck="$(ckpts_for "$suf")"
  echo ">>> [ceiling] suf=$suf route=$route -> $out"
  if [[ "$DRY" == "1" ]]; then
    printf '    %s ablate_ceiling.py --route %s --out_dir %s\n' "$PY" "$route" "$out"
    return 0
  fi
  "$PY" ablate_ceiling.py \
    --guide_ckpts "$ck" \
    --guide_labels "$LABELS" \
    --eval_rewards "$EVAL_COMPONENTS" \
    --train_betas "$betas" \
    --route "$route" --n_traj "$NTRAJ" --n_score "$NSCORE" \
    --out_dir "$out"
}

# -------------------------------------------------------- ablate_singles_weights
# args: suffix  route  kind  betas  out_dir   (#3 fixed singles + #4 weight skew)
run_singles () {
  local suf="$1" route="$2" kind="$3" betas="$4" out="$5"
  local ck; ck="$(ckpts_for "$suf")"
  echo ">>> [singles] suf=$suf route=$route kind=$kind -> $out"
  if [[ "$DRY" == "1" ]]; then
    printf '    %s ablate_singles_weights.py --route %s --product_kind %s --out_dir %s\n' \
           "$PY" "$route" "$kind" "$out"
    return 0
  fi
  "$PY" ablate_singles_weights.py \
    --guide_ckpts "$ck" \
    --guide_labels "$LABELS" \
    --eval_rewards "$EVAL_COMPONENTS" \
    --route "$route" --product_kind "$kind" --train_betas "$betas" \
    --n_samples "$NSAMPLES" --weight_sets "$WEIGHT_SETS" \
    --out_dir "$out"
}

# ------------------------------------------------------ diag_rollout (mechs 2,3,4)
# args: suffix  route  out_dir
run_diag_rollout () {
  local suf="$1" route="$2" out="$3"
  local ck; ck="$(ckpts_for "$suf")"
  echo ">>> [diag_rollout] suf=$suf route=$route -> $out"
  if [[ "$DRY" == "1" ]]; then
    printf '    %s diag_rollout.py --route %s --out_dir %s\n' "$PY" "$route" "$out"
    return 0
  fi
  "$PY" diag_rollout.py \
    --guide_ckpts "$ck" \
    --guide_labels "$LABELS" \
    --eval_rewards "$EVAL_COMPONENTS" \
    --route "$route" --n_samples "$NSAMPLES" \
    --train_temp "$TRAIN_TEMP" --train_eps "$TRAIN_EPS" \
    --out_dir "$out"
}

WHICH="${1:-all}"

if [[ "$WHICH" == "all" || "$WHICH" == "diag_logs" ]]; then
  echo "==================== diag_training_logs (mechanisms 1,5; wandb) ===================="
  if [[ -z "$WANDB_ENTITY" ]]; then
    echo "  [skip] set WANDB_ENTITY=... to pull training curves"
  else
    # one call per beta/objective group; run names are gfn-geom-osim-comp{0..3}--SUFFIX
    for suf in beta50 db-beta50; do
      names=""
      for c in 0 1 2 3; do names+="gfn-geom-osim-comp${c}--${suf},"; done
      names="${names%,}"
      out="${CKPT_ROOT}/training-diag/${suf}"
      echo ">>> [diag_logs] $suf -> $out"
      if [[ "$DRY" == "1" ]]; then
        printf '    %s diag_training_logs.py --runs "%s" --out_dir %s\n' "$PY" "$names" "$out"
      else
        "$PY" diag_training_logs.py \
          --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" \
          --runs "$names" --out_dir "$out"
      fi
    done
  fi
fi

echo "=============================================================="
echo " done."
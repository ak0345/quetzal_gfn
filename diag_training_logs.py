"""
diag_training_logs.py -- WHY did each guide fail to learn to steer?
Reads the TRAINING curves from wandb (no GPU) and partitions the components into
fix categories, addressing:

  MECHANISM 1: terminal loss never converged / no valid terminals.
    In DB the guide learns reward ONLY through the terminal residual. If
    db/terminal_loss plateaued high, or db/frac_valid_terminal ~ 0 (no valid
    molecules to learn from), the flow head never learned the reward direction
    -> residual is noise -> KL~0 at eval. We report terminal-loss trajectory,
    its final vs initial value, and frac_valid_terminal over training.

  MECHANISM 5: flow head vs policy fighting (DB).
    The same residual must satisfy interior flow-matching AND the terminal reward
    condition. If interior dominates, the reward term gets no say -> guide matches
    flow consistency but not reward. We report the interior:terminal loss ratio
    over training.

  Plus supporting signals when present: train/reward_valid_frac, train/log_reward_*
  (did the sampled reward ever climb?), guide weight growth proxy if logged.

Verdict per run:
  - "terminal never converged"  -> retrain longer / higher lr on flow head.
  - "no valid terminals"        -> reward floor too high, or axis unreachable.
  - "interior dominates"        -> lower db_interior_weight.
  - "reward never climbed"      -> dead axis or exploration problem (see rollout probe).
  - "converged & reward climbed"-> training was fine; weak steering is the
                                    saturated-prior ceiling, not a training bug.

Usage:
    python diag_training_logs.py \
        --entity YOUR_ENTITY --project quetzal-gfn \
        --runs "gfn-geom-osim-comp0--db-beta50,gfn-geom-osim-comp1--db-beta50,gfn-geom-osim-comp2--db-beta50,gfn-geom-osim-comp3--db-beta50" \
        --out_dir logs/quetzal-gfn/osim-compose-db/training-diag

    # or match by a name prefix instead of listing runs:
    python diag_training_logs.py --entity YOUR_ENTITY --project quetzal-gfn \
        --name_contains "osim" "db-beta50"
"""
import os
import json
import argparse

import numpy as np


# metrics we try to pull; missing ones are skipped gracefully
KEYS = [
    "train/loss",
    "db/terminal_loss", "db/interior_loss", "db/frac_valid_terminal",
    "db/terminal_res_abs_mean", "db/mean_logF_x",
    "train/reward_valid_frac", "train/valid_frac",
    "train/log_reward_mean", "train/log_reward_max",
]


def _tail_mean(x, frac=0.2):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return float("nan")
    k = max(1, int(len(x) * frac))
    return float(np.mean(x[-k:]))


def _head_mean(x, frac=0.1):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return float("nan")
    k = max(1, int(len(x) * frac))
    return float(np.mean(x[:k]))


def analyze_run(history):
    """history: dict metric_name -> list of values. Returns verdict + stats."""
    def col(k):
        return history.get(k, [])

    term = col("db/terminal_loss")
    inter = col("db/interior_loss")
    fvt = col("db/frac_valid_terminal")
    rvf = col("train/reward_valid_frac")
    lrm = col("train/log_reward_mean")

    stats = {}
    # M1: terminal loss trajectory
    if term:
        t0, t1 = _head_mean(term), _tail_mean(term)
        stats["terminal_loss_head"] = t0
        stats["terminal_loss_tail"] = t1
        stats["terminal_loss_drop_frac"] = float((t0 - t1) / t0) if t0 not in (0, float("nan")) else float("nan")
    # M1: valid terminals
    if fvt:
        stats["frac_valid_terminal_tail"] = _tail_mean(fvt)
        stats["frac_valid_terminal_min"] = float(np.min([v for v in fvt if v is not None]))
    # M5: interior vs terminal magnitude
    if term and inter:
        it = _tail_mean(inter); tt = _tail_mean(term)
        stats["interior_tail"] = it
        stats["terminal_tail"] = tt
        stats["interior_over_terminal"] = float(it / tt) if tt > 1e-9 else float("inf")
    # did reward climb?
    if lrm:
        stats["log_reward_mean_head"] = _head_mean(lrm)
        stats["log_reward_mean_tail"] = _tail_mean(lrm)
        stats["log_reward_climb"] = stats["log_reward_mean_tail"] - stats["log_reward_mean_head"]
    if rvf:
        stats["reward_valid_frac_tail"] = _tail_mean(rvf)

    # ---- verdict logic ----
    verdicts = []
    fvt_tail = stats.get("frac_valid_terminal_tail")
    if fvt_tail is not None and fvt_tail < 0.05:
        verdicts.append("NO VALID TERMINALS (reward floor too high / axis unreachable)")
    drop = stats.get("terminal_loss_drop_frac")
    if drop is not None and np.isfinite(drop) and drop < 0.3:
        verdicts.append("TERMINAL NEVER CONVERGED (loss barely dropped)")
    iot = stats.get("interior_over_terminal")
    if iot is not None and np.isfinite(iot) and iot > 10:
        verdicts.append("INTERIOR DOMINATES (terminal/reward term starved -> lower db_interior_weight)")
    climb = stats.get("log_reward_climb")
    if climb is not None and np.isfinite(climb) and climb < 0.2:
        verdicts.append("REWARD NEVER CLIMBED (dead axis or exploration problem)")
    if not verdicts:
        verdicts.append("TRAINING OK -> weak steering is likely the saturated-prior CEILING, not a training bug")
    stats["verdict"] = "; ".join(verdicts)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=os.getenv("WANDB_ENTITY"))
    ap.add_argument("--project", default="quetzal-gfn")
    ap.add_argument("--runs", default="",
                    help="comma-separated run NAMES (exact). If empty, use --name_contains")
    ap.add_argument("--name_contains", nargs="*", default=None,
                    help="substrings ALL of which a run name must contain to be included")
    ap.add_argument("--samples", type=int, default=2000,
                    help="wandb history sampling resolution")
    ap.add_argument("--out_dir", default="logs/quetzal-gfn/training-diag")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        import wandb
    except ImportError:
        raise SystemExit("pip install wandb")
    api = wandb.Api()

    # resolve the run set
    path = f"{args.entity}/{args.project}"
    if args.runs.strip():
        want = [r.strip() for r in args.runs.split(",") if r.strip()]
        runs = []
        for r in api.runs(path):
            if r.name in want:
                runs.append(r)
    else:
        subs = args.name_contains or []
        runs = [r for r in api.runs(path)
                if all(s in r.name for s in subs)]
    if not runs:
        raise SystemExit(f"no runs matched under {path}. "
                         f"runs={args.runs!r} name_contains={args.name_contains!r}")

    print(f"[wandb] {len(runs)} run(s): {[r.name for r in runs]}")

    report = {}
    for r in runs:
        # pull only the keys we care about that exist in this run
        avail = [k for k in KEYS if k in r.history(samples=1).columns] if hasattr(r, "history") else KEYS
        try:
            hist = r.history(keys=avail, samples=args.samples, pandas=True)
            history = {k: hist[k].tolist() for k in hist.columns if k in KEYS}
        except Exception:
            # fallback: scan_history
            history = {k: [] for k in KEYS}
            for row in r.scan_history(keys=KEYS):
                for k in KEYS:
                    if k in row:
                        history[k].append(row[k])
        stats = analyze_run(history)
        report[r.name] = stats
        print(f"\n=== {r.name} ===")
        for k in ["terminal_loss_head", "terminal_loss_tail", "terminal_loss_drop_frac",
                  "frac_valid_terminal_tail", "interior_over_terminal",
                  "log_reward_climb", "reward_valid_frac_tail"]:
            if k in stats:
                print(f"   {k:28} = {stats[k]:.4f}")
        print(f"   VERDICT: {stats['verdict']}")

    with open(os.path.join(args.out_dir, "training_diag.json"), "w") as f:
        json.dump(report, f, indent=2)

    # summary table
    print("\n" + "=" * 70)
    print(f"{'run':<40} {'term_drop':>10} {'valid_term':>11} {'rwd_climb':>10}")
    for name, s in report.items():
        print(f"{name[-38:]:<40} "
              f"{s.get('terminal_loss_drop_frac', float('nan')):>10.3f} "
              f"{s.get('frac_valid_terminal_tail', float('nan')):>11.3f} "
              f"{s.get('log_reward_climb', float('nan')):>10.3f}")
    print(f"\n[done] -> {os.path.join(args.out_dir, 'training_diag.json')}")


if __name__ == "__main__":
    main()
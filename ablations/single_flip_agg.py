#!/usr/bin/env python3
"""
aggregate_single_flips.py -- collect single_flip_ablation.py reports into one CSV
(plus a pooled per-position table and an optional combined plot).

Reads every flip_report*.json written by single_flip_ablation.py, whether they
sit flat in one directory (the run_all_ckpts.sh layout, flips/flip_report_<run>.json)
or one-per-subdirectory (flips/<run>/flip_report.json). Pulls the causal-chain
metrics at each temperature, parses the run name into axes, and writes a table
sorted so you can scan the ceiling signature across the whole sweep.

The columns that matter for the ceiling argument:
  flip_rate_high_gap  -- can the guide flip HARD (prior-dominant) decisions?
                         The ceiling predicts ~0 here regardless of guide/reward.
  flip_rate_low_gap   -- flips on already-close decisions (where the guide CAN act)
  sample_flip_rate    -- overall decision-change rate
  mean_prior_top1_gap -- how saturated the prior is (bigger = harder ceiling)

POOLING. Group summaries are computed from the RAW counts each report carries,
by summing numerators and denominators and dividing once at the end. Averaging
the per-run rates instead would weight a run of 400 states the same as one of
40,000, and at deep sequence positions (where only a few long molecules survive)
that difference is large. Positions no molecule ever reached stay empty rather
than counting as zero-flip.

Usage:
    python aggregate_single_flips.py --flips_root flips
    python aggregate_single_flips.py --flips_root flips --plot \
        --guides hidden,base --temp 1.0 --min_states 200

        
"""
import os
import re
import csv
import json
import glob
import argparse
from collections import defaultdict

# sweep-fexo-base-db-replay_on-b10  /  rtb-osim-hidden-db-replay_off-b1
NAME_RE_SWEEP = re.compile(
    r"^(?:sweep|rtb)-(?P<reward>[^-]+)-(?P<guide>[^-]+)-(?P<objective>[^-]+)"
    r"-replay_(?P<replay>on|off)-b(?P<beta>\d+)$")
NAME_RE_STAB = re.compile(r"^stability-geom-(?P<guide>[^-]+)-db-b(?P<beta>\d+)$")


def parse_name(name):
    m = NAME_RE_SWEEP.match(name)
    if m:
        d = m.groupdict(); d["beta"] = int(d["beta"])
        d["family"] = "rtb" if name.startswith("rtb") else "sweep"
        return d
    m = NAME_RE_STAB.match(name)
    if m:
        return {"reward": "atom_stability", "guide": m.group("guide"),
                "objective": "db", "replay": "off", "beta": int(m.group("beta")),
                "family": "stability"}
    return {"reward": "?", "guide": "?", "objective": "?", "replay": "?",
            "beta": -1, "family": "?"}


METRICS = ["delivered_frac", "argmax_flip_rate", "sample_flip_rate",
           "mean_total_variation", "mean_KL", "mean_prior_top1_gap",
           "flip_rate_high_gap", "flip_rate_low_gap", "frac_states_high_gap",
           "n_states", "n_positions_reported", "deepest_position_reached"]


def find_reports(root):
    """Both layouts: flat flip_report_<tag>.json and per-run <run>/flip_report.json."""
    paths = set(glob.glob(os.path.join(root, "flip_report*.json")))
    paths |= set(glob.glob(os.path.join(root, "*", "flip_report*.json")))
    return sorted(paths)


def run_name(path, rep):
    """Prefer the label the probe recorded; fall back to the path."""
    lab = rep.get("label")
    if lab:
        return lab
    base = os.path.basename(path)
    if base.startswith("flip_report_"):
        return base[len("flip_report_"):-len(".json")]
    return os.path.basename(os.path.dirname(path))


def _add_positions(dst, flips, states):
    """Accumulate per-position counts, padding to the longest run seen."""
    n = max(len(flips), len(states))
    if len(dst["flips"]) < n:
        dst["flips"].extend([0] * (n - len(dst["flips"])))
        dst["states"].extend([0] * (n - len(dst["states"])))
    for i in range(n):
        dst["flips"][i] += int(flips[i]) if i < len(flips) else 0
        dst["states"][i] += int(states[i]) if i < len(states) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flips_root", default="flips")
    ap.add_argument("--out", default=None,
                    help="row-per-(run,temp) CSV (default: <flips_root>/_flip_table.csv)")
    ap.add_argument("--pos_out", default=None,
                    help="pooled per-position CSV (default: "
                         "<flips_root>/_flip_by_position.csv)")
    ap.add_argument("--group_by", default="guide",
                    help="comma-separated axes to pool the position curves over "
                         "(default guide; e.g. 'reward,guide' or 'name')")
    ap.add_argument("--temp", default=None,
                    help="only this temperature (default: all; the ceiling "
                         "summary always uses 1.0 if present)")
    ap.add_argument("--guides", default=None, help="comma-separated filter")
    ap.add_argument("--rewards", default=None, help="comma-separated filter")
    ap.add_argument("--min_states", type=int, default=50,
                    help="hide positions pooled from fewer than this many states "
                         "when plotting (the CSV always keeps them). Hidden "
                         "positions break the line rather than being bridged. "
                         "Use --min_states 1 --ci to keep everything and show "
                         "uncertainty instead")
    ap.add_argument("--ci", action="store_true",
                    help="shade a 95%% Wilson interval per position. Preferred "
                         "over a high --min_states: thin positions stay visible "
                         "but declare their own uncertainty instead of being "
                         "silently deleted")
    ap.add_argument("--support_panel", action="store_true",
                    help="add a lower panel showing states per position (log "
                         "scale), so the reader can see the denominator collapse")
    ap.add_argument("--plot", action="store_true",
                    help="write a combined flip-rate-by-position figure")
    args = ap.parse_args()

    out = args.out or os.path.join(args.flips_root, "_flip_table.csv")
    pos_out = args.pos_out or os.path.join(args.flips_root, "_flip_by_position.csv")
    keep_guides = set(args.guides.split(",")) if args.guides else None
    keep_rewards = set(args.rewards.split(",")) if args.rewards else None

    rows = []
    pos_acc = defaultdict(lambda: {"flips": [], "states": []})
    group_axes = [a.strip() for a in args.group_by.split(",") if a.strip()]

    for path in find_reports(args.flips_root):
        try:
            rep = json.load(open(path))
        except Exception as e:
            print(f"[skip] {path}: {e}"); continue

        name = run_name(path, rep)
        base = {"name": name, "guide_type": rep.get("guide_type"),
                "reward_scored": rep.get("reward"),
                "guide_source": rep.get("guide_source"),
                "n_traj": rep.get("n_traj"), "ckpt": rep.get("ckpt")}
        base.update(parse_name(name))
        # the guide TYPE from the checkpoint is authoritative; the name is a hint
        if base.get("guide") == "?" and rep.get("guide_type"):
            base["guide"] = rep["guide_type"]
        if keep_guides and base.get("guide") not in keep_guides:
            continue
        if keep_rewards and str(base.get("reward")) not in keep_rewards:
            continue

        # one row per temperature present in the report
        for key in rep:
            if not key.startswith("flip_temp"):
                continue
            temp = key.replace("flip_temp", "")
            if args.temp is not None and temp != args.temp:
                continue
            r = rep[key]
            if not isinstance(r, dict):
                continue
            row = dict(base); row["temp"] = temp
            for m in METRICS:
                row[m] = r.get(m)
            rows.append(row)

            raw = r.get("raw") or {}
            flips = raw.get("flip_by_position")
            states = raw.get("state_by_position")
            if flips is None or states is None:
                # older reports (schema 1) carry only the rates, which cannot be
                # pooled correctly without their denominators
                continue
            gkey = tuple(str(row.get(a)) for a in group_axes) + (temp,)
            _add_positions(pos_acc[gkey], flips, states)

    if not rows:
        print(f"[error] no flip_report*.json under {args.flips_root}"); return

    front = ["name", "family", "reward", "guide", "guide_type", "objective",
             "replay", "beta", "temp", "guide_source", "n_traj"]
    cols = front + METRICS + ["ckpt"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # sort so the ceiling signature is easy to scan: by reward, then guide, then temp
    rows.sort(key=lambda r: (str(r.get("reward")), str(r.get("guide")),
                             str(r.get("beta")), str(r.get("temp"))))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {out} ({len(rows)} rows across "
          f"{len({r['name'] for r in rows})} checkpoints)")

    # ---- pooled per-position table (raw counts summed, then divided) ----
    if pos_acc:
        os.makedirs(os.path.dirname(pos_out) or ".", exist_ok=True)
        with open(pos_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(group_axes + ["temp", "position", "flips", "states",
                                     "flip_rate"])
            for gkey, d in sorted(pos_acc.items()):
                for i, (fl, st) in enumerate(zip(d["flips"], d["states"])):
                    # no molecule ever reached this position: leave the rate
                    # empty rather than writing a 0.0 that reads as "no flips"
                    rate = (fl / st) if st > 0 else ""
                    w.writerow(list(gkey) + [i, fl, st, rate])
        print(f"[write] {pos_out} ({len(pos_acc)} group(s))")

    # ---- ceiling summary, pooled over raw counts ----
    print("\n=== ceiling check: pooled flip rate on HIGH-gap vs LOW-gap decisions ===")
    summary_temp = args.temp or "1.0"
    agg = defaultdict(lambda: {"hi_f": 0, "hi_n": 0, "lo_f": 0, "lo_n": 0, "runs": 0})
    missing_raw = 0
    for path in find_reports(args.flips_root):
        try:
            rep = json.load(open(path))
        except Exception:
            continue
        name = run_name(path, rep)
        ax = parse_name(name)
        if keep_guides and ax.get("guide") not in keep_guides:
            continue
        if keep_rewards and str(ax.get("reward")) not in keep_rewards:
            continue
        r = rep.get(f"flip_temp{summary_temp}")
        if not isinstance(r, dict):
            continue
        raw = r.get("raw")
        if not raw:
            missing_raw += 1
            continue
        a = agg[ax.get("reward")]
        a["hi_f"] += raw.get("gap_hi_flipped", 0); a["hi_n"] += raw.get("gap_hi_states", 0)
        a["lo_f"] += raw.get("gap_lo_flipped", 0); a["lo_n"] += raw.get("gap_lo_states", 0)
        a["runs"] += 1
    for reward, d in sorted(agg.items(), key=lambda kv: str(kv[0])):
        hi = d["hi_f"] / d["hi_n"] if d["hi_n"] else float("nan")
        lo = d["lo_f"] / d["lo_n"] if d["lo_n"] else float("nan")
        print(f"  {str(reward):16s} high-gap flip={hi:.4f} (n={d['hi_n']:,})  "
              f"low-gap flip={lo:.4f} (n={d['lo_n']:,})  [{d['runs']} runs]")
    if missing_raw:
        print(f"  [note] {missing_raw} report(s) lack raw counts and were skipped "
              f"here; re-run them to pool correctly")
    print("  (ceiling => high-gap flip ~0 everywhere; guide only moves low-gap)")

    # ---- optional combined figure ----
    if args.plot and pos_acc:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import math

            def wilson(f, n, z=1.96):
                """Wilson score interval: sane at small n and at rates near 0,
                unlike the normal approximation, which gives zero width when
                zero flips are observed."""
                if n <= 0:
                    return (float("nan"), float("nan"))
                p = f / n
                d = 1 + z * z / n
                c = (p + z * z / (2 * n)) / d
                h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
                return (max(c - h, 0.0), min(c + h, 1.0))

            if args.support_panel:
                fig, (ax, axs) = plt.subplots(
                    2, 1, figsize=(9, 6), sharex=True,
                    gridspec_kw={"height_ratios": [3, 1]})
            else:
                fig, ax = plt.subplots(figsize=(9, 4.5)); axs = None

            for gkey, d in sorted(pos_acc.items()):
                label = "/".join(gkey)
                # keep the x-axis contiguous and insert NaN where support is too
                # thin: matplotlib breaks the line at NaN, so a gap reads as a
                # gap instead of being bridged by a straight segment that no
                # data supports
                xs = list(range(len(d["states"])))
                ys, lo, hi = [], [], []
                for f_, n_ in zip(d["flips"], d["states"]):
                    if n_ >= args.min_states and n_ > 0:
                        ys.append(f_ / n_)
                        a, b = wilson(f_, n_)
                        lo.append(a); hi.append(b)
                    else:
                        ys.append(float("nan"))
                        lo.append(float("nan")); hi.append(float("nan"))
                if all(y != y for y in ys):      # all NaN
                    continue
                n_pts = sum(1 for y in ys if y == y)
                line, = ax.plot(xs, ys, "o-" if n_pts <= 16 else "-", label=label)
                if args.ci:
                    ax.fill_between(xs, lo, hi, alpha=0.15,
                                    color=line.get_color(), linewidth=0)
                if axs is not None:
                    axs.plot(xs, [n if n > 0 else float("nan") for n in d["states"]],
                             "-", color=line.get_color())

            ax.set_ylabel("pooled sample-flip rate")
            title = "Guide influence by position (pooled"
            title += f"; <{args.min_states} states hidden" if args.min_states > 1 else ""
            title += "; 95% Wilson CI)" if args.ci else ")"
            ax.set_title(title)
            if axs is not None:
                axs.set_yscale("log"); axs.set_ylabel("states")
                axs.set_xlabel("sequence position")
                axs.axhline(args.min_states, ls=":", lw=0.8, color="grey")
            else:
                ax.set_xlabel("sequence position")
            if ax.has_data():
                ax.legend(fontsize=8)
                p = os.path.splitext(pos_out)[0] + ".png"
                fig.tight_layout(); fig.savefig(p, dpi=130)
                print(f"[plot] {p}")
            else:
                print(f"[plot] nothing above --min_states {args.min_states}")
            plt.close(fig)
        except Exception as e:
            print(f"[plot] failed: {e}")


if __name__ == "__main__":
    main()
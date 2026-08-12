#!/usr/bin/env python3
"""
aggregate_flips.py -- collect flip_ablation.py reports into one CSV.

Reads every flips/<name>/flip_report.json, pulls the causal-chain metrics at
each temperature, parses the model name into axes, and writes a table sorted so
you can scan the ceiling signature across the whole sweep.

The columns that matter for the ceiling argument:
  flip_rate_high_gap  -- can the guide flip HARD (prior-dominant) decisions?
                         The ceiling predicts ~0 here regardless of guide/reward.
  flip_rate_low_gap   -- flips on already-close decisions (where the guide CAN act)
  sample_flip_rate    -- overall decision-change rate
  mean_prior_top1_gap -- how saturated the prior is (bigger = harder ceiling)
"""
import os
import re
import csv
import json
import glob
import argparse

NAME_RE_SWEEP = re.compile(
    r"^sweep-(?P<reward>[^-]+)-(?P<guide>[^-]+)-(?P<objective>[^-]+)-replay_(?P<replay>on|off)-b(?P<beta>\d+)$")
NAME_RE_STAB = re.compile(r"^stability-geom-(?P<guide>[^-]+)-db-b(?P<beta>\d+)$")


def parse_name(name):
    m = NAME_RE_SWEEP.match(name)
    if m:
        d = m.groupdict(); d["beta"] = int(d["beta"]); d["family"] = "sweep"; return d
    m = NAME_RE_STAB.match(name)
    if m:
        return {"reward": "atom_stability", "guide": m.group("guide"),
                "objective": "db", "replay": "off", "beta": int(m.group("beta")),
                "family": "stability"}
    return {"reward": "?", "guide": "?", "objective": "?", "replay": "?",
            "beta": -1, "family": "?"}


METRICS = ["delivered_frac", "argmax_flip_rate", "sample_flip_rate",
           "mean_total_variation", "mean_KL", "mean_prior_top1_gap",
           "flip_rate_high_gap", "flip_rate_low_gap", "frac_states_high_gap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flips_root", default="flips")
    ap.add_argument("--out", default="flips/_flip_table.csv")
    args = ap.parse_args()

    rows = []
    for path in glob.glob(os.path.join(args.flips_root, "*", "flip_report.json")):
        try:
            rep = json.load(open(path))
        except Exception as e:
            print(f"[skip] {path}: {e}"); continue
        name = os.path.basename(os.path.dirname(path))
        base = {"name": name, "guide_type": rep.get("guide_type"),
                "reward_scored": rep.get("reward")}
        base.update(parse_name(name))
        # one row per temperature present in the report
        for key in rep:
            if not key.startswith("flip_temp"):
                continue
            temp = key.replace("flip_temp", "")
            r = rep[key]
            row = dict(base); row["temp"] = temp
            for m in METRICS:
                row[m] = r.get(m)
            rows.append(row)

    if not rows:
        print(f"[error] no flip_report.json under {args.flips_root}"); return

    front = ["name", "family", "reward", "guide", "guide_type", "objective",
             "replay", "beta", "temp"]
    cols = front + METRICS
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # sort so the ceiling signature is easy to scan: by reward, then guide, then temp
    rows.sort(key=lambda r: (str(r.get("reward")), str(r.get("guide")),
                             str(r.get("beta")), str(r.get("temp"))))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {args.out} ({len(rows)} rows across "
          f"{len({r['name'] for r in rows})} checkpoints)")

    # quick ceiling summary to stdout
    print("\n=== ceiling check: mean flip_rate on HIGH-gap vs LOW-gap decisions ===")
    from collections import defaultdict
    agg = defaultdict(lambda: {"hi": [], "lo": []})
    for r in rows:
        if r.get("temp") != "1.0":
            continue
        k = r.get("reward")
        if isinstance(r.get("flip_rate_high_gap"), (int, float)):
            agg[k]["hi"].append(r["flip_rate_high_gap"])
        if isinstance(r.get("flip_rate_low_gap"), (int, float)):
            agg[k]["lo"].append(r["flip_rate_low_gap"])
    for reward, d in sorted(agg.items()):
        hi = sum(d["hi"]) / len(d["hi"]) if d["hi"] else float("nan")
        lo = sum(d["lo"]) / len(d["lo"]) if d["lo"] else float("nan")
        print(f"  {reward:16s} high-gap flip={hi:.4f}  low-gap flip={lo:.4f}")
    print("  (ceiling => high-gap flip ~0 everywhere; guide only moves low-gap)")


if __name__ == "__main__":
    main()
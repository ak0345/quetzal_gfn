#!/usr/bin/env python3
"""
exclude_decoder_artifacts.py -- recompute the guide-sweep headline numbers with
the 3D-to-SMILES failures removed.

A molecule carrying a formal charge on a carbon atom is a bond-perception
failure, not chemistry: rdDetermineBonds infers bond orders from coordinates
alone, and when no assignment satisfies every valence it balances the books with
alternating formal charges. The result sanitises cleanly, so it passes every
downstream filter and enters the reported numbers. GEOM-Drugs contains none of
these; roughly 15% of everything this pipeline generates is one.

This script re-derives, per run, the quantities the sweep reports:

    top-1 / top-10 / top-100 mean reward, and the mean reward

over ALL sampled molecules and over the CLEAN subset alone, so the paper can say
what the artifact was worth. Reward is exp(log_reward), matching the convention
in `results/dumps/_aggregate/master_table.csv`.

OUTPUT
  results/dumps/_aggregate/artifact_exclusion.csv    one row per run
  a summary per benchmark family on stdout

USAGE
  python ablations/exclude_decoder_artifacts.py
  python ablations/exclude_decoder_artifacts.py --rewards osim,peri,fexo
  python ablations/exclude_decoder_artifacts.py --jobs 8
"""
import os
import csv
import glob
import math
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def charged_carbon(mol):
    return any(a.GetAtomicNum() == 6 and a.GetFormalCharge() != 0
               for a in mol.GetAtoms())


def topk_mean(rewards, k):
    if not len(rewards):
        return float("nan")
    return float(np.mean(sorted(rewards, reverse=True)[:k]))


def score_dump(path):
    """(run, seed, source) -> stats over all molecules and over clean ones."""
    out = []
    by_source = collections.defaultdict(list)
    try:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                try:
                    lr = float(row["log_reward"])
                except (TypeError, ValueError, KeyError):
                    continue
                by_source[row.get("source", "guided")].append(
                    (row.get("smiles", ""), lr))
    except OSError:
        return out

    seed_dir = os.path.basename(os.path.dirname(path))
    run = os.path.basename(os.path.dirname(os.path.dirname(path)))

    for source, rows in by_source.items():
        rew_all, rew_clean, n_bad = [], [], 0
        for smi, lr in rows:
            m = Chem.MolFromSmiles(smi) if smi else None
            if m is None:
                continue
            r = math.exp(lr)
            rew_all.append(r)
            if charged_carbon(m):
                n_bad += 1
            else:
                rew_clean.append(r)
        if not rew_all:
            continue
        rec = {"run": run, "seed": seed_dir, "source": source,
               "n": len(rew_all), "n_clean": len(rew_clean),
               "artifact_frac": n_bad / len(rew_all),
               "mean_all": float(np.mean(rew_all)),
               "mean_clean": float(np.mean(rew_clean)) if rew_clean else float("nan")}
        for k in (1, 10, 100):
            rec[f"top{k}_all"] = topk_mean(rew_all, k)
            rec[f"top{k}_clean"] = topk_mean(rew_clean, k)
        out.append(rec)
    return out


def family_of(run):
    parts = run.split("-")
    return parts[1] if len(parts) > 1 else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps_root", default=os.path.join(ROOT, "results", "dumps"))
    ap.add_argument("--rewards", default="osim,peri,fexo,zaleplon,nitrogen",
                    help="comma-separated benchmark families to include")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results", "dumps", "_aggregate", "artifact_exclusion.csv"))
    args = ap.parse_args()

    families = [f.strip() for f in args.rewards.split(",") if f.strip()]
    paths = []
    for fam in families:
        paths += sorted(glob.glob(os.path.join(
            args.dumps_root, f"sweep-{fam}-*", "seed*", "per_molecule.csv")))
    if not paths:
        raise SystemExit(f"no per_molecule.csv under {args.dumps_root}")
    print(f"[artifact] scoring {len(paths)} dump(s) on {args.jobs} process(es)")

    records = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, res in enumerate(ex.map(score_dump, paths, chunksize=1), 1):
            records += res
            if i % 25 == 0:
                print(f"[artifact]   {i}/{len(paths)}")

    if not records:
        raise SystemExit("nothing scored")

    fields = list(records[0])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    print(f"[artifact] wrote {args.out}  ({len(records)} rows)")

    # ------------------------------------------------------------- summary
    guided = [r for r in records if r["source"] == "guided"]
    base = [r for r in records if r["source"] == "base"]

    print(f"\n{'family':>9} {'runs':>5} {'artifact%':>10} "
          f"{'top10 all':>10} {'top10 clean':>12} {'delta':>8} "
          f"{'spread all':>11} {'spread clean':>13}")
    for fam in families:
        rs = [r for r in guided if family_of(r["run"]) == fam]
        if not rs:
            continue
        a = np.array([r["top10_all"] for r in rs], dtype=float)
        c = np.array([r["top10_clean"] for r in rs], dtype=float)
        af = np.mean([r["artifact_frac"] for r in rs])
        ok = np.isfinite(a) & np.isfinite(c)
        print(f"{fam:>9} {len(rs):>5} {100*af:>9.1f}% "
              f"{np.nanmean(a):>10.4f} {np.nanmean(c):>12.4f} "
              f"{np.nanmean(c[ok]-a[ok]):>+8.4f} "
              f"{np.nanmax(a)-np.nanmin(a):>11.4f} "
              f"{np.nanmax(c)-np.nanmin(c):>13.4f}")

    if base:
        print(f"\n{'':>9} {'frozen prior (base dumps)':>34}")
        for fam in families:
            rs = [r for r in base if family_of(r["run"]) == fam]
            if not rs:
                continue
            a = np.nanmean([r["top10_all"] for r in rs])
            c = np.nanmean([r["top10_clean"] for r in rs])
            af = np.mean([r["artifact_frac"] for r in rs])
            print(f"{fam:>9} {len(rs):>5} {100*af:>9.1f}% "
                  f"{a:>10.4f} {c:>12.4f} {c-a:>+8.4f}")

    allg = np.array([r["artifact_frac"] for r in guided], dtype=float)
    print(f"\n[artifact] guided artifact rate: mean {100*allg.mean():.1f}%, "
          f"range {100*allg.min():.1f}-{100*allg.max():.1f}% over {len(guided)} runs")
    da = np.array([r["top10_clean"] - r["top10_all"] for r in guided], dtype=float)
    da = da[np.isfinite(da)]
    print(f"[artifact] top-10 mean shifts by {da.mean():+.4f} on average "
          f"(range {da.min():+.4f} to {da.max():+.4f}) when artifacts are dropped")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
synth_compare.py -- measure synthesizability across generators to test the
hypothesis: a GFlowNet guiding a synthesizable prior (Quetzal) stays synthesizable
while from-scratch reward optimizers (Graph GA, Genetic-GFlowNet, ...) drift off
the synthesizable manifold.

Scores every method's SMILES by up to THREE complementary metrics:
  * SA score   (Ertl)   -- heuristic, always available (ships with RDKit contrib).
  * SCScore             -- learned from reaction corpora (optional).
  * RAscore    (RAscore) -- retrosynthesis-model based; closest to "can CASP route it"
                           (optional, heavier).
Using several guards against any single metric being gamed; agreement => robust.

Representation note: synthesizability is a 2D-graph property, so we score SMILES.
Quetzal's 3D outputs are assumed already converted to SMILES upstream. For any
method we still report the RDKit parse rate, so conversion/validity artifacts
can't silently bias the comparison.

INPUT: one SMILES file per method (one SMILES per line; blank/invalid tolerated).
OUTPUT:
  * <out_dir>/synth_per_molecule.csv   (method, smiles, sa, scscore, rascore)
  * <out_dir>/synth_summary.json       (per-method distributions + parse rates)
  * <out_dir>/synth_distributions.png  (violin/hist per metric per method)
  * console table

Then, to get the reward-vs-synth PARETO view, merge your per-molecule reward
scores onto synth_per_molecule.csv by SMILES and run --pareto (see bottom).

Usage:
  python synth_compare.py \
    --inputs "geom=results/geom_drugs_smiles.txt,\
quetzal_prior=dumps/osim_real/base_smiles.txt,\
quetzal_guided=dumps/osim_real/guided_smiles.txt"
    --metrics sa \
    --out_dir results/synth_compare
"""
import os
import csv
import json
import argparse

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


# ----------------------------- metric backends ----------------------------
def _load_sa():
    """SA score from RDKit contrib. Returns fn(mol)->float in [1,10] (1=easy)."""
    try:
        from rdkit.Chem import RDConfig
        import sys
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        return lambda m: float(sascorer.calculateScore(m))
    except Exception as e:
        print(f"[sa] unavailable: {e}")
        return None


def _load_scscore():
    """SCScore. pip install scscore, or clone connorcoley/scscore. Returns
    fn(smiles)->float in [1,5] (higher = harder)."""
    try:
        from scscore.standalone_model_numpy import SCScorer
        import scscore, glob
        # find a shipped weights file
        base = os.path.dirname(scscore.__file__)
        wts = glob.glob(os.path.join(base, "models", "*", "model.ckpt-*.as_numpy.json.gz"))
        model = SCScorer()
        model.restore(wts[0])
        def fn(smi):
            _, s = model.get_score_from_smi(smi)
            return float(s)
        return fn
    except Exception as e:
        print(f"[scscore] unavailable ({e}); skipping. "
              f"install: pip install scscore  (or clone connorcoley/scscore)")
        return None


def _load_rascore():
    """RAscore (retrosynthetic accessibility). pip install RAscore + a model.
    Returns fn(smiles)->float in [0,1] (higher = more accessible/synthesizable)."""
    try:
        from RAscore import RAscore_NN
        scorer = RAscore_NN.RAScorerNN()
        return lambda smi: float(scorer.predict(smi))
    except Exception as e:
        print(f"[rascore] unavailable ({e}); skipping. "
              f"install: pip install RAscore  (+ download a model)")
        return None


# ----------------------------- scoring ------------------------------------
def read_smiles(path, cap=None):
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip().split()[0] if line.strip() else ""
            if s:
                out.append(s)
            if cap and len(out) >= cap:
                break
    return out


def score_method(name, smis, backends):
    sa_fn, sc_fn, ra_fn = backends
    rows = []
    n_parsed = 0
    for smi in smis:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            rows.append({"method": name, "smiles": smi, "sa": None,
                         "scscore": None, "rascore": None, "parsed": 0})
            continue
        n_parsed += 1
        canon = Chem.MolToSmiles(m)
        rec = {"method": name, "smiles": canon, "parsed": 1,
               "sa": None, "scscore": None, "rascore": None}
        if sa_fn:
            try: rec["sa"] = sa_fn(m)
            except Exception: pass
        if sc_fn:
            try: rec["scscore"] = sc_fn(canon)
            except Exception: pass
        if ra_fn:
            try: rec["rascore"] = ra_fn(canon)
            except Exception: pass
        rows.append(rec)
    parse_rate = n_parsed / max(len(smis), 1)
    return rows, parse_rate


def dist(vals):
    v = np.array([x for x in vals if x is not None], float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std()),
            "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True,
                    help="comma-separated name=path pairs, one per method")
    ap.add_argument("--metrics", nargs="+", default=["sa"],
                    choices=["sa", "scscore", "rascore"])
    ap.add_argument("--cap", type=int, default=None,
                    help="max molecules per method (for quick runs)")
    ap.add_argument("--out_dir", default="results/synth_compare")
    # Binary "synthesizable rate" thresholds -- for comparability with headline
    # claims like S3-GFN's ">=95% synthesizable" (which is a FRACTION above a
    # threshold, not a mean SA). RAscore >= ra_thresh, and/or SA <= sa_thresh.
    ap.add_argument("--ra_thresh", type=float, default=0.5,
                    help="RAscore >= this counts as synthesizable (S3-GFN-style rate)")
    ap.add_argument("--sa_thresh", type=float, default=4.5,
                    help="SA <= this counts as 'easy' (secondary binary rate)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sa_fn = _load_sa() if "sa" in args.metrics else None
    sc_fn = _load_scscore() if "scscore" in args.metrics else None
    ra_fn = _load_rascore() if "rascore" in args.metrics else None
    backends = (sa_fn, sc_fn, ra_fn)
    active = [m for m, fn in zip(["sa", "scscore", "rascore"], backends) if fn]
    print(f"[metrics] active: {active or 'NONE -- check installs'}")

    methods = {}
    for pair in args.inputs.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, path = pair.split("=", 1)
        methods[name.strip()] = path.strip()

    all_rows = []
    summary = {"metrics": active, "methods": {}}
    for name, path in methods.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: file not found: {path}")
            continue
        smis = read_smiles(path, cap=args.cap)
        rows, parse_rate = score_method(name, smis, backends)
        all_rows.extend(rows)
        # binary synthesizable rates (fraction above/below threshold), among PARSED
        ra_vals = [r["rascore"] for r in rows if r["rascore"] is not None]
        sa_vals = [r["sa"] for r in rows if r["sa"] is not None]
        ra_rate = (float(np.mean(np.array(ra_vals) >= args.ra_thresh))
                   if ra_vals else None)
        sa_rate = (float(np.mean(np.array(sa_vals) <= args.sa_thresh))
                   if sa_vals else None)
        summary["methods"][name] = {
            "n_input": len(smis), "parse_rate": parse_rate,
            "sa": dist([r["sa"] for r in rows]),
            "scscore": dist([r["scscore"] for r in rows]),
            "rascore": dist([r["rascore"] for r in rows]),
            # the numbers directly comparable to headline "% synthesizable" claims
            "synth_rate_rascore_ge_%.2f" % args.ra_thresh: ra_rate,
            "synth_rate_sa_le_%.1f" % args.sa_thresh: sa_rate,
        }
        rr = f"{ra_rate*100:.1f}%" if ra_rate is not None else "n/a"
        sr = f"{sa_rate*100:.1f}%" if sa_rate is not None else "n/a"
        print(f"[{name}] n={len(smis)} parse_rate={parse_rate:.3f}  "
              f"synth(RAscore>={args.ra_thresh})={rr}  synth(SA<={args.sa_thresh})={sr}")

    # write per-molecule CSV (for merging reward later)
    csv_path = os.path.join(args.out_dir, "synth_per_molecule.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "smiles", "sa", "scscore", "rascore", "parsed"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    with open(os.path.join(args.out_dir, "synth_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # console table
    print("\n" + "=" * 78)
    hdr = f"{'method':<20} {'parse':>6}"
    for mt in active:
        hdr += f" {mt+'_mean':>12} {mt+'_med':>10}"
    print(hdr)
    for name, s in summary["methods"].items():
        row = f"{name:<20} {s['parse_rate']:>6.3f}"
        for mt in active:
            d = s[mt]
            row += f" {d.get('mean', float('nan')):>12.3f} {d.get('median', float('nan')):>10.3f}"
        print(row)
    print("\nLower SA / lower SCScore / HIGHER RAscore = more synthesizable.")
    print("Hypothesis holds if quetzal_guided ~ quetzal_prior << from-scratch baselines")
    print("(on SA/SCScore) and quetzal_guided has HIGHER RAscore than baselines.")

    # plots
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(summary["methods"].keys())
        fig, axes = plt.subplots(1, len(active), figsize=(5.5 * len(active), 4.6), squeeze=False)
        for ai, mt in enumerate(active):
            ax = axes[0][ai]
            data = []
            for nm in names:
                vals = [r[mt] for r in all_rows if r["method"] == nm and r[mt] is not None]
                data.append(np.array(vals, float))
            parts = ax.violinplot([d for d in data if len(d)], showmedians=True)
            ax.set_xticks(range(1, len(names) + 1))
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(mt)
            ax.set_title(f"{mt} by method")
        fig.tight_layout()
        p = os.path.join(args.out_dir, "synth_distributions.png")
        fig.savefig(p, dpi=130); plt.close(fig)
        print(f"[plot] {p}")
    except Exception as e:
        print(f"[plot] failed: {e}")

    print(f"\n[done] per-molecule CSV -> {csv_path}")
    print("To get the reward-vs-synth PARETO view: merge your reward scores onto")
    print("that CSV by 'smiles', then run:  python synth_compare.py pareto ...")


# ------------------------- pareto subcommand ------------------------------
def pareto():
    """Separate entry: python synth_compare.py --pareto_csv merged.csv ...
    merged.csv must have columns: method, smiles, reward, and one synth metric."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto_csv", required=True)
    ap.add_argument("--synth_col", default="sa")
    ap.add_argument("--reward_col", default="reward")
    ap.add_argument("--synth_better", choices=["low", "high"], default="low",
                    help="low for SA/SCScore, high for RAscore")
    ap.add_argument("--out", default="reward_vs_synth.png")
    args = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(args.pareto_csv)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for nm, g in df.groupby("method"):
        s = g[args.synth_col].astype(float)
        r = g[args.reward_col].astype(float)
        ax.scatter(s, r, s=12, alpha=0.4, label=nm)
    ax.set_xlabel(f"{args.synth_col}  ({'lower=better' if args.synth_better=='low' else 'higher=better'})")
    ax.set_ylabel(args.reward_col)
    ax.set_title("reward vs synthesizability (upper-left is the sweet spot for SA)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out, dpi=140)
    print(f"[pareto] {args.out}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "pareto":
        sys.argv.pop(1)
        pareto()
    else:
        main()
#!/usr/bin/env python3
"""
make_table_pmo.py -- our fine-tuned configurations against the PMO leaderboard.

Emits a LaTeX table comparing our best AUC top-10 at a 10,000-call oracle budget
against the ten best-performing methods in PMO (Gao et al., 2022, Table 2). The
budget and the metric match, which is the reason this comparison is meaningful
at all: PMO reports AUC of the top-10 mean over 10,000 oracle calls, and so do
our fine-tuned runs, which record generation order.

WHAT THE COMPARISON DOES AND DOES NOT SHOW. PMO's methods search over a
string, graph or synthesis action space with no 3D constraint. Ours samples 3D
geometries from a frozen prior and converts them to SMILES, so a fraction of
every batch is lost to bond perception before scoring. The numbers are on the
same axis but the methods are not doing the same task, and the point of putting
them side by side is the size of the gap rather than a ranking.

CAVEAT ON ZALEPLON. PMO states that its `zaleplon_mpo` and `sitagliptin_mpo`
differ from the GuacaMol implementations. Our Zaleplon is GuacaMol's
`zaleplon_with_other_formula`, so that column is not like-for-like and the table
marks it.

INPUTS
  results/oracle_gfn_mols/_results/*.json      (stage 8 harvest)

USAGE
  python figures/make_table_pmo.py --out ../paper/GFlownet_workshop/tab_pmo.tex
  python figures/make_table_pmo.py --plot out/fig_pmo.pdf
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs


def our_best(bench, metric="auc_top10"):
    """Best and median AUC top-10 over our fine-tuned runs on one benchmark."""
    try:
        h = fs.load_harvest(bench)
    except SystemExit:
        return None
    vals = []
    for name, v in h.items():
        if name.startswith("_") or "nitrogen" in name:
            continue
        s = (v.get("budgeted") or {}).get(metric)
        if s is not None:
            vals.append((float(s), name))
    if not vals:
        return None
    vals.sort()
    ref = (h.get("_reference") or {}).get("top10")
    return {"best": vals[-1], "median": vals[len(vals) // 2][0],
            "worst": vals[0], "n": len(vals), "dataset": ref}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="osim,peri,zaleplon")
    ap.add_argument("--metric", default="auc_top10",
                    choices=["auc_top10", "top10"])
    ap.add_argument("--out", default="out/tab_pmo.tex")
    ap.add_argument("--plot", default=None, help="also write a dot plot here")
    args = ap.parse_args()

    benches = [b for b in args.benches.split(",") if b]
    ours = {b: our_best(b, args.metric) for b in benches}
    missing = [b for b in benches if ours[b] is None]
    if missing:
        print(f"[table] no harvest yet for {missing}; those columns show --")

    # ------------------------------- LaTeX ---------------------------------
    head = " & ".join(fs.BENCH_TITLE[b].replace(" MPO", "") for b in benches)
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering\small")
    L.append(r"\caption{AUC top-10 at a 10,000-call oracle budget against the ten "
             r"best-performing methods of PMO \cite{Gao2022Sample}, Table 2 "
             r"(mean $\pm$ std over five runs). Our rows are the best and median "
             r"over the fine-tuned configurations. PMO reimplements "
             r"\texttt{zaleplon\_mpo}, so that column compares against a different "
             r"objective and is marked $\dagger$.}")
    L.append(r"\label{tab:pmo}")
    L.append(r"\begin{tabular}{l" + "r" * len(benches) + "}")
    L.append(r"\toprule")
    L.append(f"Method & {head} \\\\")
    L.append(r"\midrule")
    for m, row in fs.PMO_AUC_TOP10.items():
        cells = []
        for b in benches:
            mu, sd = row.get(b, (None, None))
            mark = r"^{\dagger}" if b in fs.PMO_NOT_COMPARABLE else ""
            cells.append(f"${mu:.3f} \\pm {sd:.3f}{mark}$" if mu is not None else "--")
        L.append(f"{m} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    for label, key in (("GEOM best-of-10k", "dataset"),
                       ("Ours, best", "best"), ("Ours, median", "median")):
        cells = []
        for b in benches:
            o = ours[b]
            if o is None or o.get(key) is None:
                cells.append("--"); continue
            v = o[key][0] if isinstance(o[key], tuple) else o[key]
            cells.append(f"${v:.3f}$")
        L.append(f"{label} & " + " & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    text = "\n".join(L) + "\n"

    import os
    d = os.path.dirname(os.path.abspath(args.out))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"wrote {args.out}")

    for b in benches:
        o = ours[b]
        pmo = [v[b][0] for v in fs.PMO_AUC_TOP10.values() if b in v]
        if o is None:
            print(f"[{b}] no harvest yet | PMO best {max(pmo):.3f}")
            continue
        gap = max(pmo) - o["best"][0]
        print(f"[{b}] ours best {o['best'][0]:.3f} ({o['best'][1]}), "
              f"median {o['median']:.3f} over {o['n']} runs | "
              f"PMO best {max(pmo):.3f}, worst {min(pmo):.3f} | gap {gap:+.3f}")

    # -------------------------------- plot ---------------------------------
    if args.plot:
        fs.use_paper_style()
        fig, axes = plt.subplots(1, len(benches), figsize=(2.4 * len(benches), 3.4),
                                 squeeze=False)
        for ax, b in zip(axes[0], benches):
            names = list(fs.PMO_AUC_TOP10)
            y = np.arange(len(names))
            mu = [fs.PMO_AUC_TOP10[m][b][0] for m in names]
            sd = [fs.PMO_AUC_TOP10[m][b][1] for m in names]
            ax.errorbar(mu, y, xerr=sd, fmt="o", ms=4, color="0.45",
                        elinewidth=0.8, capsize=1.5, zorder=3)
            o = ours[b]
            if o is not None:
                ax.axvline(o["best"][0], color=fs.FAMILY_COLOURS["FT: proj"],
                           lw=1.6, zorder=2)
                ax.text(o["best"][0], -0.8, " ours (best)", fontsize=6.5,
                        color=fs.FAMILY_COLOURS["FT: proj"], va="bottom")
                if o.get("dataset") is not None:
                    ax.axvline(o["dataset"], color=fs.REF_COLOUR, ls="--", lw=1.3,
                               zorder=2)
            ax.set_yticks(y)
            ax.set_yticklabels(names if b == benches[0] else [""] * len(names),
                               fontsize=6.5)
            ax.invert_yaxis()
            title = fs.BENCH_TITLE[b]
            if b in fs.PMO_NOT_COMPARABLE:
                title += " (PMO differs)"
            ax.set_title(title, fontsize=8.5)
            ax.set_xlabel("AUC top-10")
        fs.save(fig, args.plot)


if __name__ == "__main__":
    main()

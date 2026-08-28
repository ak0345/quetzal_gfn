#!/usr/bin/env python3
"""
make_fig14_quality_vs_score.py -- Figure 14: compound quality against benchmark
score.

Quality-filter pass rate over the top-100 molecules against the top-10 benchmark
score, one point per configuration, with the GEOM-Drugs reference marked. The
point is that configurations scoring higher do not do so by producing molecules
that fail the filters.

INPUTS
  results/oracle_gfn_mols/_results/*.json    extended.quality, budgeted  (stage 8)

USAGE
  python figures/make_fig14_quality_vs_score.py --out out/fig14.pdf
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="auto",
                    choices=list(fs.ALL_BENCHES) + ["auto", "both", "all"])
    ap.add_argument("--metric", default="top10", choices=["top10", "auc_top10"])
    fs.add_arg_common(ap, "out/fig14_quality_vs_score.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    benches = fs.resolve_benches(args.bench)
    fig, axes = plt.subplots(1, len(benches), figsize=(args.width, 3.2),
                             squeeze=False)
    axes = axes[0]

    for ax, bench in zip(axes, benches):
        h = fs.load_harvest(bench)
        seen = set()
        for name, v in sorted(h.items()):
            if name.startswith("_") or "nitrogen" in name:
                continue
            b = v.get("budgeted") or {}
            q = ((v.get("extended") or {}).get("quality") or {}).get("pass_rate")
            s = b.get(args.metric)
            if q is None or s is None:
                continue
            fam = fs.ft_family(name)
            ax.scatter(s, q, s=48, zorder=3, alpha=0.9,
                       color=fs.FAMILY_COLOURS.get(fam, "0.5"),
                       label=fam if fam not in seen else None)
            seen.add(fam)
        ref = h.get("_reference") or {}
        rq, rs = ref.get("quality_pass_rate"), ref.get("top10")
        if rq is not None and rs is not None:
            ax.scatter([rs], [rq], marker="*", s=150, color=fs.REF_COLOUR, zorder=4)
            ax.annotate("GEOM-Drugs", (rs, rq), fontsize=6.5, color=fs.REF_COLOUR,
                        xytext=(5, -8), textcoords="offset points")
        ax.set_xlabel({"top10": "top-10 mean",
                       "auc_top10": "AUC top-10"}[args.metric])
        ax.set_ylabel("quality-filter pass rate (top-100)")
        ax.set_ylim(0, 1.05)
        ax.set_title(fs.BENCH_TITLE[bench])
        ax.legend(frameon=False, loc="lower left", fontsize=6.5)
    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_fig13_sample_efficiency.py -- Figure 13: top-10 mean against oracle calls.

Best-so-far top-10 score as a function of the number of molecules scored, for
each fine-tuned configuration, with the GEOM best-of-10k baseline marked. This
curve is the quantity the AUC metric integrates: AUC_k(B) = B^-1 * integral of
f_k(n) dn, and since f_k is non-decreasing, AUC_k(B) <= f_k(B). The gap between
the curve and its endpoint is how quickly a method reached its best molecules.

INPUTS
  results/oracle_gfn_mols/_results/*.json     extended.curve_top10   (stage 8)

USAGE
  python figures/make_fig13_sample_efficiency.py --out out/fig13.pdf
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="auto",
                    choices=list(fs.ALL_BENCHES) + ["auto", "both", "all"])
    fs.add_arg_common(ap, "out/fig13_sample_efficiency.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    benches = fs.resolve_benches(args.bench)
    fig, axes = plt.subplots(1, len(benches), figsize=(args.width, 3.2),
                             squeeze=False)
    axes = axes[0]

    for ax, bench in zip(axes, benches):
        h = fs.load_harvest(bench)
        n = 0
        for name, v in sorted(h.items()):
            if name.startswith("_") or "nitrogen" in name:
                continue
            c = ((v.get("extended") or {}).get("curve_top10") or {})
            if not c.get("calls"):
                continue
            fam = fs.ft_family(name)
            ax.plot(c["calls"], c["topk"], lw=1.1, alpha=0.85,
                    color=fs.FAMILY_COLOURS.get(fam, "0.5"))
            n += 1
        ref = (h.get("_reference") or {}).get("top10")
        if ref is not None:
            ax.axhline(ref, color=fs.REF_COLOUR, ls="--", lw=1.3)
            ax.text(0, ref, " GEOM best-of-10k", fontsize=6.5,
                    color=fs.REF_COLOUR, va="bottom", ha="left")
        ax.set_xlabel("oracle calls")
        ax.set_ylabel("best-so-far top-10 mean")
        ax.set_title(fs.BENCH_TITLE[bench])
        print(f"[{bench}] {n} configurations plotted")

    fams = ["FT: proj", "FT: atom", "FT: full", "FT: LoRA"]
    handles = [plt.Line2D([], [], color=fs.FAMILY_COLOURS[f], lw=1.4) for f in fams]
    fig.legend(handles, fams, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()

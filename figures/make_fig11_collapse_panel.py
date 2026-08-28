#!/usr/bin/env python3
"""
make_fig11_collapse_panel.py -- Figure 11: distribution statistics over the
course of training.

From left: 3D-to-SMILES conversion rate; share of the most frequent molecule
within a window; effective number of distinct molecules, computed as the
exponential of the Shannon entropy over the window; and mean heavy-atom count.
All are computed over successive windows of the recorded generation stream.

The figure exists to rule out an alternative reading of the flat scores: that
fine-tuning collapsed onto a small set of molecules. Conversion holds near 0.95
and the concentration measures are flat, so none of the runs collapses.

The unique-count fraction is deliberately not shown: it saturates once a window
exceeds the number of distinct molecules and is insensitive to concentration.

INPUTS
  results/oracle_gfn_mols/_results/*.json     extended.buckets   (stage 8)

The `n_atoms` panel needs a bucket field that older harvests predate. If it is
absent the panel is dropped with a note; re-run stage 8 to record it.

USAGE
  python figures/make_fig11_collapse_panel.py --bench osim --out out/fig11.pdf
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

PANELS = [("validity", "3D->SMILES rate", None),
          ("top_share", "share of most common mol", None),
          ("eff_distinct", "effective distinct mols", "log"),
          ("n_atoms", "mean heavy atoms", None)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="auto",
                    choices=list(fs.ALL_BENCHES) + ["auto"])
    fs.add_arg_common(ap, "out/fig11_collapse_panel.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    bench = fs.default_bench() if args.bench == "auto" else args.bench
    h = fs.load_harvest(bench)
    runs = {n: v for n, v in sorted(h.items())
            if not n.startswith("_") and "nitrogen" not in n
            and ((v.get("extended") or {}).get("buckets") or {}).get("calls")}
    if not runs:
        fs.die("no bucketed generation stream in the harvest",
               how="bash scripts/08_analysis.sh harvest   # needs --extended")

    have = [p for p in PANELS
            if any(p[0] in (v["extended"]["buckets"] or {}) for v in runs.values())]
    missing = [p[0] for p in PANELS if p not in have]
    if missing:
        print(f"[fig] not in these harvests, panel(s) dropped: {missing}. "
              f"Re-run stage 8 to record them.")

    fig, axes = plt.subplots(1, len(have), figsize=(args.width * 1.35, 3.0),
                             squeeze=False)
    axes = axes[0]
    for (key, lab, scale), ax in zip(have, axes):
        for name, v in runs.items():
            b = v["extended"]["buckets"]
            if key not in b:
                continue
            ax.plot(b["calls"], b[key], lw=1.1, alpha=0.85,
                    color=fs.FAMILY_COLOURS.get(fs.ft_family(name), "0.5"))
        ax.set_xlabel("oracle calls")
        ax.set_ylabel(lab)
        if scale:
            ax.set_yscale(scale)

    fams = ["FT: proj", "FT: atom", "FT: full", "FT: LoRA"]
    handles = [plt.Line2D([], [], color=fs.FAMILY_COLOURS[f], lw=1.4) for f in fams]
    fig.legend(handles, fams, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    print(f"[{bench}] {len(runs)} runs, {len(have)} panels")
    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_fig01_landscape.py -- Figure 1: configurations cluster at the prior's
dataset score.

One point per configuration on a single score axis, coloured by family, against
the GEOM best-of-10k baseline (dashed) and the published GuacaMol baselines
(dotted). The figure's claim is that the spread across three guide
architectures, four objectives and four fine-tuning scopes is smaller than the
distance separating all of them from published methods.

THE AXIS IS MIXED, AND DELIBERATELY SO. Guides and the dataset are post-hoc
i.i.d. samples with no oracle-call order, so they are reported as final top-10.
Fine-tuned runs record generation order and are reported as AUC top-10 over
10,000 calls. Since top-k is non-decreasing in n, AUC top-10 <= top-10, so the
axis slightly favours the guides -- as does their smaller sample (N=5,000
against N=10,000). Both are printed to stdout so the gap is auditable.

INPUTS
  results/dumps/_aggregate/master_table.csv       guides   (stage 4)
  results/oracle_gfn_mols/_results/*.json         fine-tuned + reference (stage 8)

USAGE
  python figures/make_fig01_landscape.py --out out/fig01_landscape.pdf
  python figures/make_fig01_landscape.py --bench osim
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs


def guide_points(rows, bench, exclude=()):
    """Guide-sweep configurations for one benchmark, as final top-10."""
    pts = []
    for r in rows:
        name = r.get("name", "")
        if not name.startswith("sweep-") or r.get("reward") != bench:
            continue
        fam = fs.guide_family(name)
        if fam is None or any(x in name for x in exclude):
            continue
        v = fs.f(r.get("guided_reward_top10_mean"))
        if np.isnan(v):
            continue
        pts.append({"name": name, "family": fam, "score": v,
                    "err": fs.f(r.get("guided_reward_top10_std"), 0.0)})
    # The composed sampler, where present, is its own family. Composed rows
    # record the full benchmark key ("osimertinib") rather than the short tag
    # the sweep uses ("osim"), so they need their own match or they leak into
    # the other benchmark's panel.
    long_key = {"osim": "osimertinib", "peri": "perindopril",
                "fexo": "fexofenadine"}.get(bench, bench)
    for r in rows:
        name = r.get("name", "")
        if not name.startswith("compose-") or r.get("reward") != long_key:
            continue
        v = fs.f(r.get("guided_reward_top10_mean"))
        if not np.isnan(v):
            pts.append({"name": name, "family": "composed", "score": v,
                        "err": fs.f(r.get("guided_reward_top10_std"), 0.0)})
    return pts


def prior_point(rows, bench):
    """The frozen prior with no intervention, from the shared base dump."""
    for r in rows:
        if r.get("name") == f"base_quetzal-{bench}":
            v = fs.f(r.get("base_reward_top10_mean"))
            if np.isnan(v):
                v = fs.f(r.get("guided_reward_top10_mean"))
            if not np.isnan(v):
                return {"name": "frozen prior", "family": "frozen prior",
                        "score": v, "err": fs.f(r.get("base_reward_top10_std"), 0.0)}
    return None


def ft_points(harvest, metric="auc_top10"):
    pts = []
    for name, v in harvest.items():
        if name.startswith("_") or "nitrogen" in name:
            continue
        s = (v.get("budgeted") or {}).get(metric)
        if s is None:
            continue
        pts.append({"name": name, "family": fs.ft_family(name),
                    "score": float(s), "err": 0.0})
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="auto",
                    choices=list(fs.ALL_BENCHES) + ["auto", "both", "all"])
    ap.add_argument("--ft_metric", default="auc_top10",
                    choices=["auc_top10", "top10"])
    ap.add_argument("--exclude", default="",
                    help="comma-separated substrings of run names to drop, "
                         "e.g. tempgain")
    ap.add_argument("--no_finetune", action="store_true",
                    help="drop the fine-tuned lanes, leaving the guides, the "
                         "composed sampler and the frozen prior")
    ap.add_argument("--no_composed", action="store_true",
                    help="drop the composed-sampler lane")
    fs.add_arg_common(ap, "out/fig01_landscape.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    exclude = tuple(x for x in args.exclude.split(",") if x)
    rows = fs.load_master_table()

    if args.bench in ("auto", "both", "all") and args.no_finetune:
        # Without the fine-tuned lanes this figure is drawn from the guide
        # sweep, so the benchmarks it can cover are the ones the sweep trained
        # against -- not the ones that happen to have a fine-tuning harvest.
        have = {r.get("reward") for r in rows if r.get("name", "").startswith("sweep-")}
        benches = [b for b in fs.ALL_BENCHES
                   if b in have and b not in exclude]
        if not benches:
            fs.die("no guide-sweep rows for any benchmark",
                   how="bash scripts/04_dump_guides.sh")
    else:
        benches = fs.resolve_benches(args.bench)

    # a stable row order, so both panels line up on the same family sequence
    ORDER = ["guide: base", "guide: hidden", "guide: tempgain", "composed",
             "FT: proj", "FT: atom", "FT: full", "FT: LoRA", "frozen prior"]

    # Panels share the lane axis, so only the leftmost carries tick labels;
    # repeating them once per panel is what made the earlier version unreadable.
    fig, axes = plt.subplots(1, len(benches),
                             figsize=(max(args.width, 3.1 * len(benches)), 3.6),
                             squeeze=False, sharey=True)
    axes = axes[0]

    for ax, bench in zip(axes, benches):
        # the harvest is the fine-tuning record and the source of the GEOM
        # reference line; without the fine-tuned lanes it is only needed for the
        # latter, and its absence is not fatal
        harvest = {}
        if not args.no_finetune or os.path.exists(fs.HARVEST.get(bench, "")):
            harvest = fs.load_harvest(bench)

        pts = guide_points(rows, bench, exclude)
        if args.no_composed:
            pts = [p for p in pts if p["family"] != "composed"]
        if not args.no_finetune:
            pts += ft_points(harvest, args.ft_metric)
        p0 = prior_point(rows, bench)
        if p0:
            pts.append(p0)
        if not pts:
            fs.die(f"no configurations found for {bench}")

        lanes = [fam for fam in ORDER if any(p["family"] == fam for p in pts)]
        ypos = {fam: i for i, fam in enumerate(lanes)}

        for p in pts:
            y = ypos[p["family"]]
            # jitter within the lane so coincident scores stay countable
            jitter = (hash(p["name"]) % 7 - 3) * 0.035
            ax.errorbar(p["score"], y + jitter,
                        xerr=(p["err"] if p["err"] else None),
                        fmt="o", ms=5, elinewidth=0.8, capsize=1.5,
                        color=fs.FAMILY_COLOURS.get(p["family"], "0.4"),
                        alpha=0.9, zorder=3)

        # Reference lines are labelled vertically along the line itself. Written
        # horizontally they collided with each other and with the panel edge
        # whenever two baselines fell close together.
        ref = ((harvest.get("_reference") or fs.load_reference(bench) or {})
               .get("top10"))
        if ref is not None:
            ax.axvline(ref, color=fs.REF_COLOUR, ls="--", lw=1.3, zorder=2)
            ax.text(ref, len(lanes) - 0.55, f" GEOM best-of-10k {ref:.3f}",
                    fontsize=6, color=fs.REF_COLOUR, va="bottom", ha="left",
                    rotation=90)
        # Published baselines often sit within a hair of each other (0.837 and
        # 0.839 on Osimertinib), so their labels are staggered down the lane
        # axis instead of all starting at the same height.
        published = fs.PUBLISHED.get(bench, {})
        for i, (lab, v) in enumerate(published.items()):
            ax.axvline(v, color="0.55", ls=":", lw=1.1, zorder=2)
            ax.text(v, len(lanes) - 0.55 - i * 1.15, f" {lab} {v:g}",
                    fontsize=6, color="0.4", va="bottom", ha="left",
                    rotation=90)
        if not fs.PUBLISHED.get(bench):
            print(f"[fig] {bench}: no published GuacaMol baseline in "
                  f"figstyle.PUBLISHED, so no dotted reference line is drawn. "
                  f"Add it there to show one.")

        ax.set_yticks(range(len(lanes)))
        ax.set_yticklabels(lanes)
        ax.set_ylim(-1.0, len(lanes) - 0.3)
        ax.invert_yaxis()
        ax.set_title(fs.BENCH_TITLE[bench])
        ax.grid(axis="y", alpha=0.12)
        ax.tick_params(axis="x", labelsize=7)

        scores = [p["score"] for p in pts]
        spread = max(scores) - min(scores)
        gap = (min(abs(min(fs.PUBLISHED[bench].values()) - max(scores)),
                   abs(max(fs.PUBLISHED[bench].values()) - max(scores)))
               if fs.PUBLISHED.get(bench) else float("nan"))
        print(f"[{bench}] {len(pts)} configurations | spread {spread:.3f} "
              f"| gap to nearest published {gap:.3f} | GEOM ref {ref}")

    # One shared x-label rather than one per panel, and it names only the
    # conventions actually drawn: with the fine-tuned lanes off, every point on
    # the axis is a final top-10 and the AUC half of the caption is misleading.
    xlab = ("top-10 mean" if args.no_finetune else
            "top-10  (guides, dataset)   /   AUC top-10  (fine-tuned, 10k calls)")
    fig.supxlabel(xlab, fontsize=8.5)
    fig.tight_layout()
    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()

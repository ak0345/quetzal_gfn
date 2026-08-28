#!/usr/bin/env python3
"""
make_fig03_capacity.py -- Figure 3: score declines as trainable capacity
increases.

Reads the fine-tune harvest JSONs and plots benchmark score against the number
of trainable parameters, with uniqueness and compound-quality pass rate on a
twin axis. The point of the figure is that the configurations with the most
parameters score lowest while shifting furthest from the prior's distribution.

PARAMETER COUNTS
  Derived from the architecture below, not hand-entered, so they stay
  consistent if you change d_model or the vocabulary. Override any single run
  with --params "name=1234,other=5678" if a count is wrong; the startup banner
  of rtb_finetune.py prints the true value for each run
  ("[scope] X.XXXM trainable / XX.XM total").

  proj    : d_model * vocab                       = 768 * 128
  LoRA(r) : r * (d_model + vocab)                 (A is r x d_in, B is d_out x r)
  atom    : proj + encode1 trunk
  full    : every parameter in the model

INPUTS
  results/oracle_gfn_mols/_results/*.json      (stage 8)

USAGE
  python figures/make_fig03_capacity.py --out out/fig03_capacity.pdf
  python figures/make_fig03_capacity.py --bench peri --metric top10
  python figures/make_fig03_capacity.py --params "rtb-full-osim-b10=84900000"
"""

import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle as fs

D_MODEL = 768
VOCAB = 128
N_TRUNK = 43_000_000     # encode1: embeddings + blocks1 (6 layers)
N_TOTAL = 85_000_000     # whole model, including encode2 and the denoiser

# the capacity axis splits LoRA by scope, which is finer than the family split
# the other figures use; both live in figstyle so the colours stay consistent
FAMILY_COLOURS = {k: fs.FAMILY_COLOURS[k] for k in
                  ("LoRA (proj)", "LoRA (atom)", "proj", "atom", "full")}


def infer_params(name):
    """Trainable-parameter count for a run name. Returns (count, family)."""
    proj = D_MODEL * VOCAB
    lora_rank = None
    for r in (64, 16, 4):                      # longest first: 'lora4' vs 'lora64'
        if f"lora{r}" in name:
            lora_rank = r
            break
    is_atom = "-atom-" in name
    if lora_rank is not None:
        n = lora_rank * (D_MODEL + VOCAB)
        if is_atom:
            return n + N_TRUNK, "LoRA (atom)"
        return n, "LoRA (proj)"
    if "-full-" in name:
        return N_TOTAL, "full"
    if is_atom:
        return proj + N_TRUNK, "atom"
    return proj, "proj"


def load(path, bench_sub, metric, exclude=("nitrogen",)):
    # exit 2, not 1, so make_all.sh reports this as a missing input rather than
    # a broken script and prints the command that produces it
    fs.need(path, how=fs.harvest_cmd(bench_sub))
    d = json.load(open(path))
    pts = []
    for k, v in d.items():
        if k.startswith("_") or bench_sub not in k:
            continue
        if any(x in k for x in exclude):
            continue
        b = v.get("budgeted", {})
        e = v.get("extended", {}) or {}
        div = e.get("diversity", {}) or {}
        s = b.get(metric) if metric != "composite" else b.get("guacamol_score")
        if s is None:
            continue
        n, fam = infer_params(k)
        pts.append({
            "name": k, "params": n, "family": fam, "score": float(s),
            "uniqueness": b.get("uniqueness_among_valid"),
            "quality": (e.get("quality") or {}).get("pass_rate"),
            "diversity": div.get("internal_diversity"),
            "calls": b.get("n_oracle_calls"),
        })
    return pts, d.get("_reference", {})


def main():
    ap = argparse.ArgumentParser()
    # osim.json / peri.json carry the `extended` block (quality, diversity) AND
    # the `_reference` baseline; the *_budget10000.json files carry neither, so
    # prefer the former unless you only need score.
    ap.add_argument("--osim_ft", default=fs.HARVEST["osim"])
    ap.add_argument("--peri_ft", default=fs.HARVEST["peri"])
    ap.add_argument("--bench", default="auto",
                    choices=list(fs.ALL_BENCHES) + ["auto", "both", "all"])
    ap.add_argument("--metric", default="auc_top10",
                    choices=["auc_top10", "top10", "composite"])
    ap.add_argument("--params", default=None,
                    help='override counts: "run=1234,other=5678"')
    ap.add_argument("--out", default="out/fig03_capacity.pdf")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--width", type=float, default=7.2)
    args = ap.parse_args()
    fs.use_paper_style()

    overrides = {}
    if args.params:
        for kv in args.params.split(","):
            k, _, v = kv.partition("=")
            overrides[k.strip()] = int(v)

    # The per-benchmark overrides stay for the two the paper names; every other
    # benchmark resolves through figstyle, so a pipeline run covering a
    # different objective still produces the panel.
    override_path = {"osim": args.osim_ft, "peri": args.peri_ft}
    specs = [(b, override_path.get(b) or fs.HARVEST[b], fs.BENCH_TITLE[b])
             for b in fs.resolve_benches(args.bench)]

    fig, axes = plt.subplots(1, len(specs), figsize=(args.width, 3.4),
                             squeeze=False)
    axes = axes[0]

    for ax, (sub, path, title) in zip(axes, specs):
        pts, ref = load(path, sub, args.metric)
        for p in pts:
            if p["name"] in overrides:
                p["params"] = overrides[p["name"]]
        pts.sort(key=lambda p: p["params"])
        if not pts:
            continue

        xs = np.array([p["params"] for p in pts], dtype=float)
        ys = np.array([p["score"] for p in pts], dtype=float)

        # trend line through log-capacity, to make the decline legible
        lx = np.log10(xs)
        if len(set(lx)) > 1:
            m, c = np.polyfit(lx, ys, 1)
            gx = np.linspace(lx.min(), lx.max(), 50)
            ax.plot(10 ** gx, m * gx + c, "-", color="0.6", lw=1.2, zorder=1)
            print(f"[{sub}] slope = {m:+.4f} score per decade of parameters")

        for p in pts:
            ax.scatter(p["params"], p["score"], s=55, zorder=3,
                       color=FAMILY_COLOURS.get(p["family"], "0.4"),
                       label=p["family"])

        refv = {"top10": ref.get("top10"), "auc_top10": ref.get("top10"),
                "composite": ref.get("guacamol_score")}.get(args.metric)
        if refv is not None:
            ax.axhline(refv, color=fs.REF_COLOUR, ls="--", lw=1.3, zorder=2)
            ax.text(xs.min(), refv, " GEOM best-of-10k", fontsize=6.5,
                    color=fs.REF_COLOUR, va="bottom", ha="left")
            if args.metric == "auc_top10":
                print(f"[{sub}] NOTE: reference is a final top-10 (a static "
                      f"dataset has no call order), so it upper-bounds the AUC "
                      f"line it is drawn against.")

        ax.set_xscale("log")
        ax.set_xlabel("trainable parameters", fontsize=8.5)
        ax.set_ylabel({"auc_top10": "AUC top-10",
                       "top10": "top-10 mean",
                       "composite": "GuacaMol composite"}[args.metric],
                      fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

        # distribution health on the twin axis
        ax2 = ax.twinx()
        for key, mark, col in (("uniqueness", "^", "0.45"),
                               ("quality", "v", "0.65")):
            v = [(p["params"], p[key]) for p in pts if p.get(key) is not None]
            if v:
                ax2.plot([a for a, _ in v], [b for _, b in v], mark, ls=":",
                         ms=4, lw=1, color=col, label=key)
        ax2.set_ylabel("uniqueness / quality", fontsize=8.5, color="0.45")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(labelsize=8, colors="0.45")

    handles, labels = [], []
    for fam, col in FAMILY_COLOURS.items():
        handles.append(plt.Line2D([], [], marker="o", ls="", color=col))
        labels.append(fam)
    for lab, mark in (("uniqueness", "^"), ("quality", "v")):
        handles.append(plt.Line2D([], [], marker=mark, ls=":", color="0.5"))
        labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, -0.12))

    fig.tight_layout()
    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()
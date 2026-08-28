#!/usr/bin/env python3
"""
make_fig10_excluded_flow.py -- Figure 10: the excluded flow-route runs.

The flow route computes the guide from the flow head rather than the policy
head. The claim it was excluded for is a DELIVERY failure: the residual is
computed but never applied, so the flip rate is identically zero while the
policy route on the same checkpoints moves decisions.

This script does not assume that outcome. It plots what the report contains and
prints the delivery fraction alongside it, so a run that does deliver is visible
as such rather than being drawn as if it had failed.

  Left   sampled-flip rate by sequence position, one line per guide, at each
         recorded flip temperature.
  Right  per-guide summary: the fraction of states at which the residual was
         actually delivered, against the sampled-flip rate it produced.

INPUTS
  flip report(s) from a --route flow run of ablate_logit_flip_compose.py. There
  is no default: the flow route is the excluded configuration and is not part of
  the standard pipeline, so the reports must be named explicitly.

USAGE
  python figures/make_fig10_excluded_flow.py \
      --report results/ablations/flow/flip-t1.0/flip_report.json \
               results/ablations/flow/flip-t0.3/flip_report.json \
      --out out/fig10.pdf
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

HOW = ("ROUTE=flow OUT_ROOT=results/ablations/flow RUNS=\"<runs>\" "
       "bash scripts/07_ablations.sh flip\n"
       "    # the flow route is the excluded configuration; --route policy is "
       "the one used for reported results")


def load_blocks(paths):
    """[(temperature, {guide: block})] for every report given.

    ablate_logit_flip_compose.py writes one directory per temperature, with the
    temperature in `config.flip_temp` and the per-guide results under `flip`.
    Older single-guide reports nested blocks as `flip_temp<T>` instead, so both
    shapes are accepted.
    """
    out = []
    for p in paths:
        d = fs.load_json(p, how=HOW)
        if "flip" in d:
            temp = str((d.get("config") or {}).get("flip_temp", "?"))
            out.append((temp, d["flip"]))
            continue
        for key, blk in d.items():
            if key.startswith("flip_temp") and isinstance(blk, dict):
                out.append((key.replace("flip_temp", ""), {"(pooled)": blk}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, nargs="+",
                    help="flip report(s) from --route flow runs, one per "
                         "temperature")
    ap.add_argument("--max_pos", type=int, default=8)
    fs.add_arg_common(ap, "out/fig10_excluded_flow.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    blocks = load_blocks(args.report)
    if not blocks:
        fs.die(f"no flip results in {args.report}", HOW)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(args.width * 1.15, 3.3))

    # ---- left: flip rate by position, per guide and temperature ----
    styles = {"1.0": "-", "0.3": "--"}
    line_short = dict(zip([n for _, pg in blocks for n in pg],
                          fs.distinguishing_labels(
                              [n for _, pg in blocks for n in pg])))
    for temp, per_guide in blocks:
        for name, blk in per_guide.items():
            pr = (blk.get("flip_rate_first8_positions")
                  or blk.get("flip_rate_by_position") or [])
            n = min(args.max_pos, len(pr))
            if not n:
                continue
            guide = next((g for g in ("hidden", "tempgain", "base")
                          if f"-{g}-" in name), None)
            axL.plot(np.arange(n), [np.nan if v is None else v for v in pr[:n]],
                     styles.get(temp, ":"), marker="o", ms=3.5, lw=1.2,
                     color=fs.GUIDE_COLOURS.get(guide, "0.5"),
                     label=f"{line_short[name]}  T={temp}")
    axL.set_xlabel("sequence position (atom decision)")
    axL.set_ylabel("sampled-flip rate")
    axL.set_ylim(bottom=0)
    axL.set_title("Flip rate by position")
    axL.legend(frameon=False, fontsize=5.5, loc="upper right")

    # ---- right: delivery against the flip rate it bought ----
    all_names = [n for _, pg in blocks for n in pg]
    short = dict(zip(all_names, fs.distinguishing_labels(all_names)))
    names, delivered, flips, temps = [], [], [], []
    for temp, per_guide in blocks:
        for name, blk in per_guide.items():
            names.append(f"{short[name]}\nT={temp}")
            delivered.append(blk.get("delivered_frac"))
            flips.append(blk.get("sample_flip_rate")
                         or blk.get("flip_rate") or 0.0)
            temps.append(temp)

    x = np.arange(len(names))
    axR.bar(x - 0.2, [d if d is not None else np.nan for d in delivered],
            width=0.4, color=fs.GUIDE_COLOURS["hidden"],
            label="delivered fraction")
    axR.bar(x + 0.2, flips, width=0.4, color=fs.GUIDE_COLOURS["tempgain"],
            label="sampled-flip rate")
    axR.set_xticks(x)
    axR.set_xticklabels(names, fontsize=5.5, rotation=30, ha="right")
    axR.set_ylim(0, 1.05)
    axR.set_ylabel("fraction")
    axR.set_title("Delivery vs. flips")
    axR.legend(frameon=False, fontsize=6)

    # ---- say plainly whether this is the delivery-failure signature ----
    for (temp, per_guide) in blocks:
        for name, blk in per_guide.items():
            d = blk.get("delivered_frac")
            f_ = blk.get("sample_flip_rate")
            pr = [v for v in (blk.get("flip_rate_first8_positions") or [])
                  if v is not None]
            allzero = bool(pr) and max(pr) == 0.0
            print(f"[fig] T={temp} {fs.clean_label(name):<42} "
                  f"delivered={d} sample_flip={f_} all-zero-by-position={allzero}")
    any_delivered = any(
        (blk.get("delivered_frac") or 0) > 0.5
        for _, pg in blocks for blk in pg.values())
    if any_delivered:
        print("[fig] NOTE: these reports show the residual BEING DELIVERED "
              "(delivered_frac > 0.5) with a non-zero flip rate. That is NOT "
              "the delivery-failure signature the figure caption describes -- "
              "either the flow-route defect is not present in these "
              "checkpoints, or it was fixed. Do not caption this as a "
              "zero-delivery run without re-checking.")

    fig.suptitle("Excluded flow-route runs", fontsize=9.5)
    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Collect every dump_summary.json under <dumps_root>/<name>/seed<k>/ into one
master table, with cross-model plots carrying seed error bars.

Run names are parsed into columns -- sweep-<reward>-<guide>-<objective>-replay_
<on|off>-b<beta>, stability-geom-<guide>-db-b<beta>, and the composed form --
so the table can be pivoted along any experimental axis. Each metric is
aggregated as mean and standard deviation across seeds.

Outputs (in --out_dir):
  master_table.csv        one row per (name), all metrics mean/std across seeds
  master_long.csv         one row per (name, seed), unaggregated
  steering_by_model.png   top10 delta (guided-base) per model, error bars = seed std
  stability_by_model.png  guided atom-stability per model (stability + all runs)
  fcd_vs_ref.png          guided/base FCD to GEOM per model
  fcd_guided_vs_base_by_model.png   FCD(guided,base) per model (reward-agnostic
                          steering distance; ~0 = no shift from frozen prior)
  fcd_guided_vs_base_by_reward.png  same, grouped by reward (easy vs hard)
  reward_mean_by_beta.png log_reward_mean vs beta, split by guide (steering axis)
"""
import os
import re
import csv
import json
import glob
import argparse
from collections import defaultdict

import numpy as np


NAME_RE_SWEEP = re.compile(
    r"^sweep-(?P<reward>[^-]+)-(?P<guide>[^-]+)-(?P<objective>[^-]+)-replay_(?P<replay>on|off)-b(?P<beta>\d+)$")
NAME_RE_STAB = re.compile(
    r"^stability-geom-(?P<guide>[^-]+)-db-b(?P<beta>\d+)$")
# compose-<benchkey>-<operator>-k<K>-b<beta>
NAME_RE_COMPOSE = re.compile(
    r"^compose-(?P<bench>[^-]+)-(?P<operator>[^-]+)-k(?P<k>\d+)-b(?P<beta>\d+)$")


def parse_name(name):
    m = NAME_RE_SWEEP.match(name)
    if m:
        d = m.groupdict()
        d["beta"] = int(d["beta"])
        d["family"] = "sweep"
        return d
    m = NAME_RE_STAB.match(name)
    if m:
        return {"reward": "atom_stability", "guide": m.group("guide"),
                "objective": "db", "replay": "off", "beta": int(m.group("beta")),
                "family": "stability"}
    m = NAME_RE_COMPOSE.match(name)
    if m:
        # guide axis carries the operator (linear/product/harmonic); reward is the
        # benchmark being composed toward; objective='compose', replay='off'.
        return {"reward": m.group("bench"), "guide": m.group("operator"),
                "objective": "compose", "replay": "off",
                "beta": int(m.group("beta")), "family": "compose",
                "n_components": int(m.group("k"))}
    return {"reward": "?", "guide": "?", "objective": "?", "replay": "?",
            "beta": -1, "family": "?"}


# flat metrics pulled from each summary (guided/base nested + top-level)
def _reward_stats_from_npy(dump_dir, source):
    """Compute correct actual-reward stats by loading the per-molecule log-reward
    array the dumper saved (guided_rewards.npy / base_rewards.npy /
    composed_rewards.npy) and exponentiating PER MOLECULE. Works on EXISTING
    dumps -- no re-dump needed. Returns {} if the file is absent."""
    import numpy as _np
    if not dump_dir:
        return {}
    candidates = (["composed_rewards.npy", "guided_rewards.npy"] if source == "guided"
                  else ["base_rewards.npy"])
    for fn in candidates:
        p = os.path.join(dump_dir, fn)
        if os.path.exists(p):
            try:
                lr = _np.asarray(_np.load(p), dtype=float)
            except Exception:
                continue
            if len(lr) == 0:
                return {}
            sr = _np.sort(_np.exp(lr))   # actual reward per molecule, sorted
            return {
                f"{source}_reward_mean": float(_np.exp(lr).mean()),
                f"{source}_reward_top1": float(sr[-1]),
                f"{source}_reward_top10": float(sr[-10:].mean()) if len(lr) >= 10 else None,
                f"{source}_reward_top100": float(sr[-100:].mean()) if len(lr) >= 100 else None,
            }
    return {}


def flatten(summary, dump_dir=None):
    out = {}
    # actual-reward stats: prefer summary (new dumps), else load per-molecule
    # arrays from the dump dir (works on EXISTING dumps, no re-dump).
    npy_stats = {}
    for src in ("guided", "base"):
        s = summary.get(src, {}) or {}
        if not any(isinstance(s.get(rk), (int, float))
                   for rk in ("reward_mean", "reward_top10")):
            npy_stats.update(_reward_stats_from_npy(dump_dir, src))
    for src in ("guided", "base"):
        s = summary.get(src, {}) or {}
        for k in ("parse_rate", "uniqueness", "log_reward_mean", "log_reward_top1",
                  "log_reward_top10", "log_reward_top100", "atom_stability",
                  "mol_stability", "n_valid_smiles"):
            out[f"{src}_{k}"] = s.get(k)
        # actual reward: summary value if present, else the npy-derived one
        for k in ("reward_mean", "reward_top1", "reward_top10", "reward_top100"):
            out[f"{src}_{k}"] = s.get(k) if isinstance(s.get(k), (int, float)) \
                else npy_stats.get(f"{src}_{k}")

    # ---- deltas: guided - base, on BOTH log-reward and actual-reward ----
    g = summary.get("guided", {}) or {}
    b = summary.get("base", {}) or {}
    def _d(key):
        gv, bv = g.get(key), b.get(key)
        return (gv - bv) if isinstance(gv, (int, float)) and isinstance(bv, (int, float)) else None
    def _dr(key):   # actual-reward delta using merged (summary-or-npy) values
        gv, bv = out.get(f"guided_{key}"), out.get(f"base_{key}")
        return (gv - bv) if isinstance(gv, (int, float)) and isinstance(bv, (int, float)) else None
    # log-reward deltas
    out["top10_delta"] = summary.get("top10_delta_guided_minus_base")
    if out["top10_delta"] is None:
        out["top10_delta"] = _d("log_reward_top10")
    out["top100_delta"] = _d("log_reward_top100")
    out["top1_delta"] = _d("log_reward_top1")
    out["mean_delta"] = _d("log_reward_mean")
    # actual-reward deltas (correct, from per-molecule stats)
    out["reward_mean_delta"] = _dr("reward_mean")
    out["reward_top1_delta"] = _dr("reward_top1")
    out["reward_top10_delta"] = _dr("reward_top10")
    out["reward_top100_delta"] = _dr("reward_top100")

    fcd = summary.get("fcd", {}) or {}
    for k in ("guided_vs_base", "guided_vs_ref", "base_vs_ref"):
        out[f"fcd_{k}"] = fcd.get(k)
    wass = summary.get("descriptor_wasserstein", {}) or {}
    for dk, dv in wass.items():
        if isinstance(dv, dict):
            out[f"wass_{dk}_guided_vs_ref"] = dv.get("guided_vs_ref")
            out[f"wass_{dk}_base_vs_ref"] = dv.get("base_vs_ref")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps_root", default="dumps")
    ap.add_argument("--out_dir", default="dumps/_aggregate")
    ap.add_argument("--extra_roots", default="",
                    help="comma-separated ADDITIONAL dump roots to scan (e.g. the "
                         "composed dumps at dumps_composed). Recursively globbed.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    roots = [args.dumps_root] + [r.strip() for r in args.extra_roots.split(",") if r.strip()]
    # recursive glob catches BOTH the 2-level single-guide layout
    # (<root>/<name>/seed<k>/) and the 3-level composed layout
    # (<root>/<bench>/<operator>/seed<k>/). name comes from the JSON, so depth
    # doesn't matter for identity.
    summary_paths = []
    for root in roots:
        summary_paths += glob.glob(os.path.join(root, "**", "dump_summary.json"),
                                   recursive=True)
    summary_paths = sorted(set(summary_paths))

    rows = []   # long: one per (name, seed)
    n_base_as_row = 0
    for path in summary_paths:
        parts = path.split(os.sep)
        # gflow_multi's own output dir (compose_native/) uses a different schema
        # (<tag>_summary.json, not dump_summary.json) -- skip if it sneaks in.
        if "compose_native" in parts:
            continue
        try:
            summary = json.load(open(path))
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue

        # ---- standalone base dumps (dumps/_base/<family>/seed<k>/): emit as
        # their OWN rows (base Quetzal is a first-class model in the table) ----
        if "_base" in parts:
            # path is .../_base/<family>/seed<k>/dump_summary.json
            try:
                fam = parts[parts.index("_base") + 1]
            except Exception:
                fam = summary.get("reward", "?")
            seed = summary.get("seed")
            b = summary.get("base", {}) or {}
            # a base dump stores its numbers under "base"; if it was produced by a
            # --skip_guided run, "guided" may be absent -> use "base" as the row's
            # primary (guided_*) numbers, since for base Quetzal the model output
            # IS the base output.
            src = b if b else (summary.get("guided", {}) or {})
            row = {
                "name": f"base_quetzal-{fam}", "seed": seed,
                "family": "base_prior", "reward": fam, "guide": "base_prior",
                "objective": "none", "replay": "off", "beta": 0,
            }
            # populate BOTH guided_* and base_* with the base numbers so this row
            # aligns column-wise with guided models (base vs itself -> deltas ~0).
            # actual reward for base: from summary if present, else the npy array
            base_npy = _reward_stats_from_npy(os.path.dirname(path), "base")
            for prefix in ("guided", "base"):
                for k in ("parse_rate", "uniqueness", "log_reward_mean",
                          "log_reward_top1", "log_reward_top10", "log_reward_top100",
                          "atom_stability", "mol_stability", "n_valid_smiles"):
                    row[f"{prefix}_{k}"] = src.get(k)
                for k in ("reward_mean", "reward_top1", "reward_top10", "reward_top100"):
                    row[f"{prefix}_{k}"] = src.get(k) if isinstance(src.get(k), (int, float)) \
                        else base_npy.get(f"base_{k}")
            for dk in ("top1_delta", "top10_delta", "top100_delta", "mean_delta",
                       "reward_mean_delta", "reward_top1_delta", "reward_top10_delta",
                       "reward_top100_delta"):
                row[dk] = 0.0   # base vs base
            n_base_as_row += 1
            rows.append(row)
            continue

        name = summary.get("name") or path.split(os.sep)[-3]
        if name.startswith("_"):
            continue
        seed = summary.get("seed")
        row = {"name": name, "seed": seed}
        row.update(parse_name(name))
        row.update(flatten(summary, dump_dir=os.path.dirname(path)))
        # note whether this row's base came from a reused _base dump
        row["base_reused_from"] = summary.get("base_reused_from") or (
            (summary.get("base") or {}).get("reused_from"))
        rows.append(row)

    if n_base_as_row:
        print(f"[info] added {n_base_as_row} base-Quetzal rows (family=base_prior, "
              f"one per reward-family x seed)")

    if not rows:
        print(f"[error] no model dump_summary.json found under {args.dumps_root}")
        return

    # ---- long csv ----
    all_keys = sorted({k for r in rows for k in r})
    front = ["name", "seed", "family", "reward", "guide", "objective", "replay", "beta"]
    cols = front + [k for k in all_keys if k not in front]
    long_path = os.path.join(args.out_dir, "master_long.csv")
    with open(long_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {long_path} ({len(rows)} rows)")

    # ---- aggregate across seeds -> master_table ----
    by_name = defaultdict(list)
    for r in rows:
        by_name[r["name"]].append(r)

    numeric_keys = [k for k in cols if k not in front + ["seed"]]
    agg_rows = []
    for name, rs in sorted(by_name.items()):
        base = {k: rs[0].get(k) for k in ("name", "family", "reward", "guide",
                                          "objective", "replay", "beta")}
        base["n_seeds"] = len(rs)
        for k in numeric_keys:
            vals = [r.get(k) for r in rs if isinstance(r.get(k), (int, float))]
            if vals:
                base[f"{k}_mean"] = float(np.mean(vals))
                base[f"{k}_std"] = float(np.std(vals))
            else:
                base[f"{k}_mean"] = None
                base[f"{k}_std"] = None
        agg_rows.append(base)

    agg_cols = (["name", "family", "reward", "guide", "objective", "replay",
                 "beta", "n_seeds"]
                + [f"{k}_{s}" for k in numeric_keys for s in ("mean", "std")])
    master_path = os.path.join(args.out_dir, "master_table.csv")
    with open(master_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_cols)
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)
    print(f"[write] {master_path} ({len(agg_rows)} models)")

    # ---- coverage report (essential when aggregating PARTIAL results) ----
    _coverage(args, rows, agg_rows)

    _plots(args, agg_rows)


def _coverage(args, rows, agg_rows):
    """Report which (model, seed) cells are present vs the full grid, plus a
    per-family breakdown, so a half-finished aggregate is honest about what it
    covers. Written to coverage.txt and printed."""
    # seeds actually seen, per model
    seen = defaultdict(set)
    for r in rows:
        seen[r["name"]].add(r.get("seed"))
    n_models = len(seen)
    total_cells = sum(len(v) for v in seen.values())
    # models by seed-count
    by_ct = defaultdict(list)
    for name, seeds in seen.items():
        by_ct[len(seeds)].append(name)
    # per-family completeness
    fam_models = defaultdict(set)
    fam_cells = defaultdict(int)
    for r in agg_rows:
        fam_models[r["family"]].add(r["name"])
    for r in rows:
        fam = r.get("family", "?")
        fam_cells[fam] += 1

    lines = []
    lines.append("=== dump coverage (partial-aware) ===")
    lines.append(f"models with >=1 seed: {n_models}")
    lines.append(f"total (model,seed) cells present: {total_cells}")
    lines.append("")
    lines.append("models by #seeds completed:")
    for ct in sorted(by_ct, reverse=True):
        lines.append(f"  {ct} seed(s): {len(by_ct[ct])} models")
    # which models are NOT yet at 3 seeds (the ones still pending)
    incomplete = sorted(n for n, s in seen.items() if len(s) < 3)
    lines.append("")
    lines.append(f"models with <3 seeds (still pending): {len(incomplete)}")
    for n in incomplete:
        lines.append(f"  {n}: seeds {sorted(x for x in seen[n] if x is not None)}")
    lines.append("")
    lines.append("per-family cells present:")
    for fam in sorted(fam_cells):
        lines.append(f"  {fam}: {fam_cells[fam]} cells across {len(fam_models[fam])} models")
    lines.append("")
    lines.append("NOTE: master_table means/stds are over WHATEVER seeds are present")
    lines.append("      per model (n_seeds column). Rows with n_seeds<3 have higher")
    lines.append("      variance and their std is over fewer points -- treat as")
    lines.append("      provisional until the full grid finishes.")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out_dir, "coverage.txt"), "w") as f:
        f.write(text + "\n")


def _plots(args, agg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable: {e}")
        return

    def bar(metric_mean, metric_std, title, fname, ylabel, filt=None, sort=True):
        # NOTE seed error bars intentionally NOT drawn: with 3 seeds the within-
        # model variance is tiny vs between-model differences, so error bars add
        # clutter without informing the comparison. metric_std kept in the CSV.
        rows = [r for r in agg if (filt is None or filt(r))]
        rows = [r for r in rows if isinstance(r.get(metric_mean), (int, float))]
        if not rows:
            print(f"[plot] {fname}: no data"); return
        if sort:
            rows.sort(key=lambda r: r[metric_mean])
        names = [r["name"].replace("sweep-", "").replace("stability-geom-", "stab-")
                 .replace("base_quetzal-", "BASE:") for r in rows]
        means = [r[metric_mean] for r in rows]
        # color base_prior rows distinctly so the reference stands out
        colors = ["#c44" if r.get("family") == "base_prior" else "#37a"
                  for r in rows]
        h = max(4, 0.28 * len(rows))
        fig, ax = plt.subplots(figsize=(10, h))
        y = np.arange(len(rows))
        ax.barh(y, means, color=colors)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel(ylabel); ax.set_title(title)
        p = os.path.join(args.out_dir, fname)
        fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
        print(f"[plot] {p}")

    # ================= SCATTER PER REWARD (the main comparison view) =========
    # One panel per reward. Each point = a model. x = steering (top10 log-reward
    # delta, guided-base); y = distribution shift (FCD guided-vs-base). Color =
    # guide type. The ceiling lives at the LEFT (delta~0); real steering moves
    # RIGHT. A point high-y but low-x = distribution moved without reward gain
    # (drift, not steering). base_prior sits at the origin (0,0) as reference.
    try:
        import matplotlib.pyplot as plt
        # group models by reward, excluding base_prior (drawn as the origin ref)
        by_reward = defaultdict(list)
        for r in agg:
            if r.get("family") == "base_prior":
                continue
            if isinstance(r.get("top10_delta_mean"), (int, float)):
                by_reward[r["reward"]].append(r)
        rewards = sorted(by_reward)
        if rewards:
            ncol = min(3, len(rewards))
            nrow = (len(rewards) + ncol - 1) // ncol
            fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.5 * nrow),
                                     squeeze=False)
            # stable color per guide type across panels
            guides = sorted({r["guide"] for rs in by_reward.values() for r in rs})
            cmap = plt.get_cmap("tab10")
            gcolor = {g: cmap(i % 10) for i, g in enumerate(guides)}
            for idx, rew in enumerate(rewards):
                ax = axes[idx // ncol][idx % ncol]
                rs = by_reward[rew]
                for r in rs:
                    x = r.get("top10_delta_mean")
                    yv = r.get("fcd_guided_vs_base_mean")
                    if not isinstance(x, (int, float)):
                        continue
                    yv = yv if isinstance(yv, (int, float)) else 0.0
                    ax.scatter(x, yv, color=gcolor[r["guide"]], s=60,
                               edgecolor="k", linewidth=0.4, zorder=3)
                    # short label: guide+obj+replay+beta
                    lab = f"{r['guide'][:4]}/{r['objective']}/{r['replay'][:3]}/b{r['beta']}"
                    ax.annotate(lab, (x, yv), fontsize=5, xytext=(3, 3),
                                textcoords="offset points")
                # origin reference = base Quetzal (delta 0, no shift)
                ax.axvline(0, color="#c44", lw=1, ls="--", zorder=1)
                ax.scatter([0], [0], marker="*", s=200, color="#c44",
                           edgecolor="k", zorder=4, label="base Quetzal")
                ax.set_title(f"{rew}  (n={len(rs)} models)")
                ax.set_xlabel("steering: top10 delta (guided - base)")
                ax.set_ylabel("distribution shift: FCD(guided, base)")
                ax.grid(alpha=0.2)
            # hide unused panels
            for j in range(len(rewards), nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            # one shared legend for guide colors
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], marker="o", ls="", color=gcolor[g],
                              markeredgecolor="k", label=g) for g in guides]
            handles.append(Line2D([0], [0], marker="*", ls="", color="#c44",
                                  markeredgecolor="k", markersize=12, label="base Quetzal"))
            fig.suptitle("Steering vs distribution shift, per reward "
                         "(each point a model; ceiling = clustered near x=0)",
                         y=0.99, fontsize=11)
            fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                       fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.95))
            fig.tight_layout(rect=[0, 0, 1, 0.91])
            p = os.path.join(args.out_dir, "scatter_steering_vs_shift_by_reward.png")
            fig.savefig(p, dpi=140); plt.close(fig)
            print(f"[plot] {p}")
    except Exception as e:
        print(f"[plot] scatter_by_reward failed: {e}")

    # a second scatter: steering (x) vs GEOM realism (y = FCD to ref), per reward.
    # answers "did steering cost realism?" -- points drifting UP moved away from
    # GEOM. Same layout.
    try:
        import matplotlib.pyplot as plt
        by_reward = defaultdict(list)
        for r in agg:
            if r.get("family") == "base_prior":
                continue
            if isinstance(r.get("top10_delta_mean"), (int, float)):
                by_reward[r["reward"]].append(r)
        rewards = sorted(by_reward)
        if rewards:
            ncol = min(3, len(rewards)); nrow = (len(rewards) + ncol - 1) // ncol
            fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.5 * nrow), squeeze=False)
            guides = sorted({r["guide"] for rs in by_reward.values() for r in rs})
            cmap = plt.get_cmap("tab10"); gcolor = {g: cmap(i % 10) for i, g in enumerate(guides)}
            for idx, rew in enumerate(rewards):
                ax = axes[idx // ncol][idx % ncol]
                base_ref = None
                for r in agg:
                    if r.get("family") == "base_prior" and r.get("reward") == rew:
                        base_ref = r.get("fcd_base_vs_ref_mean") or r.get("guided_log_reward_mean_mean")
                for r in by_reward[rew]:
                    x = r.get("top10_delta_mean"); yv = r.get("fcd_guided_vs_ref_mean")
                    if not isinstance(x, (int, float)) or not isinstance(yv, (int, float)):
                        continue
                    ax.scatter(x, yv, color=gcolor[r["guide"]], s=60, edgecolor="k", linewidth=0.4, zorder=3)
                # base realism reference line
                bref = None
                for r in agg:
                    if r.get("family") == "base_prior" and r.get("reward") == rew:
                        bref = r.get("fcd_base_vs_ref_mean")
                if isinstance(bref, (int, float)):
                    ax.axhline(bref, color="#c44", ls="--", lw=1, label="base FCD-to-GEOM")
                ax.set_title(f"{rew}"); ax.set_xlabel("steering: top10 delta")
                ax.set_ylabel("FCD(guided, GEOM)  (lower=realistic)"); ax.grid(alpha=0.2)
            for j in range(len(rewards), nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            fig.suptitle("Steering vs realism cost, per reward "
                         "(up = drifted from GEOM)", y=0.99, fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            p = os.path.join(args.out_dir, "scatter_steering_vs_realism_by_reward.png")
            fig.savefig(p, dpi=140); plt.close(fig)
            print(f"[plot] {p}")
    except Exception as e:
        print(f"[plot] scatter_realism_by_reward failed: {e}")

    # steering: top10 delta per model (the headline ceiling number)
    bar("top10_delta_mean", "top10_delta_std",
        "Steering: top-10 log-reward delta (guided - base)",
        "steering_by_model.png", "top10 delta (near 0 => ceiling)")

    # NEW: top100 and mean deltas per model
    bar("top100_delta_mean", "top100_delta_std",
        "Steering: top-100 log-reward delta (guided - base)",
        "steering_top100_by_model.png", "top100 delta (near 0 => ceiling)")
    bar("mean_delta_mean", "mean_delta_std",
        "Steering: MEAN log-reward delta (guided - base)",
        "steering_mean_by_model.png", "mean delta (near 0 => ceiling)")

    # stability: guided atom stability per model
    bar("guided_atom_stability_mean", "guided_atom_stability_std",
        "Guided atom stability (EDM)",
        "stability_by_model.png", "atom stability")

    # FCD to GEOM ref (guided)
    bar("fcd_guided_vs_ref_mean", "fcd_guided_vs_ref_std",
        "FCD(guided, GEOM) -- lower = closer to data", "fcd_vs_ref.png",
        "FCD to GEOM", sort=True)

    # FCD(guided, base): reward-agnostic distribution shift from the frozen prior.
    bar("fcd_guided_vs_base_mean", "fcd_guided_vs_base_std",
        "FCD(guided, base Quetzal) -- ~0 = no steering, larger = bigger shift",
        "fcd_guided_vs_base_by_model.png", "FCD (guided vs frozen prior)",
        sort=True)

    # FCD(guided,base) grouped by reward family: the direct "did the guide move
    # the distribution, and does it differ by how hard the reward is" view. Shows
    # each family's mean +/- spread across its models, so e.g. nitrogen (easy)
    # vs osim (hard) sit side by side on the same reward-agnostic scale.
    try:
        import matplotlib.pyplot as plt
        fam_vals = defaultdict(list)
        for r in agg:
            v = r.get("fcd_guided_vs_base_mean")
            if isinstance(v, (int, float)):
                fam_vals[r["reward"]].append(v)   # reward (osim/fexo/.../atom_stability)
        if fam_vals:
            fams = sorted(fam_vals, key=lambda f: np.mean(fam_vals[f]))
            means = [np.mean(fam_vals[f]) for f in fams]
            stds = [np.std(fam_vals[f]) for f in fams]
            ns = [len(fam_vals[f]) for f in fams]
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(fams))
            ax.bar(x, means, yerr=stds, capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{f}\n(n={n})" for f, n in zip(fams, ns)], fontsize=8)
            ax.set_ylabel("FCD(guided, base)")
            ax.set_title("Distribution shift from frozen prior, by reward "
                         "(higher = guide moved more)")
            p = os.path.join(args.out_dir, "fcd_guided_vs_base_by_reward.png")
            fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
            print(f"[plot] {p}")
    except Exception as e:
        print(f"[plot] fcd_guided_vs_base_by_reward failed: {e}")

    # reward_mean vs beta, split by guide (steering trend). Uses ACTUAL reward
    # (from npy) when available, else falls back to log-reward.
    try:
        import matplotlib.pyplot as plt
        use_actual = any(isinstance(r.get("guided_reward_mean_mean"), (int, float))
                         for r in agg)
        ykey = "guided_reward_mean_mean" if use_actual else "guided_log_reward_mean_mean"
        ylab = "guided reward mean" if use_actual else "guided log_reward_mean"
        fig, ax = plt.subplots(figsize=(7, 5))
        by_guide = defaultdict(list)
        for r in agg:
            if isinstance(r.get(ykey), (int, float)):
                by_guide[r["guide"]].append(r)
        for guide, rs in sorted(by_guide.items()):
            rs.sort(key=lambda r: r["beta"])
            xs = [r["beta"] for r in rs]
            ys = [r[ykey] for r in rs]
            es = [r.get(ykey.replace("_mean", "_std")) or 0 for r in rs]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=guide)
        ax.set_xlabel("beta"); ax.set_ylabel(ylab)
        ax.set_title(f"Guided {'reward' if use_actual else 'log-reward'} vs beta, "
                     f"by guide type")
        ax.set_xscale("symlog"); ax.legend()
        p = os.path.join(args.out_dir, "reward_mean_by_beta.png")
        fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
        print(f"[plot] {p}")
    except Exception as e:
        print(f"[plot] reward_mean_by_beta failed: {e}")

    # ---- per-reward scatter plots (one subplot per reward family) ----
    # points = models; color = guide type; marker shape = replay. Three variants:
    #   (a) base vs guided reward   -> diagonal = no steering
    #   (b) beta vs guided reward   -> does more tilt help
    #   (c) guided reward vs guided atom-stability -> reward/quality tradeoff
    # Uses ACTUAL reward (reward_top10) as primary since it's the interpretable
    # score; falls back to log if actual missing.
    _scatter_grid(args, agg)
    _decay_curves(args, agg)
    _delta_heatmap(args, agg)


def _scatter_grid(args, agg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable: {e}")
        return

    # color by guide type, marker by replay
    GUIDE_COLORS = {"hidden": "tab:blue", "base": "tab:orange",
                    "tempgain": "tab:green", "base_prior": "black",
                    "linear": "tab:purple", "product": "tab:red",
                    "harmonic": "tab:brown"}
    REPLAY_MARKERS = {"on": "o", "off": "s"}

    def _rv(r, prefix, tier="top10"):
        # prefer actual reward, fall back to log
        v = r.get(f"{prefix}_reward_{tier}_mean")
        if isinstance(v, (int, float)):
            return v
        return r.get(f"{prefix}_log_reward_{tier}_mean")

    rewards = sorted({r["reward"] for r in agg if r.get("reward") not in (None, "?")})
    if not rewards:
        print("[plot] scatter: no reward families"); return

    specs = [
        ("base_vs_guided", "base reward (top10)", "guided reward (top10)",
         lambda r: (_rv(r, "base"), _rv(r, "guided")), True),   # draw diagonal
        ("beta_vs_guided", "beta", "guided reward (top10)",
         lambda r: (r.get("beta"), _rv(r, "guided")), False),
        ("reward_vs_stability", "guided reward (top10)", "guided atom stability",
         lambda r: (_rv(r, "guided"), r.get("guided_atom_stability_mean")), False),
    ]

    for fname, xlab, ylab, getxy, diag in specs:
        ncol = min(len(rewards), 3)
        nrow = (len(rewards) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow),
                                 squeeze=False)
        for i, rew in enumerate(rewards):
            ax = axes[i // ncol][i % ncol]
            sub = [r for r in agg if r.get("reward") == rew]
            xs_all, ys_all = [], []
            for r in sub:
                x, y = getxy(r)
                if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                    continue
                xs_all.append(x); ys_all.append(y)
                color = GUIDE_COLORS.get(r.get("guide"), "gray")
                marker = REPLAY_MARKERS.get(r.get("replay"), "^")
                ax.scatter(x, y, c=color, marker=marker, s=60,
                           edgecolors="k", linewidths=0.4, alpha=0.85)
            if diag and xs_all and ys_all:
                lo = min(min(xs_all), min(ys_all)); hi = max(max(xs_all), max(ys_all))
                ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
            ax.set_title(rew, fontsize=10)
            ax.set_xlabel(xlab, fontsize=8); ax.set_ylabel(ylab, fontsize=8)
        # hide empty subplots
        for j in range(len(rewards), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        # title at the very top; legend BELOW it (not overlapping). Reserve the
        # top strip via subplots_adjust so neither collides with subplot titles.
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markeredgecolor="k", label=g, markersize=8)
                   for g, c in GUIDE_COLORS.items()]
        handles += [Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
                           markeredgecolor="k", label=f"replay {rp}", markersize=8)
                    for rp, m in REPLAY_MARKERS.items()]
        fig.suptitle(f"{xlab} vs {ylab}  (color=guide, shape=replay)",
                     y=0.99, fontsize=11)
        # place legend in its own reserved band just under the title
        fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=7,
                   framealpha=0.9, bbox_to_anchor=(0.5, 0.955))
        # leave room at top for BOTH title and legend band
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        p = os.path.join(args.out_dir, f"scatter_{fname}.png")
        fig.savefig(p, dpi=140); plt.close(fig)
        print(f"[plot] {p}")


def _decay_curves(args, agg):
    """Per-model 'steering decay' curves: delta (guided-base) at
    mean -> top100 -> top10 -> top1, ordered bulk->tail. A model that only
    steers the tail rises steeply toward top1; a model that moves the whole
    distribution is flat. One figure for LOG-reward deltas, one for ACTUAL.
    One subplot per reward family; one line per model, colored by guide."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as e:
        print(f"[plot] matplotlib unavailable: {e}"); return

    GUIDE_COLORS = {"hidden": "tab:blue", "base": "tab:orange",
                    "tempgain": "tab:green", "base_prior": "black",
                    "linear": "tab:purple", "product": "tab:red",
                    "harmonic": "tab:brown"}
    # x order: bulk -> tail
    TIERS = ["mean", "top100", "top10", "top1"]
    variants = [
        ("log",    [f"{t}_delta_mean" for t in TIERS], "log-reward delta (guided - base)"),
        ("actual", [f"reward_{t}_delta_mean" for t in TIERS], "reward delta (guided - base)"),
    ]
    rewards = sorted({r["reward"] for r in agg if r.get("reward") not in (None, "?")})
    if not rewards:
        return

    for vname, cols, ylab in variants:
        # skip the actual variant entirely if no model has actual deltas
        if not any(isinstance(r.get(c), (int, float)) for r in agg for c in cols):
            print(f"[plot] decay_{vname}: no data, skipping"); continue
        ncol = min(len(rewards), 3); nrow = (len(rewards) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
        x = list(range(len(TIERS)))
        for i, rew in enumerate(rewards):
            ax = axes[i // ncol][i % ncol]
            for r in [m for m in agg if m.get("reward") == rew]:
                ys = [r.get(c) for c in cols]
                if not all(isinstance(y, (int, float)) for y in ys):
                    continue
                ax.plot(x, ys, marker="o", ms=4, lw=1,
                        color=GUIDE_COLORS.get(r.get("guide"), "gray"), alpha=0.7)
            ax.axhline(0, color="k", lw=0.6, ls=":")
            ax.set_xticks(x); ax.set_xticklabels(TIERS, fontsize=8)
            ax.set_title(rew, fontsize=10); ax.set_ylabel(ylab, fontsize=8)
            ax.set_xlabel("bulk  ->  tail", fontsize=8)
        for j in range(len(rewards), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        handles = [Line2D([0], [0], color=c, marker="o", label=g)
                   for g, c in GUIDE_COLORS.items()]
        fig.suptitle(f"Steering decay ({vname}-reward): delta vs window "
                     f"(steep rise = tail-only steering)", y=0.99, fontsize=11)
        fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                   fontsize=7, bbox_to_anchor=(0.5, 0.955), framealpha=0.9)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        p = os.path.join(args.out_dir, f"decay_{vname}_by_reward.png")
        fig.savefig(p, dpi=140); plt.close(fig)
        print(f"[plot] {p}")


def _delta_heatmap(args, agg):
    """Heatmap: models (rows) x metrics (cols) of guided-base deltas, so you can
    scan the whole sweep at once for 'who steered on what'. Uses LOG-reward
    deltas (mean/top100/top10/top1) + atom-stability delta + fcd(guided,base).
    Rows grouped/sorted by family then reward then guide. Diverging colormap
    centered at 0 (blue = guide lowered it, red = raised it)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np
    except Exception as e:
        print(f"[plot] matplotlib unavailable: {e}"); return

    metric_cols = [
        ("mean_delta_mean", "Δ mean"),
        ("top100_delta_mean", "Δ top100"),
        ("top10_delta_mean", "Δ top10"),
        ("top1_delta_mean", "Δ top1"),
        ("atom_stability_delta", "Δ atom-stab"),
        ("fcd_guided_vs_base_mean", "FCD(g,base)"),
    ]
    # atom_stability_delta isn't precomputed -> derive from guided/base columns
    for r in agg:
        gv = r.get("guided_atom_stability_mean"); bv = r.get("base_atom_stability_mean")
        r["atom_stability_delta"] = (gv - bv) if isinstance(gv, (int, float)) \
            and isinstance(bv, (int, float)) else None

    rows_sorted = sorted(
        [r for r in agg if r.get("family") != "base_prior"],
        key=lambda r: (r.get("reward", ""), r.get("guide", ""),
                       r.get("objective", ""), r.get("replay", ""), r.get("beta", 0)))
    if not rows_sorted:
        print("[plot] heatmap: no models"); return

    labels = [r["name"].replace("sweep-", "").replace("stability-geom-", "stab-")
              .replace("compose-osimertinib", "cmp") for r in rows_sorted]
    M = _np.full((len(rows_sorted), len(metric_cols)), _np.nan)
    for i, r in enumerate(rows_sorted):
        for j, (col, _) in enumerate(metric_cols):
            v = r.get(col)
            if isinstance(v, (int, float)):
                M[i, j] = v

    # normalize each column to its own scale so one big-magnitude column (FCD)
    # doesn't wash out the reward deltas: z-score per column for COLORING, but
    # annotate with the RAW value.
    Mz = _np.full_like(M, _np.nan)
    for j in range(M.shape[1]):
        col = M[:, j]; ok = ~_np.isnan(col)
        if ok.sum() >= 2 and _np.nanstd(col) > 1e-9:
            Mz[ok, j] = (col[ok] - _np.nanmean(col[ok])) / _np.nanstd(col[ok])
        else:
            Mz[ok, j] = 0.0

    h = max(4, 0.24 * len(rows_sorted)); w = 1.6 * len(metric_cols) + 3
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(Mz, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels([lbl for _, lbl in metric_cols], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows_sorted)))
    ax.set_yticklabels(labels, fontsize=5)
    # annotate raw values
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not _np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=4, color="black")
    ax.set_title("Guided − base deltas across models (color = per-column z-score, "
                 "text = raw value)", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("z-score within column", fontsize=8)
    fig.tight_layout()
    p = os.path.join(args.out_dir, "delta_heatmap.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[plot] {p}")


if __name__ == "__main__":
    main()
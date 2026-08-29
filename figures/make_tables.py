#!/usr/bin/env python3
"""
make_tables.py -- every LaTeX table the paper needs, from the same artifacts the
figures read.

One table per result that is better read as numbers than as a plot. Each is
written as a standalone `\\begin{table}` block with a `\\label`, so it can be
`\\input` directly. A table whose input is missing is skipped with a note rather
than emitted half-empty, and the script still writes the others.

Booktabs is assumed (`\\usepackage{booktabs}`), as is `siunitx` NOT being
required — every number is preformatted here.

OUTPUT (into --out_dir)
  tab_landscape.tex   score landscape per benchmark: guides, prior, baselines
  tab_ceiling.tex     flip rate by prior margin, and the per-component variance
  tab_flip.tex        pooled flip rate by sequence position, both temperatures
  tab_scale.tex       residual scale sweep and residual magnitude
  tab_tempgain.tex    what the temperature head learned
  tab_chemspace.tex   joint-space separability and the decoder artifact rate
  tab_artifact.tex    headline numbers with decoder artifacts excluded

USAGE
  python figures/make_tables.py --out_dir out
"""
import os
import csv
import glob
import json
import argparse
import collections

import numpy as np

import figstyle as fs


# ------------------------------------------------------------------ helpers
def fmt(v, nd=3, dash="--"):
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(f):
        return dash
    return f"{f:.{nd}f}"


def esc(s):
    """LaTeX-safe text. Run names carry underscores."""
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def cell(s):
    """A first-column cell that is safe after a row break.

    A cell beginning with `[` is swallowed by the preceding `\\` as its optional
    vertical-space argument, so a bin label like `[0,1)` silently eats the row.
    Bracing the leading bracket stops that.
    """
    s = str(s)
    return "{" + s + "}" if s.startswith("[") else s


def bin_label(b):
    """Readable margin-bin label: '[32,1e+09)' -> '$\\ge 32$'."""
    s = str(b).strip()
    try:
        lo, hi = s.strip("[)").split(",")
        lo_f, hi_f = float(lo), float(hi)
    except ValueError:
        return esc(s)
    if hi_f >= 1e8:
        return rf"$\ge {lo_f:g}$"
    return rf"$[{lo_f:g},{hi_f:g})$"


def table(body, caption, label, colspec, header, note=None):
    lines = [r"\begin{table}[t]", r"\centering\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header,
             r"\midrule"]
    lines += body
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(rf"\vspace{{2pt}}{{\footnotesize {note}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write(out_dir, name, text):
    path = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    print(f"wrote {path}")


def skip(name, why):
    print(f"[tab] {name}: skipped -- {why}")


# ------------------------------------------------------------- 1. landscape
def tab_landscape(out_dir, exclude=("zaleplon",), ref_budget=5000):
    """Per benchmark: each guide family's score band, the prior, the baselines.

    ref_budget=5000 by default: matches the guide and frozen-prior sample size
    (see scripts/04_dump_guides.sh, final_dump.py --n). Passing 10000 reproduces
    the earlier, unmatched comparison this table used before 2026-08-28 -- kept
    as an option only for reproducing that old table, not a recommended default.
    """
    try:
        rows = fs.load_master_table()
    except SystemExit:
        return skip("tab_landscape", "no master_table.csv")

    have = {r.get("reward") for r in rows if r.get("name", "").startswith("sweep-")}
    benches = [b for b in fs.ALL_BENCHES if b in have and b not in exclude]
    if not benches:
        return skip("tab_landscape", "no guide-sweep rows")

    body = []
    gaps = []
    for bench in benches:
        pts = collections.defaultdict(list)
        for r in rows:
            name = r.get("name", "")
            if not name.startswith("sweep-") or r.get("reward") != bench:
                continue
            fam = fs.guide_family(name)
            if fam is None or any(x in name for x in exclude):
                continue
            v = fs.f(r.get("guided_reward_top10_mean"))
            if np.isfinite(v):
                pts[fam].append(v)
        if not pts:
            continue

        prior = None
        for r in rows:
            if r.get("name") == f"base_quetzal-{bench}":
                prior = fs.f(r.get("base_reward_top10_mean"))
                if not np.isfinite(prior):
                    prior = fs.f(r.get("guided_reward_top10_mean"))
        ref = (fs.load_reference(bench, budget=ref_budget) or {}).get("top10")
        pub = fs.PUBLISHED.get(bench, {})

        body.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{fs.BENCH_TITLE[bench]}}}}} \\")
        allv = []
        for fam in ("guide: base", "guide: hidden", "guide: tempgain"):
            v = pts.get(fam)
            if not v:
                continue
            allv += v
            body.append(f"\\quad {esc(fam)} & {len(v)} & {fmt(np.mean(v))} & "
                        f"{fmt(np.min(v))} & {fmt(np.max(v))} & "
                        f"{fmt(np.max(v) - np.min(v))} \\\\")
        if prior is not None and np.isfinite(prior):
            body.append(f"\\quad frozen prior & 1 & {fmt(prior)} & -- & -- & -- \\\\")
        if allv:
            body.append(r"\quad \emph{all guides} & " +
                        f"{len(allv)} & {fmt(np.mean(allv))} & {fmt(np.min(allv))} & "
                        f"{fmt(np.max(allv))} & {fmt(np.max(allv)-np.min(allv))} \\\\")
        if ref is not None:
            ref_label = f"GEOM best-of-{ref_budget // 1000}k"
            body.append(f"\\quad {ref_label} & -- & {fmt(ref)} & -- & -- & -- \\\\")
        for lab, v in pub.items():
            body.append(f"\\quad {esc(lab)} & -- & {fmt(v)} & -- & -- & -- \\\\")
        body.append(r"\addlinespace")

        # record the comparison rather than asserting it in the caption: on
        # Osimertinib the spread is the LARGER of the two, and a caption that
        # says otherwise is wrong on its own table
        if allv and pub:
            spread = max(allv) - min(allv)
            gap = min(abs(v - max(allv)) for v in pub.values())
            gaps.append((bench, spread, gap))

    if not body:
        return skip("tab_landscape", "nothing to tabulate")

    if gaps:
        note2 = " ".join(
            f"On {fs.BENCH_TITLE[b]} the spread across all guides is "
            f"{s:.3f} against {g:.3f} to the nearest published baseline."
            for b, s, g in gaps)
    else:
        note2 = ""

    write(out_dir, "tab_landscape.tex", table(
        body,
        caption=("Score landscape on the guide sweep. Top-10 mean reward over "
                 "each guide family's configurations, against the frozen prior, "
                 "the matched-budget GEOM-Drugs baseline and the published "
                 "GuacaMol results. Every guide family sits within a few "
                 "thousandths of the frozen prior it was trained on top of."),
        label="tab:landscape",
        colspec="lrrrrr",
        header=r"Configuration & $n$ & mean & min & max & spread \\",
        note=("Guides are reported as a final top-10 over i.i.d.\\ samples. "
              "Published baselines are constants from Brown et al.\\ (2019), "
              "not measurements made here. " + note2)))


# ------------------------------------------------------------- 1b. guide MPO
def _seed_groups(rows, reward_filter=None, beta_filter=None):
    """Group guide-sweep rows by (reward, guide, objective, replay, beta),
    pooling only over training seed. Returns {key: [row, ...]}."""
    groups = collections.defaultdict(list)
    for r in rows:
        name = r.get("name", "")
        if not name.startswith("sweep-"):
            continue
        bench = r.get("reward")
        if reward_filter is not None and bench not in reward_filter:
            continue
        if beta_filter is not None and r.get("beta") not in beta_filter:
            continue
        key = (bench, r.get("guide"), r.get("objective"), r.get("replay"), r.get("beta"))
        groups[key].append(r)
    return groups


def tab_guide_mpo(out_dir, beta="10", exclude=("zaleplon", "nitrogen")):
    """Guide sweep on the MPO benchmarks at one beta, over training seeds.

    Replaces a hand-typed table with no generating code that could not be reproduced
    from results/dumps/_aggregate/master_table.csv and claimed three seeds where only
    two (0, 42) exist (see decisions.md, 2026-08-29). Pools over seed only -- every
    other axis (reward, guide, objective, replay) is its own row, so nothing is
    silently averaged over that isn't named.
    """
    try:
        rows = fs.load_master_table()
    except SystemExit:
        return skip("tab_guide_mpo", "no master_table.csv")

    prior = {}
    for r in rows:
        if r.get("name", "").startswith("base_quetzal-"):
            prior[r.get("reward")] = fs.f(r.get("base_reward_top10_mean"))

    groups = _seed_groups(rows, beta_filter={beta})
    groups = {k: v for k, v in groups.items() if k[0] not in exclude}
    if not groups:
        return skip("tab_guide_mpo", f"no guide-sweep rows at beta={beta}")

    body = []
    prior_note = []
    for bench in sorted({k[0] for k in groups}):
        p = prior.get(bench)
        prior_note.append(f"{fs.BENCH_TITLE.get(bench, bench)}: {fmt(p, 4)}")
    for key in sorted(groups):
        bench, guide, obj, replay, _beta = key
        vals = groups[key]
        p = prior.get(bench)
        deltas = np.array([
            fs.f(r.get("guided_reward_top10_mean")) - p for r in vals
            if np.isfinite(fs.f(r.get("guided_reward_top10_mean"))) and p is not None
            and np.isfinite(p)
        ])
        n = len(deltas)
        mean = deltas.mean() if n else float("nan")
        std = deltas.std(ddof=1) if n > 1 else float("nan")
        fcd = np.nanmean([fs.f(r.get("fcd_guided_vs_base_mean")) for r in vals])
        uniq = np.nanmean([fs.f(r.get("guided_uniqueness_mean")) for r in vals])
        parse = np.nanmean([fs.f(r.get("guided_parse_rate_mean")) for r in vals])
        body.append(
            f"{fs.BENCH_TITLE.get(bench, bench)} & {esc(guide)} & {esc(obj.upper())} & "
            f"{esc(replay)} & {n} & ${mean:+.4f}$ & {fmt(std, 4)} & {fmt(fcd)} & "
            f"{fmt(uniq)} & {fmt(parse)} \\\\")

    write(out_dir, "tab_guide_mpo.tex", table(
        body,
        caption=(rf"Guide sweep on the MPO benchmarks at $\beta{{=}}{beta}$, mean over "
                 r"$n$ training seeds ($n \le 2$: seeds $0$ and $42$; no third seed "
                 r"exists in this sweep). $\Delta$top-10 is guided minus frozen prior "
                 "on the benchmark's own scale; frozen prior top-10 is "
                 + "; ".join(prior_note) + "."),
        label="tab:app-guide-mpo",
        colspec="lllrrrrrrr",
        header=(r"Reward & Guide & Obj. & Replay & $n$ & $\Delta$top-10 & Std & "
                r"FCD$_{g|b}$ & Uniq. & Parse \\"),
        note=("Std is the sample standard deviation over the $n$ seeds shown; a row "
              "with $n=1$ has no Std to report.")))


def tab_nitrogen(out_dir, betas=("1", "10", "100")):
    """Nitrogen control over training seeds, all beta.

    Replaces a hand-typed table with no generating code whose Delta-top-10 for
    hidden/DB/beta=10/replay-off (the paper's headline nitrogen-control number) did
    not reproduce: seed 0 alone gives +0.532 (close to the +0.531 previously shown),
    seed 42 gives +0.266, and the genuine two-seed mean is +0.399 (see decisions.md,
    2026-08-29). This table reports the real two-seed mean and std throughout, not a
    single seed presented as an average.
    """
    try:
        rows = fs.load_master_table()
    except SystemExit:
        return skip("tab_nitrogen", "no master_table.csv")

    prior_row = next((r for r in rows if r.get("name") == "base_quetzal-nitrogen"), None)
    if prior_row is None:
        return skip("tab_nitrogen", "no base_quetzal-nitrogen row")
    prior_top10 = fs.f(prior_row.get("base_reward_top10_mean"))
    prior_mean = fs.f(prior_row.get("base_reward_mean_mean"))

    groups = _seed_groups(rows, reward_filter={"nitrogen"}, beta_filter=set(betas))
    if not groups:
        return skip("tab_nitrogen", "no nitrogen sweep rows")

    rows_out = []
    for key in sorted(groups):
        bench, guide, obj, replay, beta = key
        vals = groups[key]
        deltas = np.array([
            fs.f(r.get("guided_reward_top10_mean")) - prior_top10 for r in vals
            if np.isfinite(fs.f(r.get("guided_reward_top10_mean")))
        ])
        n = len(deltas)
        mean = deltas.mean() if n else float("nan")
        std = deltas.std(ddof=1) if n > 1 else float("nan")
        top10 = np.nanmean([fs.f(r.get("guided_reward_top10_mean")) for r in vals])
        meanv = np.nanmean([fs.f(r.get("guided_reward_mean_mean")) for r in vals])
        fcd = np.nanmean([fs.f(r.get("fcd_guided_vs_base_mean")) for r in vals])
        if n == 0:
            continue
        rows_out.append((mean, guide, obj, beta, replay, n, mean, std, top10, meanv, fcd))

    rows_out.sort(key=lambda t: -t[0])
    body = []
    for _, guide, obj, beta, replay, n, mean, std, top10, meanv, fcd in rows_out:
        body.append(
            f"{esc(guide)} & {esc(obj.upper())} & {beta} & {esc(replay)} & {n} & "
            f"${mean:+.4f}$ & {fmt(std, 4)} & {fmt(top10)} & {fmt(meanv)} & {fmt(fcd)} \\\\")

    write(out_dir, "tab_nitrogen.tex", table(
        body,
        caption=(r"Nitrogen control, mean over $n$ training seeds ($n \le 2$: seeds "
                 r"$0$ and $42$; no third seed exists). The frozen prior scores "
                 f"{fmt(prior_top10, 4)} (top-10) and {fmt(prior_mean, 4)} (mean). "
                 r"$\Delta$ is guided minus frozen prior."),
        label="tab:app-nitrogen",
        colspec="llllrrrrrr",
        header=(r"Guide & Obj. & $\beta$ & Replay & $n$ & $\Delta$top-10 & Std & "
                r"Top-10 & Mean & FCD$_{g|b}$ \\"),
        note=("Std is the sample standard deviation over the $n$ seeds shown; a row "
              "with $n=1$ has no Std to report. Sorted by $\\Delta$top-10, "
              "descending.")))


def tab_discriminator(out_dir, exclude=("zaleplon",)):
    """Best Delta-top-10 by reward, over every guide/objective/beta/replay cell.

    Replaces a hand-typed table with no generating code (see decisions.md,
    2026-08-29): the per-benchmark best-Delta and scored-fraction figures it showed
    did not match a direct recomputation from results/dumps/_aggregate/master_table.csv
    for any of the three MPO benchmarks. Pools over training seed within each
    (guide, objective, beta, replay) cell, as tab_guide_mpo/tab_nitrogen do, then
    takes the best seed-pooled mean per reward.
    """
    try:
        rows = fs.load_master_table()
    except SystemExit:
        return skip("tab_discriminator", "no master_table.csv")

    rewards = [b for b in list(fs.ALL_BENCHES) + ["nitrogen"] if b not in exclude]
    structure = {b: "multi-property" for b in fs.ALL_BENCHES}
    structure["nitrogen"] = "atom-decomposable"
    title = dict(fs.BENCH_TITLE, nitrogen="Nitrogen fraction")

    prior = {}
    for r in rows:
        if r.get("name", "").startswith("base_quetzal-"):
            prior[r.get("reward")] = fs.f(r.get("base_reward_top10_mean"))

    groups = _seed_groups(rows)
    body = []
    for bench in rewards:
        p = prior.get(bench)
        if p is None or not np.isfinite(p):
            continue
        best = None
        for key, vals in groups.items():
            if key[0] != bench:
                continue
            deltas = np.array([
                fs.f(r.get("guided_reward_top10_mean")) - p for r in vals
                if np.isfinite(fs.f(r.get("guided_reward_top10_mean")))
            ])
            if len(deltas) == 0:
                continue
            m = deltas.mean()
            if best is None or m > best:
                best = m
        parse = [fs.f(r.get("guided_parse_rate_mean")) for r in rows
                 if r.get("name", "").startswith("sweep-") and r.get("reward") == bench
                 and np.isfinite(fs.f(r.get("guided_parse_rate_mean")))]
        scored = (f"{min(parse):.2f}--{max(parse):.2f}" if parse else "--")
        if best is None:
            continue
        body.append(f"{title.get(bench, bench)} & {scored} & "
                     f"${best:+.3f}$ & {esc(structure.get(bench, '--'))} \\\\")

    if not body:
        return skip("tab_discriminator", "nothing to tabulate")

    write(out_dir, "tab_discriminator.tex", table(
        body,
        caption=("Best $\\Delta$top-10 by reward, over every guide architecture, "
                 "objective, $\\beta$ and replay setting, pooled over training seed "
                 "within each cell. Scored fraction is the range of the guided "
                 "parse rate (fraction of generated molecules that convert to a "
                 "scoreable SMILES) across the sweep."),
        label="tab:app-discriminator",
        colspec="lrrl",
        header=r"Reward & Scored fraction & Best $\Delta$top-10 & Structure \\"))


# -------------------------------------------------------------- 1c. delivery
def tab_delivery(out_dir, betas=("1", "10"), exclude=("zaleplon",)):
    """Delivery rate, TV, sampled-flip rate and the margin ceiling, pooled per guide.

    Backs the Results-section sentences that report these headline diagnostics. Pools
    raw counts (not per-run rates) over every flip report at beta in `betas`, weighted
    by n_states, so a run of 400 states does not weigh as much as one of 40,000 -- the
    same convention tab_flip() uses for the by-position curves. Restricted to
    beta in {1, 10} (the balanced grid): the temperature-gain guide has no beta=100
    runs by design, and folding beta=100 into base/hidden here roughly doubles their
    flip rates relative to the beta<=10 numbers this table reports, since beta=100 is
    a stress test meant to push harder, not part of the balanced comparison.
    """
    try:
        reports = fs.load_flip_reports(fs.FLIPS_DIR, "1.0")
    except SystemExit:
        return skip("tab_delivery", "no flip reports")

    import re
    agg = collections.defaultdict(collections.Counter)
    n_states = collections.Counter()
    n_cfg = collections.Counter()
    keys = ["delivered", "sample_flip", "argmax_flip", "mass_moved_sum",
            "gap_hi_states", "gap_hi_flipped", "gap_lo_states", "gap_lo_flipped",
            "prior_top1_logit_gap_sum"]
    for label, (_full, blk) in reports.items():
        if any(x in label for x in exclude):
            continue
        fam = fs.guide_family(label)
        if fam is None:
            continue
        m = re.search(r"-b(\d+)(?:-s\d+)?$", label)
        if m is None or m.group(1) not in betas:
            continue
        n = blk.get("n_states", 0)
        raw = blk.get("raw", {})
        n_states[fam] += n
        n_cfg[fam] += 1
        for k in keys:
            agg[fam][k] += raw.get(k, 0)

    fams = [f for f in ("guide: base", "guide: hidden", "guide: tempgain") if n_states[f]]
    if not fams:
        return skip("tab_delivery", "no matching flip reports")

    overall = collections.Counter()
    for fam in fams:
        for k in agg[fam]:
            overall[k] += agg[fam][k]
    tot = sum(n_states[f] for f in fams)

    short = {"guide: base": "base", "guide: hidden": "hidden", "guide: tempgain": "tempgain"}

    def row(label, n, a):
        hi_flip = a["gap_hi_flipped"] / a["gap_hi_states"] if a["gap_hi_states"] else float("nan")
        lo_flip = a["gap_lo_flipped"] / a["gap_lo_states"] if a["gap_lo_states"] else float("nan")
        return " & ".join([
            label,
            fmt(a["delivered"] / n, 3),
            fmt(100 * a["sample_flip"] / n, 3),
            fmt(100 * a["mass_moved_sum"] / n, 3),
            fmt(a["prior_top1_logit_gap_sum"] / n, 2),
            fmt(100 * a["gap_hi_states"] / n, 2),
            fmt(100 * hi_flip, 4),
            fmt(100 * lo_flip, 3)]) + r" \\"

    body = [row(short[f], n_states[f], agg[f]) for f in fams]
    body.append(r"\midrule")
    body.append(row(r"\emph{pooled}", tot, overall))

    write(out_dir, "tab_delivery.tex", table(
        body,
        caption=("Delivery rate, sampled-flip rate and the margin ceiling, pooled over "
                 rf"the balanced grid ($\beta \in \{{{','.join(betas)}\}}$, both training "
                 "objectives, both training seeds, replay on and off) by guide "
                 "architecture. Sample flip is the coupled sampled-disagreement rate "
                 "against the frozen prior; high/low-gap flip are the same rate "
                 "restricted to decisions above/below a prior top-1 margin of 8."),
        label="tab:delivery",
        colspec="lrrrrrrr",
        header=(r"Guide & Delivered & Sample flip \% & TV \% & Mean gap & "
                r"High-gap frac \% & High-gap flip \% & Low-gap flip \% \\"),
        note=(f"Pooled from raw per-state counts over {sum(n_cfg[f] for f in fams)} "
              "guide configurations; a run of 400 states does not weigh as much as one "
              "of 40{,}000. Zaleplon is excluded as a negative control.")))


# --------------------------------------------------------------- 2. ceiling
def tab_ceiling(out_dir):
    path = os.path.join(fs.ABL_DIR, "ceiling", "ceiling_report.json")
    if not os.path.exists(path):
        return skip("tab_ceiling", "no ceiling_report.json")
    rep = json.load(open(path))
    sat = rep.get("saturation") or {}
    bins = sat.get("gap_bins") or []
    per = sat.get("per_guide") or {}
    if not bins or not per:
        return skip("tab_ceiling", "no saturation block")

    short = dict(zip(per, fs.distinguishing_labels(list(per))))
    dfrac = sat.get("decision_frac_by_gap") or []

    body = []
    for i, b in enumerate(bins):
        cells = [bin_label(b), fmt(dfrac[i] if i < len(dfrac) else None)]
        for name, d in per.items():
            fr = d.get("flip_rate_by_gap") or []
            cells.append(fmt(fr[i] if i < len(fr) else None, 4))
        body.append(" & ".join(cells) + r" \\")

    hi = sum(v for b, v in zip(bins, dfrac)
             if _lo(b) >= 8) if dfrac else float("nan")
    header = ("Prior top-1 margin & decisions & " +
              " & ".join(esc(short[n]) for n in per) + r" \\")

    comp = rep.get("component_variance") or {}
    note = (f"{100*hi:.1f}\\% of decisions sit at a margin above 8. ")
    if comp:
        # `or 1` would treat a std of exactly 0.0 as missing -- which is the one
        # value this test exists to catch
        def _std(v):
            s = v.get("std_logr_valid")
            return float(s) if s is not None else float("inf")

        dead = [k for k, v in comp.items() if _std(v) < 0.05]
        note += ("Per-component log-reward standard deviation over prior "
                 "samples: " +
                 ", ".join(f"{esc(k)} {fmt(v.get('std_logr_valid'), 4)}"
                           for k, v in comp.items()) + ". ")
        if dead:
            note += ("A component below 0.05 carries no gradient and is "
                     "unsteerable by construction: " +
                     ", ".join(esc(d) for d in dead) + ".")

    write(out_dir, "tab_ceiling.tex", table(
        body,
        caption=("The steering ceiling. Sampled-flip rate within each bin of "
                 "the frozen prior's top-1 logit margin, with the fraction of "
                 "decisions falling in that bin. The flip rate reaches zero "
                 "above a margin of 4, while most decisions lie above 8."),
        label="tab:ceiling",
        colspec="l" + "r" * (1 + len(per)),
        header=header, note=note))


def _lo(b):
    try:
        return float(str(b).strip("[)").split(",")[0])
    except Exception:
        return -1.0


# ------------------------------------------------------------------ 3. flip
def tab_flip(out_dir, temps=("1.0", "0.3"), max_pos=8, exclude=("zaleplon",)):
    import make_fig02_positional as f02
    cols = {}
    for t in temps:
        try:
            reports = fs.load_flip_reports(fs.FLIPS_DIR, t)
        except SystemExit:
            continue
        rate, gap, states, used = f02.pooled_curves(reports, max_pos, None,
                                                    exclude)
        cols[t] = (rate, gap, states, used)
    if not cols:
        return skip("tab_flip", "no flip reports")

    any_gap = next((g for _, (r, g, s, u) in cols.items() if g is not None), None)
    body = []
    for i in range(max_pos):
        cells = [str(i)]
        for t in temps:
            if t not in cols:
                continue
            rate, gap, states, used = cols[t]
            cells.append(fmt(rate[i] if i < len(rate) else None, 4))
        cells.append(fmt(any_gap[i] if any_gap is not None
                         and i < len(any_gap) else None, 2))
        cells.append(str(int(cols[temps[0]][2][i])) if i < len(cols[temps[0]][2])
                     else "--")
        body.append(" & ".join(cells) + r" \\")

    used = cols[temps[0]][3]
    header = ("Position & " +
              " & ".join(rf"flip rate $T={t}$" for t in temps if t in cols) +
              r" & prior margin & states \\")
    write(out_dir, "tab_flip.tex", table(
        body,
        caption=("Guide influence along the construction path. Coupled "
                 "sampled-flip rate by sequence position, pooled over "
                 f"{used} guide configurations at both flip temperatures, with "
                 "the frozen prior's mean top-1 logit margin at the same "
                 "positions. The margin growth accounts for the decay."),
        label="tab:flip",
        colspec="l" + "r" * (len([t for t in temps if t in cols]) + 2),
        header=header,
        note=("Rates are pooled from raw per-position counts -- numerators and "
              "denominators summed, divided once -- so a run of 400 states does "
              "not weigh as much as one of 40{,}000. Zaleplon is excluded as a "
              "negative control.")))


# ----------------------------------------------------------------- 4. scale
def tab_scale(out_dir):
    path = os.path.join(fs.ABL_DIR, "guide-harmonic", "ablation_report.json")
    if not os.path.exists(path):
        return skip("tab_scale", "no guide-harmonic ablation_report.json")
    rep = json.load(open(path))
    sweep = rep.get("B_scale_sweep") or {}
    resid = rep.get("A_residual") or {}
    if not sweep:
        return skip("tab_scale", "no B_scale_sweep block")

    def scale_of(k):
        return float(str(k).replace("scale_", ""))

    body = []
    for k in sorted(sweep, key=scale_of):
        d = sweep[k]
        body.append(" & ".join([
            f"{scale_of(k):g}$\\times$",
            fmt(d.get("mean_shift"), 4),
            fmt(d.get("median_shift"), 4),
            fmt(d.get("wasserstein1"), 4),
            fmt(d.get("validity"), 3),
            fmt(d.get("uniqueness"), 3)]) + r" \\")

    note = ""
    if resid:
        short = dict(zip(resid, fs.distinguishing_labels(list(resid))))
        note = ("Residual magnitude relative to the prior's logits, at scale "
                "1$\\times$: " +
                ", ".join(f"{esc(short[n])} "
                          f"{fmt(d.get('residual_ratio'), 4)} "
                          f"(KL {fmt(d.get('mean_KL_guided_vs_prior'), 4)})"
                          for n, d in resid.items()) + ".")

    write(out_dir, "tab_scale.tex", table(
        body,
        caption=("Scaling the trained residual at sampling time. Mean and "
                 "median shift in log-reward against the frozen prior, the "
                 "Wasserstein-1 distance between the two reward "
                 "distributions, and the cost in validity and uniqueness."),
        label="tab:scale",
        colspec="lrrrrr",
        header=(r"Residual scale & mean shift & median shift & $W_1$ & "
                r"validity & uniqueness \\"),
        note=note))


# -------------------------------------------------------------- 5. tempgain
def tab_tempgain(out_dir):
    path = os.path.join(fs.ABL_DIR, "tempgain", "tempgain_probe.json")
    if not os.path.exists(path):
        return skip("tab_tempgain", "no tempgain_probe.json")
    rep = json.load(open(path))
    learned = rep.get("learned") or {}
    per = learned.get("per_guide") or {}
    per = {k: v for k, v in per.items() if "note" not in v}
    if not per:
        return skip("tab_tempgain", "no guide with temperature heads")

    short = dict(zip(per, fs.distinguishing_labels(list(per))))
    body = []
    for name, d in per.items():
        t = [v for v in (d.get("T_by_gap") or [])
             if isinstance(v, (int, float))]
        g = [v for v in (d.get("g_by_gap") or [])
             if isinstance(v, (int, float))]
        fr = [v for v in (d.get("flip_rate_by_gap") or [])
              if isinstance(v, (int, float))]
        body.append(" & ".join([
            esc(short[name]),
            fmt(min(t), 4) if t else "--",
            fmt(max(t), 4) if t else "--",
            fmt(min(g), 4) if g else "--",
            fmt(max(g), 4) if g else "--",
            fmt(max(fr), 4) if fr else "--"]) + r" \\")

    write(out_dir, "tab_tempgain.tex", table(
        body,
        caption=("What the temperature and gain heads learned. Range of the "
                 "learned temperature $T(h)$ and gain $g(h)$ across bins of the "
                 "prior's top-1 margin. The forward pass applies "
                 "$\\mathrm{clamp}(T,\\min=1)$, so any $T \\le 1$ is inert."),
        label="tab:tempgain",
        colspec="lrrrrr",
        header=(r"Guide & $T_{\min}$ & $T_{\max}$ & $g_{\min}$ & $g_{\max}$ & "
                r"max flip rate \\")))


# ------------------------------------------------------------- 6. chemspace
def tab_chemspace(out_dir, sep=None, artifact=None):
    """Separability and artifact rate. Both are passed in by the caller."""
    if not sep and not artifact:
        return skip("tab_chemspace", "no chemspace measurements supplied")
    body = []
    if sep:
        body.append(r"\multicolumn{2}{l}{\emph{Recovering the objective}} \\")
        body.append(rf"\quad {sep['k']}-NN accuracy & {fmt(sep['acc'])} \\")
        body.append(rf"\quad majority class & {fmt(sep['majority'])} \\")
        body.append(rf"\quad shuffled-label null & {fmt(sep['null'])} \\")
        if sep.get("acc_clean") is not None:
            body.append(rf"\quad accuracy, artifacts excluded & "
                        rf"{fmt(sep['acc_clean'])} \\")
        body.append(r"\addlinespace")
    if artifact:
        body.append(r"\multicolumn{2}{l}{\emph{Charged-carbon rate by source}} \\")
        for k, v in artifact.items():
            body.append(rf"\quad {esc(k)} & {fmt(v, 4)} \\")
    write(out_dir, "tab_chemspace.tex", table(
        body,
        caption=("The joint chemical space. A $k$-nearest-neighbour classifier "
                 "is asked to name the objective a molecule's guide was trained "
                 "against, from its fingerprint alone. It does not beat the "
                 "majority class or the shuffled-label null, so the families do "
                 "not occupy distinguishable regions. The lower block gives the "
                 "rate of formal charge on carbon, the signature of a "
                 "3D-to-SMILES bond-perception failure."),
        label="tab:chemspace",
        colspec="lr",
        header=r"Quantity & value \\",
        note=("GEOM-Drugs is real SMILES and never passes through the decoder, "
              "which is why its artifact rate is the reference point.")))


# -------------------------------------------------------------- 7. artifact
def tab_artifact(out_dir):
    path = os.path.join(os.path.dirname(fs.DUMPS_AGG), "artifact_exclusion.csv")
    if not os.path.exists(path):
        return skip("tab_artifact",
                    "no artifact_exclusion.csv -- run "
                    "ablations/exclude_decoder_artifacts.py")
    rows = list(csv.DictReader(open(path)))
    guided = [r for r in rows if r.get("source") == "guided"]
    if not guided:
        return skip("tab_artifact", "no guided rows")

    import re

    def fam(run):
        p = run.split("-")
        return p[1] if len(p) > 1 else "?"

    def arch(run):
        for a in ("base", "hidden", "tempgain"):
            if f"-{a}-" in run:
                return a
        return "?"

    def beta(run):
        m = re.search(r"-b(\d+)(?:-s\d+)?$", run)
        return m.group(1) if m else "?"

    def row(label, rs):
        af = np.array([float(r["artifact_frac"]) for r in rs])
        a = np.array([float(r["top10_all"]) for r in rs])
        c = np.array([float(r["top10_clean"]) for r in rs])
        ok = np.isfinite(a) & np.isfinite(c)
        return " & ".join([
            label, str(len(rs)), fmt(af.mean(), 3), fmt(af.max(), 3),
            fmt(np.nanmean(a)), fmt(np.nanmean(c)),
            f"{np.nanmean(c[ok] - a[ok]):+.4f}"]) + r" \\"

    body = [r"\multicolumn{7}{l}{\emph{By objective}} \\"]
    for f in ("osim", "peri", "fexo", "zaleplon", "nitrogen"):
        rs = [r for r in guided if fam(r["run"]) == f]
        if rs:
            body.append("\\quad " + row(esc(f), rs))

    body.append(r"\addlinespace")
    body.append(r"\multicolumn{7}{l}{\emph{By guide architecture and $\beta$}} \\")
    for a in ("base", "hidden", "tempgain"):
        for b in ("1", "10", "100"):
            rs = [r for r in guided
                  if arch(r["run"]) == a and beta(r["run"]) == b]
            if rs:
                body.append("\\quad " + row(rf"{esc(a)}, $\beta={b}$", rs))

    allf = np.array([float(r["artifact_frac"]) for r in guided])
    d = np.array([float(r["top10_clean"]) - float(r["top10_all"])
                  for r in guided], dtype=float)
    d = d[np.isfinite(d)]

    write(out_dir, "tab_artifact.tex", table(
        body,
        caption=("Headline numbers with 3D-to-SMILES failures removed. A "
                 "molecule carrying a formal charge on a carbon atom is a "
                 "bond-perception artifact rather than chemistry: it sanitises "
                 "cleanly and so passes every downstream filter. Removing them "
                 "moves the top-10 mean by less than 0.01 on average, so no "
                 "reported comparison turns on them. The rate itself is not "
                 "uniform -- it is flat for the base and temperature-gain "
                 "guides at every $\\beta$, and rises sharply for the hidden "
                 "guide at $\\beta=100$."),
        label="tab:artifact",
        colspec="lrrrrrr",
        header=(r"Group & runs & artifact rate & worst & top-10 all & "
                r"top-10 clean & $\Delta$ \\"),
        note=(f"Across all {len(guided)} guided dumps the artifact rate is "
              f"{allf.mean():.3f} on average (range "
              f"{allf.min():.3f}--{allf.max():.3f}) and dropping the artifacts "
              f"shifts the top-10 mean by {d.mean():+.4f} "
              f"({d.min():+.4f} to {d.max():+.4f}). GEOM-Drugs contains none.")))


# ------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--exclude", default="zaleplon",
                    help="guide families kept out of the pooled tables")
    ap.add_argument("--temps", default="1.0,0.3")
    ap.add_argument("--sep_json", default=None,
                    help="optional JSON with the fig16/fig17 measurements")
    ap.add_argument("--ref_budget", type=int, default=5000,
                    help="GEOM-Drugs reference sample size for tab_landscape; "
                         "5000 matches the guide/prior sample size, 10000 "
                         "reproduces the earlier unmatched comparison")
    args = ap.parse_args()

    exclude = tuple(x for x in args.exclude.split(",") if x)
    temps = tuple(t.strip() for t in args.temps.split(",") if t.strip())

    tab_landscape(args.out_dir, exclude, ref_budget=args.ref_budget)
    tab_guide_mpo(args.out_dir)
    tab_nitrogen(args.out_dir)
    tab_discriminator(args.out_dir, exclude=exclude)
    tab_delivery(args.out_dir, exclude=exclude)
    tab_ceiling(args.out_dir)
    tab_flip(args.out_dir, temps, exclude=exclude)
    tab_scale(args.out_dir)
    tab_tempgain(args.out_dir)
    tab_artifact(args.out_dir)

    if args.sep_json and os.path.exists(args.sep_json):
        d = json.load(open(args.sep_json))
        tab_chemspace(args.out_dir, d.get("separability"), d.get("artifact"))
    else:
        skip("tab_chemspace",
             "pass --sep_json from make_fig17_decoder_artifact.py --emit_json")


if __name__ == "__main__":
    main()

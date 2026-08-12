"""
Extended analysis for harvested training streams. Imported by harvest_eval.py.

Everything here is computed from molecules.jsonl (i, epoch, smiles, log_reward,
n_atoms) plus optional shard_*.pt coords. Nothing requires re-running training
or changing rtb_finetune.py.
"""

import os
import math
import json
import warnings
from collections import Counter, defaultdict

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, QED, Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# ---- optional: synthetic accessibility -------------------------------------
_SASCORER = None
try:
    import sys
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer as _SASCORER  # noqa
except Exception:
    _SASCORER = None

# ---- optional: PAINS / quality ---------------------------------------------
_PAINS = None
try:
    from rdkit.Chem import FilterCatalog
    _p = FilterCatalog.FilterCatalogParams()
    _p.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    _PAINS = FilterCatalog.FilterCatalog(_p)
except Exception:
    _PAINS = None


# ============================ quality filters ================================

ALLOWED_ELEMENTS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}


def quality_flags(smi):
    """Approximation of the GuacaMol / rd_filters compound-quality screen.

    NOT the official filter set -- it is a rule-of-thumb reimplementation using
    RDKit's PAINS catalog plus the structural heuristics from Brown et al.
    Report it as an internal quality proxy, not as "the GuacaMol quality score".
    Useful because osimertinib MPO is a known reward-hacking target: a high
    score with a low pass rate means the scorer was exploited.
    """
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else smi
    if m is None:
        return None
    f = {}
    f["pains"] = bool(_PAINS.HasMatch(m)) if _PAINS is not None else None
    f["bad_element"] = any(a.GetAtomicNum() not in ALLOWED_ELEMENTS for a in m.GetAtoms())
    f["charged"] = Chem.GetFormalCharge(m) != 0
    ri = m.GetRingInfo()
    sizes = [len(r) for r in ri.AtomRings()]
    f["macrocycle"] = any(s > 8 for s in sizes)
    f["small_ring"] = any(s < 5 for s in sizes)
    f["too_many_rings"] = ri.NumRings() > 6
    f["mw_out_of_range"] = not (150.0 <= Descriptors.MolWt(m) <= 650.0)
    f["heavy_out_of_range"] = not (10 <= m.GetNumHeavyAtoms() <= 60)
    f["rotb_high"] = rdMolDescriptors.CalcNumRotatableBonds(m) > 12
    f["long_chain"] = m.HasSubstructMatch(Chem.MolFromSmarts("[R0;D2][R0;D2][R0;D2][R0;D2][R0;D2][R0;D2]"))
    if _SASCORER is not None:
        try:
            f["sa_high"] = _SASCORER.calculateScore(m) > 6.0
        except Exception:
            f["sa_high"] = None
    f["pass"] = not any(v for v in f.values() if v is True)
    return f


def quality_report(smiles_list):
    flags = [quality_flags(s) for s in smiles_list]
    flags = [f for f in flags if f is not None]
    if not flags:
        return {}
    keys = [k for k in flags[0] if k != "pass"]
    out = {"n": len(flags),
           "pass_rate": float(np.mean([f["pass"] for f in flags]))}
    out["fail_reasons"] = {
        k: float(np.mean([bool(f.get(k)) for f in flags])) for k in keys}
    return out


# ============================ descriptors ====================================

def descriptors(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    d = {
        "mw": Descriptors.MolWt(m),
        "logp": Crippen.MolLogP(m),
        "qed": QED.qed(m),
        "tpsa": rdMolDescriptors.CalcTPSA(m),
        "hbd": rdMolDescriptors.CalcNumHBD(m),
        "hba": rdMolDescriptors.CalcNumHBA(m),
        "rings": m.GetRingInfo().NumRings(),
        "arom_rings": rdMolDescriptors.CalcNumAromaticRings(m),
        "rotb": rdMolDescriptors.CalcNumRotatableBonds(m),
        "heavy": m.GetNumHeavyAtoms(),
        "frac_csp3": rdMolDescriptors.CalcFractionCSP3(m),
    }
    if _SASCORER is not None:
        try:
            d["sa"] = _SASCORER.calculateScore(m)
        except Exception:
            pass
    return d


def descriptor_table(smiles_list):
    rows = [descriptors(s) for s in smiles_list]
    rows = [r for r in rows if r is not None]
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "std": float(np.std([r[k] for r in rows])),
                "median": float(np.median([r[k] for r in rows]))} for k in keys}


# ============================ diversity ======================================

def _fps(smiles_list, radius=2, nbits=2048):
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    out = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out.append(gen.GetFingerprint(m))
    return out


def internal_diversity(smiles_list, max_n=200):
    """1 - mean pairwise Tanimoto over ECFP4. Higher = more spread out.

    A collapsed policy that rediscovers one scaffold scores near 0 here even
    while its top-k reward looks excellent -- which is exactly the failure this
    catches.
    """
    fps = _fps(smiles_list[:max_n])
    if len(fps) < 2:
        return float("nan")
    sims = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    return float(1.0 - np.mean(sims)) if sims else float("nan")


def scaffold_stats(smiles_list):
    scaffs = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            scaffs.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:
            pass
    if not scaffs:
        return {}
    c = Counter(scaffs)
    return {"n_scaffolds": len(c),
            "scaffold_diversity": len(c) / len(scaffs),
            "top_scaffold_frac": c.most_common(1)[0][1] / len(scaffs),
            "top_scaffold": c.most_common(1)[0][0]}


def novelty(smiles_list, ref_smiles_set):
    if not ref_smiles_set:
        return float("nan")
    canon = set()
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            canon.add(Chem.MolToSmiles(m))
    if not canon:
        return float("nan")
    return float(len(canon - ref_smiles_set) / len(canon))


def nearest_neighbour_similarity(smiles_list, ref_smiles, max_n=100, max_ref=5000):
    """Max Tanimoto of each molecule to the reference set, averaged.

    Low = the run is producing genuinely new chemistry; high = it is
    regurgitating the prior's training distribution.
    """
    fps = _fps(smiles_list[:max_n])
    rfps = _fps(list(ref_smiles)[:max_ref])
    if not fps or not rfps:
        return float("nan")
    best = [max(DataStructs.BulkTanimotoSimilarity(f, rfps)) for f in fps]
    return float(np.mean(best))


# ============================ sample efficiency ==============================

def first_hit(stream, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """Oracle calls until the first molecule at or above each threshold."""
    out = {}
    for t in thresholds:
        hit = next((i for i, s in stream if s >= t), None)
        out[f"calls_to_{t}"] = int(hit) if hit is not None else None
    return out


def yield_above(stream, thresholds=(0.5, 0.7, 0.8, 0.9)):
    """Fraction of unique valid molecules at or above each threshold.

    Top-k says how good the best molecules are; this says how OFTEN the policy
    produces them -- the difference between a lucky sample and a steered
    distribution, which is the crux of the ceiling argument.
    """
    if not stream:
        return {}
    s = np.array([x[1] for x in stream])
    return {f"yield_{t}": float((s >= t).mean()) for t in thresholds}


def bucketize(rows, stream_scores, n_buckets=20):
    """Per-bucket curves over generation order: validity, concentration, mean
    and best score, molecule size. This is the mode-collapse detector -- a
    rising top-share or falling effective count while score climbs is the
    signature.

    width uses ceil, not floor: with floor, a total that isn't an exact
    multiple of n_buckets leaves a tiny remainder bucket holding a handful of
    molecules, whose top-share spikes and effective-count collapses purely
    because the sample is small. That looks exactly like mode collapse at the
    end of every run. Buckets holding less than a third of the nominal width
    are additionally masked to nan.
    """
    if not rows:
        return {}
    hi = max(r["i"] for r in rows) + 1
    width = max(1, math.ceil(hi / n_buckets))
    min_count = max(5, width // 3)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["i"] // width].append(r)

    score_by_i = dict(stream_scores)
    xs, valid, uniq, smean, sbest, natoms = [], [], [], [], [], []
    top_share, eff_distinct = [], []
    for b in sorted(buckets):
        rs = buckets[b]
        if len(rs) < min_count:
            continue          # undersized trailing window: statistics unreliable
        smis = [r.get("smiles") for r in rs]
        ok = [s for s in smis if s]
        xs.append((b + 0.5) * width)
        valid.append(len(ok) / len(rs))
        uniq.append(len(set(ok)) / len(ok) if ok else 0.0)
        # Unique-count fraction saturates once the window exceeds the number of
        # distinct molecules, so it misses a policy piling probability onto a
        # few modes. These two see it: top_share is the share held by the single
        # most common molecule, eff_distinct is exp(Shannon entropy), i.e. the
        # effective number of molecules the policy is actually producing.
        if ok:
            c = Counter(ok)
            n = len(ok)
            top_share.append(c.most_common(1)[0][1] / n)
            p = np.array([v / n for v in c.values()])
            eff_distinct.append(float(np.exp(-(p * np.log(p)).sum())))
        else:
            top_share.append(float("nan")); eff_distinct.append(float("nan"))
        sc = [score_by_i[r["i"]] for r in rs if r["i"] in score_by_i]
        smean.append(float(np.mean(sc)) if sc else float("nan"))
        sbest.append(float(np.max(sc)) if sc else float("nan"))
        na = [r.get("n_atoms") for r in rs if r.get("n_atoms")]
        natoms.append(float(np.mean(na)) if na else float("nan"))
    return {"calls": xs, "validity": valid, "uniqueness": uniq,
            "top_share": top_share, "eff_distinct": eff_distinct,
            "score_mean": smean, "score_best": sbest, "n_atoms": natoms}


def best_so_far_curve(stream, k=10, every=100, budget=None):
    """Top-k mean as a function of oracle calls -- the curve AUC integrates."""
    if not stream:
        return {"calls": [], "topk": []}
    hi = budget or (stream[-1][0] + 1)
    xs, ys, ptr, run = [0], [0.0], 0, []
    for g in range(every, hi + 1, every):
        while ptr < len(stream) and stream[ptr][0] < g:
            run.append(stream[ptr][1]); ptr += 1
        top = sorted(run, reverse=True)[:k]
        xs.append(g); ys.append(float(np.mean(top)) if top else 0.0)
    return {"calls": xs, "topk": ys}


# ============================ MPO components =================================

def component_breakdown(bench_key, smiles_list):
    """Score each leaf scorer of an MPO benchmark separately.

    Tells you WHICH axis of the MPO moved. Critical for osimertinib, where one
    component is a dead axis (ECFP6 with std_logr=0, flat by construction) --
    if the aggregate moved, this says whether it came from the live components
    or from something degenerate.
    """
    try:
        from guacamol import standard_benchmarks as SB
    except Exception:
        return None

    fn = getattr(SB, bench_key, None)
    if fn is None:
        return None
    try:
        bench = fn()
        obj = bench.objective
    except Exception:
        return None

    leaves = []

    def walk(node, path="obj"):
        subs = getattr(node, "scoring_functions", None)
        if subs:
            for j, s in enumerate(subs):
                walk(s, f"{path}.{j}")
        else:
            leaves.append((path, node))

    walk(obj)
    if not leaves:
        return None

    out = {}
    for path, leaf in leaves:
        name = getattr(leaf, "name", None) or type(leaf).__name__
        vals = []
        for s in smiles_list:
            try:
                vals.append(float(leaf.score(s)))
            except Exception:
                pass
        if vals:
            out[f"{path}:{name}"] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "max": float(np.max(vals)),
                "dead_axis": bool(np.std(vals) < 1e-6)}
    return out


# ============================ consistency check ==============================

def stored_vs_rescored(rows, scorer, n=300):
    """Compare exp(log_reward) recorded at training time against a fresh score.

    A large gap means the run's reward config differs from what you are
    scoring against now -- a silent config-drift bug, the same class as the
    save_hyperparameters nesting issue.
    """
    pairs = []
    for r in rows:
        if len(pairs) >= n:
            break
        smi, lr = r.get("smiles"), r.get("log_reward")
        if not smi or lr is None or lr < -20:
            continue
        try:
            pairs.append((float(math.exp(lr)), scorer(smi)))
        except Exception:
            pass
    if len(pairs) < 10:
        return {}
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    return {"n": len(pairs),
            "mean_abs_diff": float(np.mean(np.abs(a - b))),
            "corr": float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan"),
            "agrees": bool(np.mean(np.abs(a - b)) < 0.02)}


# ============================ plots ==========================================

def make_plots(per_run, out_dir, bench="", ref=None):
    """One figure per question. `ref` is an optional dict with keys
    scores / descriptors / name, e.g. a GEOM subset, drawn as the reference
    distribution the prior was trained on."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    os.makedirs(out_dir, exist_ok=True)
    written = []
    names = list(per_run.keys())
    cmap = plt.get_cmap("tab20")
    colors = {n: cmap(i % 20) for i, n in enumerate(names)}
    ref_name = (ref or {}).get("name", "reference")

    def save(fig, fname):
        p = os.path.join(out_dir, fname)
        fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
        written.append(p)

    def legend_outside(ax, ncol=1):
        ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  ncol=ncol, frameon=False)

    # 1. sample efficiency: top-10 best-so-far vs oracle calls
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n in names:
        c = per_run[n].get("curve_top10")
        if c and c["calls"]:
            ax.plot(c["calls"], c["topk"], label=n, color=colors[n], lw=1.6)
    if ref and ref.get("top10") is not None:
        ax.axhline(ref["top10"], ls="--", c="k", lw=1.2,
                   label=f"{ref_name} best-of-set top-10")
    ax.set_xlabel("oracle calls"); ax.set_ylabel("top-10 mean score")
    ax.set_title(f"sample efficiency ({bench})"); ax.grid(alpha=.3)
    legend_outside(ax); save(fig, "sample_efficiency.png")

    # 2. collapse panel
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    panel = [("validity", "3D->SMILES rate"),
             ("top_share", "share of most common mol"),
             ("eff_distinct", "effective distinct mols"),
             ("n_atoms", "mean heavy atoms")]
    for (key, lab), ax in zip(panel, axes):
        for n in names:
            b = per_run[n].get("buckets")
            if b and b.get("calls") and key in b:
                ax.plot(b["calls"], b[key], label=n, color=colors[n], lw=1.4)
        ax.set_xlabel("oracle calls"); ax.set_ylabel(lab); ax.grid(alpha=.3)
    axes[2].set_yscale("log")
    legend_outside(axes[3])
    fig.suptitle("distribution health over training "
                 "(rising top-share or falling effective count = collapse)",
                 fontsize=10)
    save(fig, "collapse_panel.png")

    # 3. score distribution, with the reference set behind it
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if ref and ref.get("scores"):
        ax.hist(ref["scores"], bins=50, density=True, color="0.75",
                label=f"{ref_name} (prior training set)", zorder=0)
    for n in names:
        sc = per_run[n].get("all_scores")
        if sc:
            ax.hist(sc, bins=50, histtype="step", label=n, color=colors[n],
                    density=True, lw=1.5)
    ax.set_xlabel("score"); ax.set_ylabel("density")
    ax.set_title(f"score distribution vs {ref_name}")
    ax.grid(alpha=.3); legend_outside(ax); save(fig, "score_hist.png")

    # 4. quality vs performance -- the reward-hacking plot
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for n in names:
        q = per_run[n].get("quality", {}).get("pass_rate")
        t = per_run[n].get("top100")
        if q is not None and t is not None and not math.isnan(t):
            ax.scatter(t, q, s=80, color=colors[n], label=n)
    if ref and ref.get("quality_pass_rate") is not None and ref.get("top100") is not None:
        ax.scatter(ref["top100"], ref["quality_pass_rate"], s=140, marker="*",
                   color="k", label=f"{ref_name}")
    ax.set_xlabel("top-100 mean score"); ax.set_ylabel("quality pass rate")
    ax.set_title("performance vs compound quality"); ax.grid(alpha=.3)
    ax.set_ylim(-0.05, 1.05); legend_outside(ax)
    save(fig, "quality_vs_score.png")

    # 5. distribution-learning metrics: validity, uniqueness, V x U, novelty
    metrics = [("validity", "validity"), ("uniqueness", "uniqueness"),
               ("valid_x_unique", "validity x uniqueness"),
               ("novelty", "novelty vs ref")]
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(names)), 4.2))
    x = np.arange(len(metrics)); w = 0.8 / max(1, len(names))
    any_bar = False
    for j, n in enumerate(names):
        vals = [per_run[n].get("dist_metrics", {}).get(k) for k, _ in metrics]
        vals = [v if isinstance(v, (int, float)) and not math.isnan(v) else 0.0
                for v in vals]
        if any(vals):
            any_bar = True
        ax.bar(x + j * w - 0.4 + w / 2, vals, w, label=n, color=colors[n])
    if any_bar:
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in metrics], fontsize=8)
        ax.set_ylabel("fraction"); ax.set_ylim(0, 1.05)
        ax.set_title("distribution-learning metrics (over all unique molecules)")
        ax.grid(alpha=.3, axis="y"); legend_outside(ax)
        save(fig, "validity_uniqueness.png")
    else:
        plt.close(fig)

    # 6. diversity of the top-100
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(names)), 4))
    vals = [(n, per_run[n].get("diversity", {}).get("internal_diversity"),
             per_run[n].get("diversity", {}).get("scaffold_diversity"))
            for n in names]
    vals = [v for v in vals if v[1] is not None and not math.isnan(v[1] or float("nan"))]
    if vals:
        x = np.arange(len(vals)); w = 0.38
        ax.bar(x - w/2, [v[1] for v in vals], w, label="internal diversity")
        ax.bar(x + w/2, [v[2] or 0 for v in vals], w, label="scaffold diversity")
        ax.set_xticks(x); ax.set_xticklabels([v[0] for v in vals], rotation=25,
                                             ha="right", fontsize=7)
        ax.set_ylabel("fraction"); ax.set_title("diversity of top-100")
        ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
        save(fig, "diversity.png")
    else:
        plt.close(fig)

    # 7. descriptor panel for the top-100, with the reference as a dashed line
    keys = ["mw", "logp", "qed", "sa", "rings", "arom_rings", "rotb",
            "tpsa", "heavy"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for k, ax in zip(keys, axes.ravel()):
        xs, ys, es = [], [], []
        for n in names:
            d = per_run[n].get("descriptors", {}).get(k)
            if d:
                xs.append(n); ys.append(d["mean"]); es.append(d["std"])
        if xs:
            ax.bar(range(len(xs)), ys, yerr=es, capsize=2,
                   color=[colors[n] for n in xs])
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs, rotation=35, ha="right", fontsize=5.5)
        rd = (ref or {}).get("descriptors", {}).get(k)
        if rd:
            ax.axhline(rd["mean"], ls="--", c="k", lw=1.1)
            ax.text(0.99, 0.95, ref_name, transform=ax.transAxes, fontsize=6,
                    ha="right", va="top")
        ax.set_title(k, fontsize=9); ax.grid(alpha=.3, axis="y")
    fig.suptitle(f"top-100 property profile (dashed = {ref_name})", fontsize=11)
    save(fig, "descriptors.png")

    # 8. MPO component breakdown
    comp_runs = {n: per_run[n]["components"] for n in names
                 if per_run[n].get("components")}
    if comp_runs:
        labels = sorted({k for v in comp_runs.values() for k in v})
        # leaf class names repeat (two TanimotoScoringFunction, two Rdkit...),
        # so disambiguate with the tree path and flag zero-variance axes
        disp = []
        for l in labels:
            path, _, cls = l.partition(":")
            dead = any(comp_runs[n].get(l, {}).get("dead_axis")
                       for n in comp_runs)
            disp.append(f"[{path.replace('obj.', '')}] {cls[:22]}"
                        + (" (DEAD)" if dead else ""))
        fig, ax = plt.subplots(figsize=(max(9, 2.2 * len(labels)), 4.6))
        x = np.arange(len(labels)); w = 0.8 / max(1, len(comp_runs))
        for j, (n, comp) in enumerate(comp_runs.items()):
            ax.bar(x + j * w - 0.4 + w / 2,
                   [comp.get(l, {}).get("mean", 0) for l in labels], w,
                   label=n, color=colors[n])
        ax.set_xticks(x)
        ax.set_xticklabels(disp, rotation=18, ha="right", fontsize=7)
        ax.set_ylabel("mean component score"); ax.set_ylim(0, 1.15)
        ax.set_title("which MPO axis moved (top-100)")
        ax.grid(alpha=.3, axis="y")
        legend_outside(ax)
        save(fig, "mpo_components.png")

    return written
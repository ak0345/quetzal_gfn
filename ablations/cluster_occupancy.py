#!/usr/bin/env python3
"""
cluster_occupancy.py -- do the guides reach new chemistry, or reweight the
prior's existing modes?

THE QUESTION. A guide that raises reward could be doing either of two things.
It could be finding regions of chemical space the frozen prior essentially never
samples, or it could be sampling the prior's own modes in different proportions.
Reward alone cannot tell these apart, and neither can validity or FCD.

WHY THIS IS NOT JUST CLUSTER OCCUPANCY. The obvious design is to partition a
shared space and compare per-bin occupancy. Measured on this prior's own dumps,
that partition is nearly degenerate:

    1,496 molecules ->  1,322 Murcko scaffolds, 84.7% of molecules alone in
                                                their scaffold
                          990 generic scaffolds, 57.6% alone
                        1,136 ECFP4 leader clusters at Tanimoto 0.35
                          593 at 0.25

With most bins holding a single molecule, "occupies a bin the prior never
occupies" is a statement about sampling, not about chemistry: a scaffold seen
once in 1,500 molecules is missed by the next 1,500 draws about a third of the
time. Occupancy metrics on such a partition measure n, not behaviour.

WHAT IS MEASURED INSTEAD. The primary statistic is continuous and needs no bins:

  nn_to_prior   for each molecule of a source, the maximum Tanimoto similarity
                to ANY molecule of the prior sample. This asks directly whether
                a molecule sits inside the region the prior actually visits,
                and it degrades gracefully as diversity rises.

Bin-based occupancy is still computed, at a deliberately coarse cutoff, for the
reweighting question that genuinely needs bins. Every partition reports its own
singleton fraction so a degenerate one is visible rather than silently trusted.

READING THE NUMBER. Higher mean nn_to_prior means MORE concentrated inside the
prior's well-sampled regions, not less. A fresh draw from the prior spreads over
its whole support, so many of its molecules land where the reference sample is
sparse and score low. A source that concentrates on a subregion the prior covers
densely scores high.

THE NULL IS THE WHOLE POINT. None of these numbers means anything alone, so
every run also reports the identical statistic computed between two independent
draws from the SAME prior (dump seed 0 against dump seed 42). That is the floor
set by sampling. On osim it looks like this:

    PRIOR seed42 (the null)                    mean nn = 0.340
    sweep-osim-base-db-replay_off-b1-s0        mean nn = 0.503
    sweep-osim-base-db-replay_off-b10-s0       mean nn = 0.435
    sweep-osim-base-db-replay_off-b100-s0      mean nn = 0.417

Every guide sits ABOVE the null, so on this evidence the guides concentrate
within chemistry the prior already reaches rather than finding new regions, and
the concentration weakens as beta rises.

Equal sample sizes are enforced throughout. Coverage, novel fractions and
nearest-neighbour distances are all n-dependent, so comparing a 4,600 molecule
prior against a 3,900 molecule guide would manufacture a difference out of
sample size alone.

UMAP IS NOT USED HERE. Neighbour embeddings are for looking at, not for
measuring: cluster sizes and inter-cluster distances in a UMAP plot are not
quantities. Visualisation lives in figures/, driven by the assignments this
writes.

INPUTS
  results/dumps/_base/<family>/seed<k>/base_smiles.txt      the prior
  results/dumps/<run>/seed<k>/guided_smiles.txt             each guide

OUTPUTS (--out_dir)
  occupancy_<family>_s<k>.json    every statistic, including the null
  occupancy_<family>_s<k>.csv     the same as a flat table
  ..._assignments.csv             molecule -> (source, scaffold, cluster)

USAGE
  python ablations/cluster_occupancy.py --family osim
  python ablations/cluster_occupancy.py --family peri --n 2000 --cutoff 0.25
"""

import os
import re
import csv
import json
import glob
import math
import argparse
import random

import numpy as np

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


# ------------------------------------------------------------------ loading

def load_smiles(path, limit=None):
    """SMILES from a dumper file, in generation order, blanks dropped."""
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
                if limit and len(out) >= limit:
                    break
    return out


def discover_sources(dumps_root, family, dump_seed, max_guides=0):
    """(prior_path, [(run_name, guided_path), ...]) for one reward family.

    The prior lives under _base/<family>/ and is shared by every guide in the
    family, which is what makes the occupancy comparison well posed: all sources
    are being compared against the same reference sample.
    """
    prior = os.path.join(dumps_root, "_base", family, f"seed{dump_seed}",
                         "base_smiles.txt")
    if not os.path.isfile(prior):
        raise SystemExit(f"[fatal] no prior dump at {prior}\n"
                         f"        run scripts/04_dump_guides.sh first")
    pat = os.path.join(dumps_root, f"sweep-{family}-*", f"seed{dump_seed}",
                       "guided_smiles.txt")
    guides = []
    for p in sorted(glob.glob(pat)):
        run = os.path.basename(os.path.dirname(os.path.dirname(p)))
        guides.append((run, p))
    if max_guides:
        guides = guides[:max_guides]
    return prior, guides


# ------------------------------------------------------------- partitioning

def scaffold_key(mol, generic=False):
    """Bemis-Murcko framework as canonical SMILES, or "" when there is none.

    An acyclic molecule has an empty scaffold. That is a real category rather
    than a failure, so it is kept as its own bin instead of being dropped: if a
    guide shifts mass toward acyclic structures that is a finding.
    """
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            core = MurckoScaffold.MakeScaffoldGeneric(core)
        return Chem.MolToSmiles(core)
    except Exception:
        return "<error>"


def leader_clusters(fps, cutoff, max_leaders, rng):
    """Sphere-exclusion clustering; returns the leaders' indices.

    Greedy and order-dependent, so the input is shuffled with a fixed seed to
    keep it reproducible. A molecule within `cutoff` Tanimoto of an existing
    leader joins it, otherwise it becomes a leader itself.

    Leaders come from a REFERENCE POOL drawn evenly across sources rather than
    from the full set: the full set is hundreds of thousands of molecules and
    the assignment step is already O(N x leaders). Drawing the pool evenly is
    what keeps the space shared rather than dominated by whichever source has
    the most runs.
    """
    order = list(range(len(fps)))
    rng.shuffle(order)
    leaders = []
    for i in order:
        if not leaders:
            leaders.append(i)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in leaders])
        if max(sims) < cutoff:
            leaders.append(i)
            if len(leaders) >= max_leaders:
                break
    return leaders


def assign_to_leaders(fps, leader_fps, cutoff):
    """Nearest leader per molecule, or -1 when nothing is within `cutoff`.

    -1 is kept as its own bin ("outlier"): a molecule too far from every leader
    is exactly the kind of thing this analysis is looking for, and folding it
    into the nearest leader anyway would hide it.
    """
    out = np.empty(len(fps), dtype=np.int32)
    for i, fp in enumerate(fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, leader_fps)
        j = int(np.argmax(sims))
        out[i] = j if sims[j] >= cutoff else -1
    return out


def counts_of(labels):
    c = {}
    for x in labels:
        c[x] = c.get(x, 0) + 1
    return c


# --------------------------------------------------- nearest-neighbour stats

def nn_similarity(query_fps, ref_fps):
    """Max Tanimoto from each query molecule to ANY reference molecule.

    This is the primary statistic. It needs no partition, so it does not
    inherit the degeneracy that makes bin occupancy unreliable on a sample this
    diverse, and it answers the question directly: a molecule with a close
    neighbour in the prior sample is somewhere the prior already goes.
    """
    return np.array([max(DataStructs.BulkTanimotoSimilarity(f, ref_fps))
                     for f in query_fps], dtype=float)


def internal_diversity(fps, rng, n_pairs=20000):
    """1 - mean pairwise Tanimoto over random pairs.

    Sampled rather than exhaustive: the full matrix is O(n^2) and the mean is
    estimated to three decimals long before that. Reported because a source can
    raise its nearest-neighbour similarity to the prior either by moving into a
    dense region or by collapsing, and only diversity separates the two.
    """
    n = len(fps)
    if n < 2:
        return None
    tot = 0.0
    for _ in range(n_pairs):
        i = rng.randrange(n); j = rng.randrange(n)
        if i == j:
            j = (j + 1) % n
        tot += DataStructs.TanimotoSimilarity(fps[i], fps[j])
    return 1.0 - tot / n_pairs


def singleton_fraction(labels):
    """Fraction of molecules alone in their bin.

    A partition where this is high is measuring sample size rather than
    chemistry, so it is reported next to every bin-based number instead of
    being left for the reader to discover.
    """
    c = counts_of(labels)
    n = len(labels) or 1
    return sum(v for v in c.values() if v == 1) / n


# ----------------------------------------------------------------- metrics

def _dist(counts, keys):
    v = np.array([counts.get(k, 0) for k in keys], dtype=float)
    t = v.sum()
    return v / t if t > 0 else v


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits, so it is bounded in [0, 1].

    Symmetric and finite even when the supports differ, which KL is not: a bin
    the prior never occupies would make KL infinite, and those bins are the
    entire point here.
    """
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def compare(src_counts, prior_counts, top_k=10):
    """Occupancy metrics for one source against the prior sample."""
    keys = sorted(set(src_counts) | set(prior_counts), key=str)
    p = _dist(src_counts, keys)
    q = _dist(prior_counts, keys)

    novel = [k for k in keys if prior_counts.get(k, 0) == 0]
    n_src = sum(src_counts.values()) or 1
    novel_mass = sum(src_counts.get(k, 0) for k in novel) / n_src

    # the reweighting component: drop bins the prior never occupies, renormalise
    shared = [k for k in keys if prior_counts.get(k, 0) > 0]
    if shared:
        ps = _dist({k: src_counts.get(k, 0) for k in shared}, shared)
        qs = _dist({k: prior_counts.get(k, 0) for k in shared}, shared)
        js_shared = js_divergence(ps, qs)
    else:
        js_shared = None

    # how much of the source still sits in the prior's biggest modes
    top = sorted(prior_counts, key=lambda k: -prior_counts[k])[:top_k]
    top_mass_src = sum(src_counts.get(k, 0) for k in top) / n_src
    n_prior = sum(prior_counts.values()) or 1
    top_mass_prior = sum(prior_counts[k] for k in top) / n_prior

    return {
        "n": int(n_src),
        "bins_occupied": int(sum(1 for k in keys if src_counts.get(k, 0) > 0)),
        "js_vs_prior": js_divergence(p, q),
        "novel_mass": novel_mass,
        "novel_bins": len([k for k in novel if src_counts.get(k, 0) > 0]),
        "js_shared": js_shared,
        f"top{top_k}_prior_mode_mass": top_mass_src,
        f"top{top_k}_prior_mode_mass_in_prior": top_mass_prior,
    }


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps_root", default="results/dumps")
    ap.add_argument("--family", required=True,
                    help="reward family: osim | peri | fexo | nitrogen | zaleplon")
    ap.add_argument("--dump_seed", type=int, default=0)
    ap.add_argument("--null_seed", type=int, default=42,
                    help="a SECOND prior dump seed, used as the sampling-noise "
                         "floor; without it novel_mass is uninterpretable")
    ap.add_argument("--n", type=int, default=1500,
                    help="molecules per source; 0 = the smallest source's size. "
                         "The nearest-neighbour step is O(n^2) per source, and "
                         "1500 already pins the mean to three decimals")
    # 0.25, not the usual 0.35-0.7. Measured on this prior, 0.35 leaves 76% of
    # molecules as their own cluster and 0.55 leaves 97%; at 0.25 it is 40%,
    # which is still coarse but leaves bins with enough mass to compare.
    ap.add_argument("--cutoff", type=float, default=0.25,
                    help="Tanimoto similarity for the same leader cluster")
    ap.add_argument("--max_leaders", type=int, default=1500)
    ap.add_argument("--ref_pool", type=int, default=4000,
                    help="molecules drawn evenly across sources to build the "
                         "cluster space")
    ap.add_argument("--max_guides", type=int, default=0, help="0 = all")
    ap.add_argument("--generic_scaffolds", action="store_true",
                    help="also bin by element-agnostic Murcko frameworks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/occupancy")
    ap.add_argument("--no_assignments", action="store_true",
                    help="skip the per-molecule csv (it is the large output)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    prior_path, guides = discover_sources(args.dumps_root, args.family,
                                          args.dump_seed, args.max_guides)
    if not guides:
        raise SystemExit(f"[fatal] no guided dumps for family {args.family!r} "
                         f"at dump seed {args.dump_seed}")

    # the null draw: the same prior at a different dump seed
    null_path = os.path.join(args.dumps_root, "_base", args.family,
                             f"seed{args.null_seed}", "base_smiles.txt")
    have_null = os.path.isfile(null_path)
    if not have_null:
        print(f"[warn] no second prior dump at {null_path}; the sampling-noise "
              f"floor cannot be computed and novel_mass will have nothing to "
              f"be compared against", flush=True)

    sources = [("__prior__", prior_path)]
    if have_null:
        sources.append(("__prior_null__", null_path))
    sources += guides

    raw = {name: load_smiles(path) for name, path in sources}
    n_eq = args.n or min(len(v) for v in raw.values())
    print(f"[occupancy] family={args.family} seed={args.dump_seed} "
          f"sources={len(sources)} n per source={n_eq}", flush=True)

    # EQUAL n, always: coverage and novel fractions are strongly n-dependent, so
    # unequal samples would manufacture differences out of sample size alone.
    sample = {}
    for name, smis in raw.items():
        if len(smis) < n_eq:
            raise SystemExit(f"[fatal] {name} has {len(smis)} < {n_eq}")
        sample[name] = rng.sample(smis, n_eq)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols, fps, scafs, gscafs, owner = {}, {}, {}, {}, []
    for name in sample:
        ms, fs, sc, gc = [], [], [], []
        for smi in sample[name]:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            ms.append(smi)
            fs.append(gen.GetFingerprint(m))
            sc.append(scaffold_key(m))
            gc.append(scaffold_key(m, generic=True) if args.generic_scaffolds else "")
        mols[name], fps[name], scafs[name], gscafs[name] = ms, fs, sc, gc
        print(f"[occupancy]   {name}: {len(ms)}/{n_eq} parsed", flush=True)

    # ---- cluster space, built from an EVENLY drawn reference pool ----
    per_src = max(1, args.ref_pool // len(sample))
    ref_fps, ref_idx = [], []
    for name in sample:
        idx = list(range(len(fps[name])))
        rng.shuffle(idx)
        for i in idx[:per_src]:
            ref_fps.append(fps[name][i]); ref_idx.append((name, i))
    print(f"[occupancy] leader clustering over {len(ref_fps)} reference mols "
          f"at Tanimoto {args.cutoff} ...", flush=True)
    leaders = leader_clusters(ref_fps, args.cutoff, args.max_leaders, rng)
    leader_fps = [ref_fps[i] for i in leaders]
    print(f"[occupancy] {len(leader_fps)} leaders", flush=True)

    clusters = {}
    for name in sample:
        clusters[name] = assign_to_leaders(fps[name], leader_fps, args.cutoff)
        print(f"[occupancy]   assigned {name}", flush=True)

    # ---- nearest-neighbour to the prior: the primary statistic ----
    ref = fps["__prior__"]
    prior_smiles = set(mols["__prior__"])
    nn = {}
    for name in sample:
        if name == "__prior__":
            continue          # comparing the reference to itself gives all 1.0
        d = nn_similarity(fps[name], ref)
        # EXACT rediscovery: the same canonical SMILES, not merely a similar
        # molecule. Two independent prior draws overlap at ~0.2% here, so
        # anything well above that is the guide converging back onto specific
        # molecules the prior already produced rather than finding new ones.
        exact = sum(1 for x in mols[name] if x in prior_smiles) / max(len(mols[name]), 1)
        nn[name] = {
            "mean": float(d.mean()), "median": float(np.median(d)),
            "p05": float(np.percentile(d, 5)), "p95": float(np.percentile(d, 95)),
            # "outside the prior's sampled neighbourhood" at two thresholds, so
            # the reader is not asked to trust one arbitrary cut
            "frac_below_0.4": float((d < 0.4).mean()),
            "frac_below_0.3": float((d < 0.3).mean()),
            "internal_diversity": internal_diversity(fps[name], rng),
            "frac_identical_to_prior": exact,
            "unique_fraction": len(set(mols[name])) / max(len(mols[name]), 1),
        }
        print(f"[nn] {name}: mean={d.mean():.4f} "
              f"frac<0.4={(d < 0.4).mean():.3f} "
              f"identical_to_prior={100*exact:.1f}%", flush=True)

    null_nn = nn.get("__prior_null__")
    if null_nn:
        print(f"[null] two prior draws give mean nn={null_nn['mean']:.4f}. "
              f"A source ABOVE this is more concentrated inside chemistry the "
              f"prior already reaches, not less.", flush=True)

    # ---- metrics ----
    spaces = {"scaffold": scafs, "cluster": {k: list(v) for k, v in clusters.items()}}
    if args.generic_scaffolds:
        spaces["generic_scaffold"] = gscafs

    report = {"family": args.family, "dump_seed": args.dump_seed,
              "null_seed": args.null_seed if have_null else None,
              "n_per_source": n_eq, "cutoff": args.cutoff,
              "n_leaders": len(leader_fps),
              "prior_internal_diversity": internal_diversity(ref, rng),
              "nn_to_prior": nn, "spaces": {}}

    rows = []
    for space, labels in spaces.items():
        prior_counts = counts_of(labels["__prior__"])
        # the singleton fraction sits next to the numbers it qualifies: a
        # partition where most molecules are alone in their bin is reporting
        # sample size, and its novel_mass should not be read as chemistry
        sing = singleton_fraction(labels["__prior__"])
        if sing > 0.5:
            print(f"[warn] {space}: {100*sing:.1f}% of prior molecules are alone "
                  f"in their bin; treat novel_mass in this space as a statement "
                  f"about n, not about chemistry", flush=True)
        block = {"prior_bins_occupied":
                 int(sum(1 for v in prior_counts.values() if v > 0)),
                 "prior_singleton_fraction": sing,
                 "sources": {}}
        for name in labels:
            if name == "__prior__":
                continue
            m = compare(counts_of(labels[name]), prior_counts)
            m["singleton_fraction"] = singleton_fraction(labels[name])
            block["sources"][name] = m
            row = {"space": space, "source": name, **m}
            row.update({f"nn_{k}": v for k, v in nn.get(name, {}).items()})
            rows.append(row)
        report["spaces"][space] = block

        null = block["sources"].get("__prior_null__")
        if null:
            print(f"[null] {space}: two prior draws give "
                  f"novel_mass={null['novel_mass']:.4f} "
                  f"js={null['js_vs_prior']:.4f} -- any guide at or below this "
                  f"is indistinguishable from resampling the prior", flush=True)

    base = os.path.join(args.out_dir, f"occupancy_{args.family}_s{args.dump_seed}")
    with open(base + ".json", "w") as f:
        json.dump(report, f, indent=2)
    if rows:
        with open(base + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"[occupancy] wrote {base}.json / .csv", flush=True)

    if not args.no_assignments:
        p = base + "_assignments.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source", "smiles", "scaffold", "cluster"])
            for name in sample:
                for smi, sc, cl in zip(mols[name], scafs[name], clusters[name]):
                    w.writerow([name, smi, sc, int(cl)])
        print(f"[occupancy] wrote {p} (input for the UMAP figure)", flush=True)


if __name__ == "__main__":
    main()

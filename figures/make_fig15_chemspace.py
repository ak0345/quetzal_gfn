#!/usr/bin/env python3
"""
make_fig15_chemspace.py -- where the guides sit relative to GEOM and the prior.

WHAT THE PANELS SHOW, AND WHY THESE PANELS.

Three things were measured before choosing them, on osim seed 0 at n=600-1500:

  a 2D PCA of ECFP4 explains 5.2% of the variance, and source centroids are
    separated by 0.40x the within-source spread. A raw scatter is one blob.
  descriptors barely move: mean MW is 357 for GEOM, 339 for the prior, 335 and
    330 for guides at beta 1 and 100. Same for logP, TPSA, ring count and QED.
  nearest-neighbour similarity to the prior separates cleanly: 0.32 between two
    independent prior draws against 0.44 for a guide.

So the effect is real but it lives at a FINER scale than either a 2D embedding
or a 0.25-cutoff cluster resolves, which is also why stage 9 finds cluster-level
js_shared sitting on top of its null. Panels A and B are therefore the
quantitative ones and panel C is explicitly qualitative.

  A  nearest-neighbour similarity to the PRIOR, as an ECDF per source, with the
     prior-vs-prior null drawn as the reference. Curves to the RIGHT of the null
     are more concentrated inside chemistry the prior already reaches. This is
     the result.
  B  nearest-neighbour similarity to GEOM. Included because the prior itself
     sits at about 0.30 from its own training corpus, which is the context every
     guide number needs: nothing here is close to GEOM.
  C  a 2D embedding, UMAP when it is importable and PCA otherwise, with GEOM as
     a backdrop and the sources over it. The axes carry the variance explained
     so the reader can see how little of the story is in them. Present for
     orientation, not for measurement: distances and cluster sizes in a
     neighbour embedding are not quantities, and at 5% variance explained
     neither are these.

INPUTS   reference/geom_drugs_smiles.txt
         results/dumps/_base/<family>/seed<k>/base_smiles.txt
         results/dumps/sweep-<family>-*/seed<k>/guided_smiles.txt

USAGE
  python figures/make_fig15_chemspace.py --family osim
  python figures/make_fig15_chemspace.py --family peri --runs sweep-peri-hidden-db-replay_off-b10-s0
"""
import os
import re
import glob
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def charged_carbon(mol):
    """A formal charge on a carbon atom: the signature of a bond-perception
    failure, not of chemistry.

    rdDetermineBonds infers bond orders from 3D coordinates alone. When no
    assignment satisfies every valence it balances the books with alternating
    formal charges, typically around an aromatic ring, and the result sanitises
    cleanly while being chemically absurd. Measured here, GEOM-Drugs contains
    ZERO such molecules in 4,000 while the prior's own dumps contain 13.8% and
    the guides 15-17%, so this is a property of the 3D-to-SMILES step rather
    than of anything the model learned.
    """
    return any(a.GetAtomicNum() == 6 and a.GetFormalCharge() != 0
               for a in mol.GetAtoms())


def load_fps(path, n, rng, want_bits=False):
    """(fingerprints, bit matrix, artifact flags) from a SMILES file."""
    smis = [l.strip().split()[0] for l in open(path) if l.strip()]
    if len(smis) > n:
        smis = rng.sample(smis, n)
    fps, bits, bad = [], [], []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = _GEN.GetFingerprint(m)
        fps.append(fp)
        bad.append(charged_carbon(m))
        if want_bits:
            v = np.zeros(2048, dtype=np.float32)
            for b in fp.GetOnBits():
                v[b] = 1.0
            bits.append(v)
    return (fps, (np.array(bits, dtype=np.float32) if want_bits else None),
            np.array(bad, dtype=bool))


def nn_to(query, ref):
    return np.array([max(DataStructs.BulkTanimotoSimilarity(f, ref))
                     for f in query], dtype=float)


def embed(X, n_neighbors=25, min_dist=0.15, seed=0):
    """2D embedding: UMAP, falling back to PCA. Returns (Z, xlabel, ylabel).

    Jaccard is the metric that matches the data: these are binary fingerprint
    bit vectors, and Tanimoto on binary vectors IS Jaccard, so the embedding
    uses the same notion of similarity as every number in panels A and B.
    Euclidean on raw bits would not.

    n_neighbors is deliberately high. This sample is diverse enough that the
    default of 15 chases local noise and manufactures islands; 25 keeps more
    global structure, which is the only thing worth reading off this panel.
    """
    try:
        import umap
        red = umap.UMAP(n_components=2, metric="jaccard",
                        n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=seed)
        return red.fit_transform(X), "UMAP-1", "UMAP-2"
    except ImportError:
        print("[fig15] umap-learn not installed; falling back to PCA. "
              "conda env update -f environment.yml")
        # PCA by SVD, for an environment without umap-learn. It is linear and
        # weaker, but it states its own weakness: the variance explained goes
        # on the axis labels, and on this data that reads 3.4% and 2.1%. UMAP
        # offers no such number, which is exactly why panel C is labelled
        # orientation-only whichever path produced it.
        # float64: the bit matrix arrives as float32 and a 2048-column matmul
        # in float32 raises divide-by-zero/overflow warnings and can return
        # non-finite coordinates, which would silently corrupt the panel.
        Xc = X.astype(np.float64)
        Xc = Xc - Xc.mean(0)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = S ** 2 / max((S ** 2).sum(), 1e-12)
        # NumPy 2.x raises spurious divide-by-zero / overflow flags out of the
        # BLAS matmul on a sparse binary matrix; the result is finite and
        # correct, which the assertion below actually checks. Suppressing the
        # flags is safe, dropping the check would not be.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            Z = Xc @ Vt[:2].T
        if not np.all(np.isfinite(Z)):
            raise RuntimeError("PCA produced non-finite coordinates")
        return (Z, f"PC1 ({100*var[0]:.1f}% var)",
                f"PC2 ({100*var[1]:.1f}% var)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="osim")
    ap.add_argument("--dump_seed", type=int, default=0)
    ap.add_argument("--null_seed", type=int, default=42)
    ap.add_argument("--dumps_root", default=fs.rel("results", "dumps"))
    ap.add_argument("--geom", default=fs.rel("reference", "geom_drugs_smiles.txt"))
    ap.add_argument("--runs", default="", help="space-separated run names; "
                    "default picks one run per architecture")
    ap.add_argument("--archs", default="base,hidden,tempgain",
                    help="architectures to include")
    ap.add_argument("--objectives", default="db,rtb",
                    help="objectives to include; one run per (arch, objective)")
    ap.add_argument("--beta", default="10",
                    help="preferred beta when several runs of an architecture "
                         "exist; falls back to whatever that architecture has")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/fig15_chemspace.pdf")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root, fam, k = args.dumps_root, args.family, args.dump_seed

    prior_p = os.path.join(root, "_base", fam, f"seed{k}", "base_smiles.txt")
    null_p = os.path.join(root, "_base", fam, f"seed{args.null_seed}",
                          "base_smiles.txt")
    fs.need(prior_p, how="bash scripts/04_dump_guides.sh")
    fs.need(args.geom, how="python data_smiles.py")

    if args.runs:
        runs = args.runs.split()
    else:
        found = sorted(glob.glob(os.path.join(
            root, f"sweep-{fam}-*", f"seed{k}", "guided_smiles.txt")))
        names = [os.path.basename(os.path.dirname(os.path.dirname(p)))
                 for p in found]
        # ONE RUN PER ARCHITECTURE, at the requested beta where it exists.
        # Striding through the sorted list instead silently drops whichever
        # architecture sorts last, which is tempgain: "base" and "hidden" fill
        # the picks and the comparison loses the arm it was meant to show.
        # Canonical run names only. One-off runs live in the same tree
        # ("sweep-osim-hidden-db-replay_off-b10-s0-smoke" is a smoke test), and
        # a suffixed name sorts BEFORE the real one because "-" precedes "/" in
        # the path comparison, so plain sorting hands you the throwaway run.
        canon = re.compile(
            r"^sweep-[a-z0-9]+-[a-z]+-(db|rtb|revkl|fwdkl)"
            r"-replay_(on|off)-b\d+(-s\d+)?$")
        names = [n for n in names if canon.match(n)]

        runs = []
        for arch in args.archs.split(","):
            for obj in args.objectives.split(","):
                # (architecture, objective) both matter: db and rtb optimise
                # different things, and picking the first candidate per
                # architecture silently returns db every time because it sorts
                # before rtb.
                cands = [n for n in names
                         if f"-{arch}-" in n and f"-{obj}-" in n]
                if not cands:
                    print(f"[fig15] no {arch}/{obj} dump for {fam} at seed {k}")
                    continue
                pref = [n for n in cands if f"-b{args.beta}-" in n or
                        n.endswith(f"-b{args.beta}")]
                runs.append((pref or cands)[0])
    if not runs:
        fs.die(f"no guided dumps for {fam} at seed {k}",
               how="bash scripts/04_dump_guides.sh")

    print(f"[fig15] family={fam} seed={k} runs={len(runs)} n={args.n}")
    sources = {}
    geom_fps, geom_bits, geom_bad = load_fps(args.geom, args.n, rng, want_bits=True)
    prior_fps, prior_bits, prior_bad = load_fps(prior_p, args.n, rng, want_bits=True)
    if os.path.isfile(null_p):
        sources["prior (2nd draw) = null"] = load_fps(null_p, args.n, rng, True)
    for r in runs:
        p = os.path.join(root, r, f"seed{k}", "guided_smiles.txt")
        if os.path.isfile(p):
            sources[r.replace(f"sweep-{fam}-", "")] = load_fps(p, args.n, rng, True)

    fs.use_paper_style()
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))

    # ---------------- A: nearest neighbour to the prior -------------------
    ax = axes[0]
    for i, (label, (f_, _, _b)) in enumerate(sources.items()):
        d = np.sort(nn_to(f_, prior_fps))
        null = "null" in label
        ax.plot(d, np.linspace(0, 1, len(d)),
                color="k" if null else None, ls="--" if null else "-",
                lw=2.0 if null else 1.3, zorder=5 if null else 2,
                label=f"{label} ({d.mean():.3f})")
    ax.set_xlabel("max Tanimoto to any prior molecule")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("A. Distance to the prior sample", fontsize=9)
    ax.legend(fontsize=5.5, loc="upper left")

    # ---------------- B: nearest neighbour to GEOM ------------------------
    ax = axes[1]
    d = np.sort(nn_to(prior_fps, geom_fps))
    ax.plot(d, np.linspace(0, 1, len(d)), color=fs.REF_COLOUR, lw=2.0,
            label=f"prior ({d.mean():.3f})", zorder=5)
    for label, (f_, _, _b) in sources.items():
        if "null" in label:
            continue
        d = np.sort(nn_to(f_, geom_fps))
        ax.plot(d, np.linspace(0, 1, len(d)), lw=1.3,
                label=f"{label} ({d.mean():.3f})")
    ax.set_xlabel("max Tanimoto to any GEOM molecule")
    ax.set_title("B. Distance to GEOM-Drugs", fontsize=9)
    ax.legend(fontsize=5.5, loc="upper left")

    # ---------------- C: 2D embedding, qualitative only -------------------
    ax = axes[2]
    mats = [geom_bits, prior_bits] + [b for _, b, _x in sources.values()]
    names = ["GEOM", "prior"] + list(sources)
    Z, xl, yl = embed(np.vstack(mats))
    off = 0
    for name, m in zip(names, mats):
        z = Z[off:off + len(m)]; off += len(m)
        if name == "GEOM":
            ax.scatter(z[:, 0], z[:, 1], s=3, c="0.82", lw=0, zorder=1,
                       label="GEOM-Drugs")
        elif name == "prior":
            ax.scatter(z[:, 0], z[:, 1], s=3, c=fs.REF_COLOUR, lw=0, alpha=0.55,
                       zorder=2, label="prior")
        elif "null" not in name:
            ax.scatter(z[:, 0], z[:, 1], s=3, lw=0, alpha=0.55, zorder=3,
                       label=name)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title("C. Embedding (orientation only)", fontsize=9)
    ax.legend(fontsize=5.5, markerscale=2, loc="best")

    # ------------- D: the same embedding, coloured by artifact ------------
    # This is what the islands in panel C are. They are not a region of
    # chemistry any source discovered, they are where rdDetermineBonds failed.
    ax = axes[3]
    flags = np.concatenate([geom_bad, prior_bad] +
                           [b for _, _x, b in sources.values()])
    ax.scatter(Z[~flags, 0], Z[~flags, 1], s=3, c="0.78", lw=0, zorder=1,
               label=f"clean ({100*(~flags).mean():.0f}%)")
    ax.scatter(Z[flags, 0], Z[flags, 1], s=3, c="#C44E52", lw=0, alpha=0.7,
               zorder=2, label=f"charge on carbon ({100*flags.mean():.0f}%)")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title("D. Bond-perception artifacts", fontsize=9)
    ax.legend(fontsize=6, markerscale=2.5, loc="best")

    fig.suptitle(f"{fs.BENCH_TITLE.get(fam, fam)}: guides against the prior and "
                 f"GEOM-Drugs (n={args.n} per source)", fontsize=10)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()

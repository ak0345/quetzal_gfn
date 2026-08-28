#!/usr/bin/env python3
"""
make_fig16_joint_chemspace.py -- every objective family in ONE chemical space,
against GEOM and the frozen prior.

Figure 15 draws one family at a time, which answers "did guiding move this
family off the prior" but cannot answer "do the families end up anywhere
DIFFERENT FROM EACH OTHER". This figure embeds all of them together, so the
question becomes a separability question with an actual answer.

  A  a single joint 2D embedding (UMAP on ECFP4 with the Jaccard metric, which
     is Tanimoto on binary vectors) with GEOM and the prior drawn as a grey
     backdrop and each objective family over them. Orientation only: a
     neighbour embedding's distances and cluster sizes are not quantities.

  B  the quantitative version of the same question. A k-nearest-neighbour
     classifier is asked to name the objective family a molecule came from,
     from its fingerprint alone, scored by cross-validation. Its accuracy is
     shown against two reference lines: the majority-class rate, which is what
     a classifier that has learned nothing still gets, and the accuracy on
     SHUFFLED labels, which is the empirical null for this sample size. Only a
     bar clearly above BOTH means the benchmark families occupy distinguishable
     regions.

  C  the row-normalised confusion matrix behind panel B, which says WHICH
     families get mistaken for each other rather than just how often.

Note what separability would and would not mean. These molecules are all drawn
from the same frozen prior; a family label is the objective its guide was
trained against. Separability therefore measures whether guiding toward
different objectives actually lands in different chemistry -- not whether the
benchmarks themselves are different.

INPUTS
  reference/geom_drugs_smiles.txt
  results/dumps/_base/<family>/seed<k>/base_smiles.txt
  results/dumps/sweep-<family>-*/seed<k>/guided_smiles.txt

USAGE
  python figures/make_fig16_joint_chemspace.py --out out/fig16.pdf
  python figures/make_fig16_joint_chemspace.py --families osim,peri,fexo
"""
import os
import re
import glob
import random
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

CANON = re.compile(
    r"^sweep-[a-z0-9]+-[a-z]+-(db|rtb|revkl|fwdkl)"
    r"-replay_(on|off)-b\d+(-s\d+)?$")


def load_bits(path, n, rng):
    """Fingerprint bit matrix from a SMILES file, subsampled to n."""
    smis = [l.strip().split()[0] for l in open(path) if l.strip()]
    if len(smis) > n:
        smis = rng.sample(smis, n)
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        v = np.zeros(2048, dtype=np.float32)
        for b in _GEN.GetFingerprint(m).GetOnBits():
            v[b] = 1.0
        out.append(v)
    return np.array(out, dtype=np.float32)


def family_runs(root, fam, seed, archs, objectives, beta):
    """One guided dump per (architecture, objective) for a family."""
    found = sorted(glob.glob(os.path.join(
        root, f"sweep-{fam}-*", f"seed{seed}", "guided_smiles.txt")))
    names = [os.path.basename(os.path.dirname(os.path.dirname(p)))
             for p in found]
    names = [n for n in names if CANON.match(n)]
    runs = []
    for arch in archs:
        for obj in objectives:
            cands = [n for n in names if f"-{arch}-" in n and f"-{obj}-" in n]
            if not cands:
                continue
            pref = [n for n in cands
                    if f"-b{beta}-" in n or n.endswith(f"-b{beta}")]
            runs.append((pref or cands)[0])
    return runs


def embed(X, seed=0, n_neighbors=25, min_dist=0.15):
    """Joint 2D embedding. UMAP with Jaccard, falling back to PCA."""
    try:
        import umap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            red = umap.UMAP(n_components=2, metric="jaccard",
                            n_neighbors=n_neighbors, min_dist=min_dist,
                            random_state=seed)
            return red.fit_transform(X), "UMAP-1", "UMAP-2"
    except ImportError:
        print("[fig16] umap-learn not installed; falling back to PCA. "
              "conda env update -f environment.yml")
        Xc = X.astype(np.float64)
        Xc = Xc - Xc.mean(0)
        with np.errstate(all="ignore"):
            _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            Z = Xc @ Vt[:2].T
        var = S ** 2 / max((S ** 2).sum(), 1e-12)
        assert np.isfinite(Z).all(), "non-finite PCA coordinates"
        return Z, f"PC1 ({100*var[0]:.1f}%)", f"PC2 ({100*var[1]:.1f}%)"


def separability(X, y, seed=0, k=15, folds=5):
    """Cross-validated k-NN accuracy on family labels, with its shuffled null.

    Jaccard is the metric the rest of this analysis uses, so the classifier uses
    it too rather than Euclidean on raw bits.
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("[fig16] scikit-learn not installed; skipping panels B and C. "
              "conda env update -f environment.yml")
        return None

    clf = KNeighborsClassifier(n_neighbors=k, metric="jaccard")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = cross_val_predict(clf, X, y, cv=cv)
        acc = float((pred == y).mean())

        rng = np.random.default_rng(seed)
        y_shuf = rng.permutation(y)
        pred_null = cross_val_predict(clf, X, y_shuf, cv=cv)
        null = float((pred_null == y_shuf).mean())

    labels = sorted(set(y))
    cm = confusion_matrix(y, pred, labels=labels).astype(float)
    cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1)
    _, counts = np.unique(y, return_counts=True)
    return {"acc": acc, "null": null, "majority": float(counts.max() / len(y)),
            "labels": labels, "cm": cm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="osim,peri,fexo,zaleplon,nitrogen")
    ap.add_argument("--dump_seed", type=int, default=0)
    ap.add_argument("--dumps_root", default=fs.rel("results", "dumps"))
    ap.add_argument("--geom", default=fs.rel("reference", "geom_drugs_smiles.txt"))
    ap.add_argument("--archs", default="base,hidden,tempgain")
    ap.add_argument("--objectives", default="db,rtb")
    ap.add_argument("--beta", default="10")
    ap.add_argument("--n_per_family", type=int, default=600)
    ap.add_argument("--n_backdrop", type=int, default=1200)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    fs.add_arg_common(ap, "out/fig16_joint_chemspace.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    rng = random.Random(args.seed)
    root, k = args.dumps_root, args.dump_seed
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    fs.need(args.geom, how="python data_smiles.py")

    # ---- backdrop: GEOM and the frozen prior ----
    blocks, labels = [], []
    geom = load_bits(args.geom, args.n_backdrop, rng)
    blocks.append(geom); labels += ["GEOM"] * len(geom)

    # the prior is the same frozen model whichever family's directory it was
    # dumped under, so one family's base dump represents it
    prior = None
    for fam in families:
        p = os.path.join(root, "_base", fam, f"seed{k}", "base_smiles.txt")
        if os.path.exists(p):
            prior = load_bits(p, args.n_backdrop, rng)
            print(f"[fig16] prior backdrop from {fam} seed {k}")
            break
    if prior is None:
        fs.die("no base dump for any requested family",
               how="bash scripts/04_dump_guides.sh")
    blocks.append(prior); labels += ["prior"] * len(prior)

    # ---- one block of guided molecules per family ----
    per_family = {}
    for fam in families:
        runs = family_runs(root, fam, k, args.archs.split(","),
                           args.objectives.split(","), args.beta)
        if not runs:
            print(f"[fig16] no guided dumps for {fam} at seed {k}; skipped")
            continue
        smis = []
        for r in runs:
            p = os.path.join(root, r, f"seed{k}", "guided_smiles.txt")
            smis += [l.strip().split()[0] for l in open(p) if l.strip()]
        if len(smis) > args.n_per_family:
            smis = rng.sample(smis, args.n_per_family)
        tmp = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                           f".{fam}_joint.smi")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "w") as fh:
            fh.write("\n".join(smis))
        X = load_bits(tmp, args.n_per_family, rng)
        os.remove(tmp)
        if not len(X):
            continue
        per_family[fam] = X
        blocks.append(X); labels += [fam] * len(X)
        print(f"[fig16] {fam}: {len(X)} molecules from {len(runs)} run(s)")

    if len(per_family) < 2:
        fs.die("need at least two families to ask about separability",
               how="bash scripts/04_dump_guides.sh")

    X = np.vstack(blocks)
    y = np.array(labels)
    print(f"[fig16] embedding {X.shape[0]} molecules jointly")

    Z, xlab, ylab = embed(X, seed=args.seed)

    sep = separability(np.vstack(list(per_family.values())),
                       np.array(sum(([f] * len(v) for f, v in per_family.items()),
                                    [])),
                       seed=args.seed, k=args.knn)

    ncols = 3 if sep else 1
    fig, axes = plt.subplots(1, ncols, figsize=(args.width * 1.5, 3.6),
                             gridspec_kw={"wspace": 0.34,
                                          "width_ratios": [1.35, 1, 1][:ncols]},
                             squeeze=False)
    axes = axes[0]

    # ---- A: joint embedding ----
    axA = axes[0]
    for name in ("GEOM", "prior"):
        m = y == name
        axA.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.5, linewidths=0,
                    color=fs.BACKDROP_COLOURS[name], label=name, zorder=1)
    for fam in per_family:
        m = y == fam
        axA.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.75, linewidths=0,
                    color=fs.BENCH_COLOURS.get(fam, "0.3"), label=fam, zorder=3)
    axA.set_xlabel(xlab)
    axA.set_ylabel(ylab)
    axA.set_title("Joint chemical space")
    axA.legend(fontsize=6, markerscale=2.2, loc="best", frameon=False)

    if sep:
        # ---- B: is the family label recoverable at all? ----
        axB = axes[1]
        axB.bar([0], [sep["acc"]], width=0.45, color=fs.GUIDE_COLOURS["hidden"])
        # The two references land on top of each other exactly when the answer
        # is "not separable", which is the interesting case, so they are pinned
        # to opposite corners rather than both sitting on their own line.
        axB.axhline(sep["majority"], color="0.55", ls=":", lw=1.2)
        axB.text(-0.48, sep["majority"] + 0.03,
                 f"majority class {sep['majority']:.3f}",
                 fontsize=6, color="0.4", va="bottom", ha="left")
        axB.axhline(sep["null"], color=fs.REF_COLOUR, ls="--", lw=1.2)
        axB.text(0.48, sep["null"] - 0.03,
                 f"shuffled-label null {sep['null']:.3f}",
                 fontsize=6, color=fs.REF_COLOUR, va="top", ha="right")
        axB.annotate(f"{sep['acc']:.3f}", (0, sep["acc"]), fontsize=7,
                     xytext=(0, 4), textcoords="offset points", ha="center")
        axB.set_xlim(-0.5, 0.5)
        axB.set_xticks([0])
        axB.set_xticklabels([f"{args.knn}-NN\n(Jaccard)"], fontsize=7)
        axB.set_ylim(0, 1.05)
        axB.set_ylabel("cross-validated accuracy")
        axB.set_title("Recovering the objective")

        # ---- C: which families are confused ----
        axC = axes[2]
        im = axC.imshow(sep["cm"], cmap="Blues", vmin=0, vmax=1)
        axC.set_xticks(range(len(sep["labels"])))
        axC.set_xticklabels(sep["labels"], rotation=35, ha="right", fontsize=6.5)
        axC.set_yticks(range(len(sep["labels"])))
        axC.set_yticklabels(sep["labels"], fontsize=6.5)
        axC.set_xlabel("predicted")
        axC.set_ylabel("true")
        axC.set_title("Confusion (row-normalised)")
        axC.grid(False)
        for i in range(len(sep["labels"])):
            for j in range(len(sep["labels"])):
                v = sep["cm"][i, j]
                axC.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=5.5, color="white" if v > 0.5 else "0.25")
        fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)

        print(f"[fig16] {args.knn}-NN accuracy {sep['acc']:.3f} | "
              f"majority class {sep['majority']:.3f} | "
              f"shuffled-label null {sep['null']:.3f}")
        margin = sep["acc"] - max(sep["majority"], sep["null"])
        verdict = ("separable: accuracy is well above both references"
                   if margin > 0.10 else
                   "NOT clearly separable: accuracy sits on its references")
        print(f"[fig16] margin over the stronger reference {margin:+.3f} -- {verdict}")

    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()

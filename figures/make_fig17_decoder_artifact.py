#!/usr/bin/env python3
"""
make_fig17_decoder_artifact.py -- the island in the joint embedding is a
decoder artifact, not a chemotype.

The joint chemical space (Figure 16) contains one small, sharply separated
island. It is tempting to read a well-separated cluster as chemistry the guides
found. It is not. Every molecule in it carries a formal charge on a carbon atom,
which is the signature of a failure in the 3D-to-SMILES step rather than of
anything the model learned.

`rdDetermineBonds` infers bond orders from coordinates alone. When no assignment
satisfies every valence it balances the books with alternating formal charges,
typically around an aromatic ring. The result sanitises cleanly and is chemically
absurd, so it survives every downstream filter and lands in the reported sets.

  A  the joint embedding, coloured by whether the molecule carries a charged
     carbon. The island is the artifact population, exactly.
  B  the rate per source. GEOM-Drugs, which is real SMILES and never passes
     through the decoder, contains none. The frozen prior and every guided
     family sit together well above zero, so this is a property of the decoder
     and not of guiding.
  C  what it costs. The objective-recovery test of Figure 16, re-run on the
     clean molecules alone. If the artifact were carrying the (absent) signal,
     removing it would change the answer.

INPUTS   the same dumps Figure 16 reads.

USAGE
  python figures/make_fig17_decoder_artifact.py --out out/fig17.pdf
"""
import os
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs
import make_fig16_joint_chemspace as f16

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def charged_carbon(mol):
    """A formal charge on carbon: the bond-perception signature."""
    return any(a.GetAtomicNum() == 6 and a.GetFormalCharge() != 0
               for a in mol.GetAtoms())


def load_with_flags(path, n, rng):
    """Bit matrix plus a charged-carbon flag per molecule."""
    smis = [l.strip().split()[0] for l in open(path) if l.strip()]
    if len(smis) > n:
        smis = rng.sample(smis, n)
    bits, flags = [], []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        v = np.zeros(2048, dtype=np.float32)
        for b in f16._GEN.GetFingerprint(m).GetOnBits():
            v[b] = 1.0
        bits.append(v)
        flags.append(charged_carbon(m))
    return np.array(bits, dtype=np.float32), np.array(flags, dtype=bool)


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
    ap.add_argument("--emit_json", default=None,
                    help="also write the measurements here, for make_tables.py")
    fs.add_arg_common(ap, "out/fig17_decoder_artifact.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    import random
    rng = random.Random(args.seed)
    root, k = args.dumps_root, args.dump_seed
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    fs.need(args.geom, how="python data_smiles.py")

    blocks, labels, flags = [], [], []
    Xg, fg = load_with_flags(args.geom, args.n_backdrop, rng)
    blocks.append(Xg); labels += ["GEOM"] * len(Xg); flags.append(fg)

    prior_src = None
    for fam in families:
        p = os.path.join(root, "_base", fam, f"seed{k}", "base_smiles.txt")
        if os.path.exists(p):
            Xp, fp = load_with_flags(p, args.n_backdrop, rng)
            blocks.append(Xp); labels += ["prior"] * len(Xp); flags.append(fp)
            prior_src = fam
            break
    if prior_src is None:
        fs.die("no base dump for any requested family",
               how="bash scripts/04_dump_guides.sh")

    per_family = {}
    for fam in families:
        runs = f16.family_runs(root, fam, k, args.archs.split(","),
                               args.objectives.split(","), args.beta)
        if not runs:
            print(f"[fig17] no guided dumps for {fam}; skipped")
            continue
        smis = []
        for r in runs:
            p = os.path.join(root, r, f"seed{k}", "guided_smiles.txt")
            smis += [l.strip().split()[0] for l in open(p) if l.strip()]
        if len(smis) > args.n_per_family:
            smis = rng.sample(smis, args.n_per_family)
        tmp = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                           f".{fam}_art.smi")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "w") as fh:
            fh.write("\n".join(smis))
        Xf, ff = load_with_flags(tmp, args.n_per_family, rng)
        os.remove(tmp)
        if not len(Xf):
            continue
        per_family[fam] = (Xf, ff)
        blocks.append(Xf); labels += [fam] * len(Xf); flags.append(ff)

    X = np.vstack(blocks)
    y = np.array(labels)
    bad = np.concatenate(flags)
    print(f"[fig17] {X.shape[0]} molecules, {bad.sum()} with a charged carbon")

    Z, xlab, ylab = f16.embed(X, seed=args.seed)

    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(args.width * 1.5, 3.6),
        gridspec_kw={"wspace": 0.34, "width_ratios": [1.35, 1.1, 1]})

    # ---- A: the embedding, coloured by the artifact flag ----
    axA.scatter(Z[~bad, 0], Z[~bad, 1], s=5, alpha=0.45, linewidths=0,
                color="#BDBDBD", label=f"clean ({100*(~bad).mean():.0f}%)",
                zorder=1)
    axA.scatter(Z[bad, 0], Z[bad, 1], s=7, alpha=0.8, linewidths=0,
                color=fs.REF_COLOUR,
                label=f"charged carbon ({100*bad.mean():.0f}%)", zorder=3)
    axA.set_xlabel(xlab)
    axA.set_ylabel(ylab)
    axA.set_title("The island is the artifact")
    axA.legend(fontsize=6.5, markerscale=2.2, loc="best", frameon=False)

    # ---- B: rate per source ----
    order = ["GEOM", "prior"] + [f for f in per_family]
    rates = [bad[y == s].mean() for s in order]
    cols = [fs.BACKDROP_COLOURS.get(s) or fs.BENCH_COLOURS.get(s, "0.4")
            for s in order]
    xb = np.arange(len(order))
    axB.bar(xb, rates, color=cols, width=0.66)
    for xi, r in zip(xb, rates):
        axB.annotate(f"{100*r:.1f}", (xi, r), fontsize=6, ha="center",
                     xytext=(0, 3), textcoords="offset points")
    axB.set_xticks(xb)
    axB.set_xticklabels(order, rotation=30, ha="right", fontsize=6.5)
    axB.set_ylabel("fraction with a charged carbon")
    axB.set_ylim(0, max(rates) * 1.35 if max(rates) else 1)
    axB.set_title("Rate by source")

    for s, r in zip(order, rates):
        print(f"[fig17] {s:>9}: {100*r:5.1f}% charged carbon  (n={int((y==s).sum())})")

    # ---- C: does removing it change the separability answer? ----
    fam_X = np.vstack([v[0] for v in per_family.values()])
    fam_y = np.array(sum(([f] * len(v[0]) for f, v in per_family.items()), []))
    fam_bad = np.concatenate([v[1] for v in per_family.values()])

    sep_all = f16.separability(fam_X, fam_y, seed=args.seed, k=args.knn)
    sep_cln = f16.separability(fam_X[~fam_bad], fam_y[~fam_bad],
                               seed=args.seed, k=args.knn)
    if sep_all and sep_cln:
        xs = np.arange(2)
        axC.bar(xs, [sep_all["acc"], sep_cln["acc"]], width=0.55,
                color=[fs.GUIDE_COLOURS["hidden"], fs.GUIDE_COLOURS["tempgain"]])
        axC.axhline(sep_all["majority"], color="0.55", ls=":", lw=1.2)
        # the bars sit on the majority line when the answer is "not separable",
        # so the reference is named up in the empty half of the panel rather
        # than on its own line, where it lands on the bar annotations
        axC.text(0.5, 0.97, f"dotted: majority class {sep_all['majority']:.3f}",
                 transform=axC.transAxes, fontsize=6, color="0.4",
                 va="top", ha="center")
        for xi, v in zip(xs, [sep_all["acc"], sep_cln["acc"]]):
            axC.annotate(f"{v:.3f}", (xi, v), fontsize=7, ha="center",
                         xytext=(0, 4), textcoords="offset points")
        axC.set_xticks(xs)
        axC.set_xticklabels(["all\nmolecules", "clean\nonly"], fontsize=7)
        axC.set_ylim(0, 1.05)
        axC.set_ylabel(f"{args.knn}-NN accuracy on objective")
        axC.set_title("Removing it changes nothing")
        print(f"[fig17] objective recovery: all {sep_all['acc']:.3f}, "
              f"clean-only {sep_cln['acc']:.3f}, "
              f"majority {sep_all['majority']:.3f}")
    else:
        axC.set_axis_off()

    geom_rate = bad[y == "GEOM"].mean()
    gen_rate = bad[y != "GEOM"].mean()
    print(f"[fig17] GEOM {100*geom_rate:.2f}% vs generated {100*gen_rate:.2f}% -- "
          f"the artifact is produced by the 3D-to-SMILES step, not present in "
          f"the corpus")

    if args.emit_json:
        import json
        payload = {
            "artifact": {s: float(r) for s, r in zip(order, rates)},
            "separability": ({
                "k": args.knn,
                "acc": sep_all["acc"],
                "majority": sep_all["majority"],
                "null": sep_all["null"],
                "acc_clean": sep_cln["acc"] if sep_cln else None,
            } if sep_all else None),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.emit_json)),
                    exist_ok=True)
        with open(args.emit_json, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"wrote {args.emit_json}")

    fig.suptitle("A decoder artifact, present in every generated set",
                 fontsize=9.5)
    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()

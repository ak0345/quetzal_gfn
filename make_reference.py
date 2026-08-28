#!/usr/bin/env python3
"""
make_reference.py -- the matched-budget best-of-dataset baseline for one
benchmark, without needing any fine-tuning runs.

harvest_eval.py already computes this block and stores it as `_reference` in the
harvest JSON, but only as a side effect of scoring a set of recorded runs. A
benchmark with no fine-tuning runs therefore has no reference line, even though
the quantity depends on nothing but the dataset and the objective.

This script computes exactly that block on its own, using the same convention:
canonicalise the reference SMILES, take the first --limit of them, score each
under the benchmark, and report top-1/10/100 means and the GuacaMol composite.

OUTPUT
  results/oracle_gfn_mols/_results/_reference_<bench>_<limit>.json

USAGE
  python make_reference.py --bench hard_osimertinib
  python make_reference.py --bench hard_osimertinib --limit 10000
"""
import os
import sys
import json
import argparse

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from harvest_eval import CachedScorer, get_scorer

try:
    import harvest_analysis as HA
except ImportError:
    HA = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True,
                    help="benchmark function name, e.g. hard_osimertinib")
    ap.add_argument("--ref_smiles", default="reference/geom_drugs_smiles.txt")
    ap.add_argument("--limit", type=int, default=10000,
                    help="matched budget: how many dataset molecules to score")
    ap.add_argument("--analysis_topn", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.ref_smiles):
        print(f"[fatal] reference SMILES not found: {args.ref_smiles}\n"
              f"        produce it with: python data_smiles.py", file=sys.stderr)
        raise SystemExit(2)

    # canonicalise and de-duplicate exactly as harvest_eval does, so the number
    # this writes is the same one a full harvest would have recorded
    ref_set = set()
    with open(args.ref_smiles) as f:
        for line in f:
            s = line.strip().split()[0] if line.strip() else ""
            m = Chem.MolFromSmiles(s) if s else None
            if m is not None:
                ref_set.add(Chem.MolToSmiles(m))
    print(f"[ref] {len(ref_set)} unique reference molecules from {args.ref_smiles}")

    ref_list = list(ref_set)[:args.limit]
    scorer = CachedScorer(get_scorer(args.bench))
    scores = [scorer(s) for s in ref_list]

    info = {"name": os.path.basename(args.ref_smiles), "n": len(ref_list),
            "benchmark": args.bench, "budget": args.limit}
    for k in (1, 10, 100):
        info[f"top{k}"] = float(np.mean(sorted(scores, reverse=True)[:k]))
    info["guacamol_score"] = float(np.mean(
        [info["top1"], info["top10"], info["top100"]]))

    if HA is not None:
        best = [s for _, s in sorted(zip(scores, ref_list),
                                     key=lambda t: t[0], reverse=True)][:args.analysis_topn]
        info["quality_pass_rate"] = HA.quality_report(best).get("pass_rate")
        info["descriptors"] = HA.descriptor_table(best)

    print(f"[ref] best-of-dataset: top1={info['top1']:.4f} "
          f"top10={info['top10']:.4f} top100={info['top100']:.4f} "
          f"composite={info['guacamol_score']:.4f}")

    out = args.out or os.path.join(
        "results", "oracle_gfn_mols", "_results",
        f"_reference_{args.bench}_{args.limit}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(info, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

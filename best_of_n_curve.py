#!/usr/bin/env python3
"""
Best-of-N curves: does the frozen prior's best-of-N keep climbing past the
dataset baseline, or does it plateau?

WHY IT MATTERS. Every configuration landing near GEOM's best-of-10k is
compatible with two different states of the world:

  (a) the high-reward region is ABSENT from the prior. No amount of sampling
      finds it, and steering cannot reach what the prior does not contain;
  (b) the high-reward region is PRESENT but rare. The prior would find it given
      enough draws, and steering failed to concentrate mass on it.

Both produce the same flat benchmark table, and they imply opposite conclusions:
(a) says change the prior, (b) says the steering objective is not doing its job.
Best-of-N separates them. If the prior's curve keeps rising through the dataset
line as N grows, the honest claim is about search rather than distributional
support.

This is also where the matched-budget dataset baseline itself comes from: the
GEOM-Drugs curve read at n = 10,000.

WHAT IT COMPUTES. Given a pool of scored molecules, the expected best-of-n and
top-k mean for n on a log grid, estimated by resampling n draws from the pool
without replacement, B times. That estimates what a fresh n-sample run would
have produced, with a percentile interval, rather than reporting a single lucky
draw.

The estimate is only trustworthy for n well below the pool size. At n = N every
resample returns the whole pool, so the interval collapses to zero width while
the point estimate still carries only one sample's worth of information.

MODES
  --generate      roll N molecules from the frozen Quetzal prior and score them
  --smiles_file   score an existing SMILES list (the dataset baseline, or any
                  previously dumped sample) -- can be passed more than once

Scoring reuses harvest_eval.get_scorer, so the scorer is the same object the
training runs and the harvest used.

EXAMPLES
--------
# prior curve to 100k samples, with GEOM as the dataset reference
nohup python best_of_n_curve.py --generate 20000 --quetzal_ckpt geom.ckpt --bench hard_osimertinib --smiles_file GEOM_smiles/geom_drugs_smiles.txt --out results/bon_osim.json --plot results/bon_osim.png > gen_prior.log 2>&1 &

# reuse a dump instead of regenerating
python best_of_n_curve.py --smiles_file prior_100k.smi --smiles_file geom.smi \\
    --bench hard_perindopril --out results/bon_peri.json
"""
import os
import json
import math
import argparse

import numpy as np


# ----------------------------- sampling the prior -----------------------------

def generate_from_prior(ckpt, n, bsz=250, max_len=192, diff_steps=18,
                        device=None, train_module="train.py", use_ema=True,
                        mask_atoms=None, temp=1.0, out_smi=None):
    """Roll n molecules from the FROZEN prior and return their SMILES.

    Uses the same call sequence as LitRTBFineTune.sample with grad off and no
    guide: encode1 -> proj_logits -> sample atom -> encode2 -> sample_coord.
    """
    import torch
    import torch.nn.functional as F
    from rtb_finetune import load_prior_and_policy, build_atom_mask, \
        mol_to_smiles, FTConfig
    from chem import Molecule, GEN, STOP

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = FTConfig(quetzal_ckpt=ckpt, train_module=train_module,
                   use_ema_prior=use_ema, max_len=max_len,
                   diff_steps=diff_steps, mask_atoms=mask_atoms)
    frozen, _policy, _pc = load_prior_and_policy(cfg)
    frozen = frozen.to(device).eval()
    mask = build_atom_mask(mask_atoms, device)
    NEG = -1e9

    smiles, done = [], 0
    fh = open(out_smi, "a") if out_smi else None
    with torch.no_grad():
        while done < n:
            b = min(bsz, n - done)
            atoms = torch.full((b, 1), GEN, dtype=torch.long, device=device)
            coords = torch.zeros(b, 1, 3, device=device)
            stop_mask = torch.zeros(b, dtype=torch.bool, device=device)
            for _ in range(max_len):
                idx = torch.arange(atoms.shape[1], device=device).expand(b, -1)
                seq = frozen.encode1(idx, atoms, coords)
                logits = frozen.proj_logits(seq[:, -1, :]).float().masked_fill(~mask, NEG)
                p = F.softmax(logits / temp, dim=-1)
                nxt = torch.multinomial(p, num_samples=1)
                stop_mask = stop_mask | (nxt.squeeze(-1) == STOP)
                if stop_mask.all():
                    break
                atoms = torch.cat([atoms, nxt], dim=1)
                z = frozen.encode2(atoms[:, 1:], seq)[:, -1, :]
                nc, _ = frozen.sample_coord(z, device=device, num_steps=diff_steps)
                coords = torch.cat([coords, nc.view(b, 1, 3)], dim=1)
            mols = Molecule(atoms=atoms[:, 1:], coords=coords[:, 1:]).to("cpu").unbatch()
            for m in mols:
                s = mol_to_smiles(m)
                # keep the None entries: a molecule that fails 3D->SMILES still
                # consumed a draw, and dropping it would quietly inflate the
                # curve by pretending the prior only ever emits parseable output
                smiles.append(s)
                if fh and s:
                    fh.write(s + "\n")
            done += b
            print(f"[gen] {done}/{n}", flush=True)
    if fh:
        fh.close()
    return smiles


def read_smiles(path):
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip().split()[0] if line.strip() else ""
            if s:
                out.append(s)
    return out


# ----------------------------- the curve -----------------------------

def bon_curve(scores, grid, k=1, n_boot=200, pcts=(2.5, 97.5), seed=0):
    """Expected top-k mean of n draws from a pool of scored molecules.

    Sampling WITHOUT replacement, because a real n-molecule run draws n distinct
    trajectories; with replacement would understate the best-of-n slightly at
    large n. Failed molecules must already be in `scores` as 0.0 -- they consume
    a draw.
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, dtype=float)
    N = len(s)
    rows = []
    for n in grid:
        if n > N:
            break
        if n >= N:                       # degenerate: every resample is the pool
            vals = np.array([np.mean(np.sort(s)[::-1][:k])])
        else:
            vals = np.empty(n_boot)
            for b in range(n_boot):
                draw = rng.choice(s, size=n, replace=False)
                vals[b] = np.mean(np.sort(draw)[::-1][:k])
        rows.append({
            "n": int(n), "mean": float(vals.mean()),
            "lo": float(np.percentile(vals, pcts[0])),
            "hi": float(np.percentile(vals, pcts[1])),
            "n_boot": int(len(vals)),
            # at n close to N the resamples overlap almost completely and the
            # interval is optimistically narrow; flag rather than hide it
            "saturated": bool(n > 0.5 * N),
        })
    return rows


def crossing_n(rows, level):
    """Smallest n whose mean curve reaches `level` (None if it never does)."""
    for r in rows:
        if r["mean"] >= level:
            return r["n"]
    return None


def log_grid(n_max, per_decade=6, n_min=10):
    lo, hi = math.log10(n_min), math.log10(n_max)
    pts = np.unique(np.round(np.logspace(lo, hi, int((hi - lo) * per_decade) + 1)))
    return [int(x) for x in pts if x >= n_min]


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="hard_osimertinib")
    ap.add_argument("--generate", type=int, default=0,
                    help="sample this many molecules from the frozen prior")
    ap.add_argument("--quetzal_ckpt", default="geom.ckpt")
    ap.add_argument("--train_module", default="train.py")
    ap.add_argument("--mask_atoms", default=None)
    ap.add_argument("--diff_steps", type=int, default=18)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--gen_bsz", type=int, default=250)
    ap.add_argument("--save_generated", default=None,
                    help="write the generated SMILES here so the (expensive) "
                         "rollout is reusable")
    ap.add_argument("--smiles_file", action="append", default=[],
                    help="score an existing SMILES list; repeatable. The first "
                         "one is treated as the dataset reference for the "
                         "crossing-point report")
    ap.add_argument("--label", action="append", default=[],
                    help="display name per --smiles_file, in the same order")
    ap.add_argument("--topk", type=int, default=1,
                    help="k for the top-k mean (1 = plain best-of-n)")
    ap.add_argument("--also_top10", action="store_true",
                    help="additionally compute the k=10 curve")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--baseline_n", type=int, default=10000,
                    help="the budget your paper reports; the dataset's value at "
                         "this n is the line the prior has to cross")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    from harvest_eval import get_scorer, CachedScorer, canonical
    scorer = CachedScorer(get_scorer(args.bench))

    pools = {}          # name -> list of scores (0.0 for unparseable)

    if args.generate:
        smis = generate_from_prior(
            args.quetzal_ckpt, args.generate, bsz=args.gen_bsz,
            max_len=args.max_len, diff_steps=args.diff_steps,
            train_module=args.train_module, mask_atoms=args.mask_atoms,
            out_smi=args.save_generated)
        pools["frozen prior"] = [scorer(canonical(s)) if canonical(s) else 0.0
                                 for s in smis]
        print(f"[prior] {len(smis)} sampled, "
              f"{sum(1 for s in smis if s)} parseable")

    labels = list(args.label) + [None] * len(args.smiles_file)
    for j, path in enumerate(args.smiles_file):
        name = labels[j] or os.path.basename(path)
        smis = read_smiles(path)
        pools[name] = [scorer(canonical(s)) if canonical(s) else 0.0 for s in smis]
        print(f"[{name}] {len(smis)} molecules scored")

    if not pools:
        ap.error("nothing to score: pass --generate and/or --smiles_file")

    report = {"benchmark": args.bench, "baseline_n": args.baseline_n,
              "curves": {}, "pool_sizes": {k: len(v) for k, v in pools.items()}}
    ks = [args.topk] + ([10] if args.also_top10 and args.topk != 10 else [])

    for name, scores in pools.items():
        n_max = len(scores)
        grid = log_grid(n_max)
        for k in ks:
            rows = bon_curve(scores, grid, k=k, n_boot=args.n_boot, seed=args.seed)
            report["curves"][f"{name}|top{k}"] = rows
            at = next((r for r in rows if r["n"] >= args.baseline_n), None)
            print(f"[{name}] top{k} @n={args.baseline_n}: "
                  + (f"{at['mean']:.4f} [{at['lo']:.4f}, {at['hi']:.4f}]"
                     if at else "pool too small"))

    # ---- the actual question: does the prior overtake the dataset baseline? ----
    ref_name = (args.label[0] if args.label else
                (os.path.basename(args.smiles_file[0]) if args.smiles_file else None))
    if ref_name and "frozen prior" in pools:
        for k in ks:
            ref_rows = report["curves"].get(f"{ref_name}|top{k}", [])
            pri_rows = report["curves"].get(f"frozen prior|top{k}", [])
            ref_at = next((r for r in ref_rows if r["n"] >= args.baseline_n), None)
            if not ref_at or not pri_rows:
                continue
            level = ref_at["mean"]
            n_cross = crossing_n(pri_rows, level)
            last = pri_rows[-1]
            report[f"verdict_top{k}"] = {
                "dataset_level": level, "prior_crosses_at_n": n_cross,
                "prior_at_largest_n": last["mean"], "largest_n": last["n"]}
            print(f"\n=== top{k}: dataset best-of-{args.baseline_n} = {level:.4f} ===")
            if n_cross:
                print(f"  the prior reaches it at n={n_cross:,} -- the region IS "
                      f"in the prior's support, so the flat benchmark result is "
                      f"a SEARCH failure, not a support failure.")
            else:
                print(f"  the prior does not reach it by n={last['n']:,} "
                      f"(best {last['mean']:.4f}). Consistent with the region "
                      f"being absent or vanishingly rare under the prior.")
            # slope over the last decade says whether it is still climbing
            dec = [r for r in pri_rows if r["n"] >= last["n"] / 10]
            if len(dec) >= 2:
                slope = (dec[-1]["mean"] - dec[0]["mean"])
                report[f"verdict_top{k}"]["gain_over_last_decade"] = float(slope)
                print(f"  gain over the last decade of n: {slope:+.4f} "
                      f"({'still climbing' if slope > 0.005 else 'flat'})")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[write] {args.out}")

    if args.plot:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            for key, rows in report["curves"].items():
                if not rows:
                    continue
                xs = [r["n"] for r in rows]
                ys = [r["mean"] for r in rows]
                line, = ax.plot(xs, ys, "-o", ms=3, label=key)
                ax.fill_between(xs, [r["lo"] for r in rows], [r["hi"] for r in rows],
                                alpha=0.15, color=line.get_color(), linewidth=0)
            ax.axvline(args.baseline_n, ls=":", c="grey", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel("molecules drawn (n)")
            ax.set_ylabel("expected top-k mean score")
            ax.set_title(f"best-of-N scaling ({args.bench})")
            ax.grid(alpha=.3); ax.legend(fontsize=8)
            os.makedirs(os.path.dirname(os.path.abspath(args.plot)) or ".",
                        exist_ok=True)
            fig.tight_layout(); fig.savefig(args.plot, dpi=140); plt.close(fig)
            print(f"[plot] {args.plot}")
        except Exception as e:
            print(f"[plot] failed: {e}")


if __name__ == "__main__":
    main()
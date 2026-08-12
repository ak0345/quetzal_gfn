#!/usr/bin/env python3
"""
final_dump_composed.py -- STANDALONE, SEEDED dump for a COMPOSED sampler
(compose.py / Composer), across operators {linear, product, harmonic}.

For each operator it:
  * runs compose.py's OWN rich reporting (reward histograms, KDE, ternary,
    hypervolume, slope test, per-component summary) by calling its main() with
    the right MultiConfig flags -> writes into <out_dir>/compose_native/;
  * ALSO emits an aggregator-compatible dump_summary.json (same schema as
    final_dump2.py) so composed runs fold into master_table.csv alongside the
    single-guide sweep. The "guided" block = the COMPOSED sampler scored on the
    PRIMARY eval reward (first in eval_rewards, e.g. osim_MPO); "base" = frozen
    Quetzal on the same reward. Adds EDM atom/mol stability + descriptor-vs-GEOM
    (reused from final_dump2) for the composed and base sets.

The composed run's "name" (for the aggregator) is:
    compose-<benchkey>-<operator>-k<K>-b<compose_beta>
so parse_name in aggregate_dumps.py treats it as family=compose.

Usage:
  python final_dump_composed.py \
    --guide_ckpts c0.ckpt,c1.ckpt,c2.ckpt,c3.ckpt \
    --guide_labels c0,c1,c2,c3 --weights 0.25,0.25,0.25,0.25 \
    --train_betas 10,10,10,10 \
    --eval_rewards "guacamol:hard_osimertinib=osim_MPO,gcomp:osimertinib:0=c0,gcomp:osimertinib:1=c1,gcomp:osimertinib:2=c2,gcomp:osimertinib:3=c3" \
    --operators linear,product,harmonic \
    --bench_key osimertinib \
    --n 5000 --seed 0 --ref_smiles geom_ref.smi --dataset geom \
    --out_dir dumps_composed/osim
"""
import os
import json
import argparse

import numpy as np
import numpy as np, scipy
if not hasattr(scipy, "histogram"): scipy.histogram = np.histogram
import torch

# reuse the metric helpers from final_dump2 (stability, descriptors, FCD wrapper)
import final_dump as fd2


OPERATOR_FLAGS = {
    "linear":   ("linear", "poe"),        # product_kind ignored for linear
    "product":  ("product", "poe"),
    "harmonic": ("product", "harmonic"),
}


def build_multiconfig(args, operator, product_kind, out_dir_native, seed):
    """Construct a compose.py MultiConfig for one operator, matching CLI flags."""
    from gflow_multi import MultiConfig
    return MultiConfig(
        quetzal_ckpt=args.quetzal_ckpt,
        train_module=args.train_module,
        dataset=args.dataset,
        diff_steps=args.diff_steps,
        guide_ckpts=args.guide_ckpts,
        guide_labels=args.guide_labels,
        guide_source=args.guide_source,
        eval_rewards=args.eval_rewards,
        operator=operator,
        product_kind=product_kind,
        weights=args.weights,
        train_betas=args.train_betas,
        compose_space=args.compose_space,
        compose_beta=args.compose_beta,
        use_logz=args.use_logz,
        n_samples=args.n,
        chunk=args.chunk,
        sample_temp=args.sample_temp,
        rand_eps=args.rand_eps,
        seed=seed,
        device=args.device,
        out_dir=out_dir_native,
        tag=f"{args.bench_key}_{operator}_{product_kind}",
        make_base=True,          # we want base for the aggregator comparison
        make_singles=args.make_singles,
        fcd_ref_smiles=args.ref_smiles,
        fcd_enabled=False,       # we compute FCD ourselves (guided/base/ref) below
        save_mols=True,
    )


def score_block(mols, primary_logr, dataset, ref_smiles, progress_every=0, tag=""):
    """Build an aggregator-schema per-source block: parse SMILES, top-k on the
    PRIMARY reward (already computed), EDM stability, uniqueness."""
    from gflow import mol_to_rdkit
    from rdkit import Chem
    smiles = []
    for m in mols:
        rd = mol_to_rdkit(m)
        if rd is None:
            continue
        try:
            s = Chem.MolToSmiles(Chem.RemoveHs(rd))
        except Exception:
            continue
        if s:
            smiles.append(s)
    logr = np.asarray(primary_logr, dtype=float)
    atom_stab, mol_stab = fd2.edm_stability_for_mols(
        mols, dataset, progress_every=progress_every, tag=tag)
    return {
        "smiles": smiles,
        "block": {
            "n_generated": len(mols),
            "n_valid_smiles": len(smiles),
            "parse_rate": len(smiles) / max(len(mols), 1),
            "uniqueness": (len(set(smiles)) / len(smiles)) if smiles else 0.0,
            "log_reward_mean": float(logr.mean()) if len(logr) else None,
            "log_reward_top1": float(np.max(logr)) if len(logr) else None,
            "log_reward_top10": float(np.mean(np.sort(logr)[-10:])) if len(logr) >= 10 else None,
            "log_reward_top100": float(np.mean(np.sort(logr)[-100:])) if len(logr) >= 100 else None,
            "atom_stability": atom_stab,
            "mol_stability": mol_stab,
        },
    }


def dump_one_operator(args, operator):
    op, pk = OPERATOR_FLAGS[operator]
    out_dir = os.path.join(args.out_dir, operator, f"seed{args.seed}")
    native_dir = os.path.join(out_dir, "compose_native")
    os.makedirs(native_dir, exist_ok=True)

    if os.path.exists(os.path.join(out_dir, "dump_summary.json")):
        print(f"[skip] {operator} seed{args.seed} (dump_summary.json exists)", flush=True)
        return

    fd2.set_seed(args.seed)
    from gflow_multi import Composer

    cfg = build_multiconfig(args, op, pk, native_dir, args.seed)
    print(f"[compose] operator={operator} ({op}/{pk}) seed={args.seed} "
          f"k={len(cfg.guide_ckpts.split(','))}", flush=True)
    comp = Composer(cfg)

    # ---- generate composed + base ----
    if args.progress:
        print(f"[compose] sampling composed x {args.n} ...", flush=True)
    composed_mols = comp.sample_composed(args.n)
    if args.progress:
        print(f"[compose] sampling base x {args.n} ...", flush=True)
    base_mols = comp.sample_base(args.n)

    # ---- score both on ALL eval rewards; primary = first (e.g. osim_MPO) ----
    reward_names = [n for n, _ in comp.eval_rewards]
    primary = reward_names[0]
    composed_scored = comp.score(composed_mols)
    base_scored = comp.score(base_mols)

    pe = args.progress_every if args.progress else 0
    g = score_block(composed_mols, composed_scored[primary], args.dataset,
                    args.ref_smiles, pe, " composed")
    b = score_block(base_mols, base_scored[primary], args.dataset,
                    args.ref_smiles, pe, " base")

    # ---- FCD + descriptors vs GEOM (reuse final_dump2 helpers) ----
    ref_smiles = fd2.load_ref_smiles(args.ref_smiles, limit=args.ref_limit)
    fcd = {}
    if not args.no_fcd:
        try:
            from gflow import _get_fcd
            fcd_fn = _get_fcd()
            if fcd_fn is not None:
                gs, bs = g["smiles"], b["smiles"]
                if len(gs) > 10 and len(bs) > 10:
                    fcd["guided_vs_base"] = fcd_fn(gs, bs)
                if ref_smiles and len(ref_smiles) > 10:
                    if len(gs) > 10:
                        fcd["guided_vs_ref"] = fcd_fn(gs, ref_smiles)
                    if len(bs) > 10:
                        fcd["base_vs_ref"] = fcd_fn(bs, ref_smiles)
        except Exception as e:
            print(f"[fcd] failed: {e}", flush=True)

    wass = {}
    try:
        g_desc = fd2.descriptors_for_smiles(g["smiles"])
        b_desc = fd2.descriptors_for_smiles(b["smiles"])
        r_desc = fd2.descriptors_for_smiles(ref_smiles) if ref_smiles else None
        if r_desc is not None:
            for kdesc in fd2._DESCRIPTORS:
                wass[kdesc] = {
                    "guided_vs_ref": fd2.wasserstein_safe(g_desc[kdesc], r_desc[kdesc]),
                    "base_vs_ref": fd2.wasserstein_safe(b_desc[kdesc], r_desc[kdesc]),
                }
    except Exception as e:
        print(f"[descriptors] failed: {e}", flush=True)

    # ---- assemble aggregator-schema summary ----
    K = len(cfg.guide_ckpts.split(","))
    name = f"compose-{args.bench_key}-{operator}-k{K}-b{int(comp.compose_beta)}"
    gt = b_top = None
    if len(g_scored_primary := np.asarray(composed_scored[primary])) >= 10 and \
       len(np.asarray(base_scored[primary])) >= 10:
        gt = float(np.mean(np.sort(g_scored_primary)[-10:]))
        b_top = float(np.mean(np.sort(np.asarray(base_scored[primary]))[-10:]))

    summary = {
        "name": name,
        "seed": args.seed,
        "n_requested": args.n,
        "diff_steps": cfg.diff_steps,
        "reward": "guacamol",         # primary is the full MPO
        "reward_benchmark": args.bench_key,
        "reward_smiles": None,
        "operator": operator,
        "product_kind": pk,
        "compose_space": args.compose_space,
        "compose_beta": comp.compose_beta,
        "weights": args.weights,
        "train_betas": args.train_betas,
        "n_components": K,
        "primary_reward": primary,
        "guided": g["block"],
        "base": b["block"],
        "fcd": fcd,
        "descriptor_wasserstein": wass,
        # keep the full multi-objective component means too, for composition analysis
        "component_means_composed": {rn: float(np.mean(composed_scored[rn]))
                                     for rn in reward_names},
        "component_means_base": {rn: float(np.mean(base_scored[rn]))
                                 for rn in reward_names},
    }
    if gt is not None:
        summary["top10_delta_guided_minus_base"] = gt - b_top

    with open(os.path.join(out_dir, "dump_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # save composed + base smiles/rewards for reuse
    with open(os.path.join(out_dir, "composed_smiles.txt"), "w") as f:
        f.write("\n".join(g["smiles"]))
    with open(os.path.join(out_dir, "base_smiles.txt"), "w") as f:
        f.write("\n".join(b["smiles"]))
    np.save(os.path.join(out_dir, "composed_rewards.npy"),
            np.asarray(composed_scored[primary]))
    np.save(os.path.join(out_dir, "base_rewards.npy"),
            np.asarray(base_scored[primary]))

    print(f"[compose] {operator}: composed top10={summary.get('top10_delta_guided_minus_base')} "
          f"atom_stab={g['block']['atom_stability']} -> {out_dir}", flush=True)

    # ---- ALSO run compose.py's OWN rich reporting for this operator ----
    if not args.skip_native_plots:
        try:
            _run_compose_native(args, op, pk, native_dir)
        except Exception as e:
            print(f"[compose_native] plotting failed for {operator}: {e}", flush=True)


def _run_compose_native(args, op, pk, native_dir):
    """Invoke compose.py's main() via its argv parser so its full plot/report
    suite (hist, kde, ternary, hypervolume, slope) is produced natively."""
    import sys, runpy
    argv = [
        "compose.py",
        "--quetzal_ckpt", args.quetzal_ckpt,
        "--dataset", args.dataset,
        "--diff_steps", str(args.diff_steps),
        "--guide_ckpts", args.guide_ckpts,
        "--guide_labels", args.guide_labels,
        "--guide_source", args.guide_source,
        "--eval_rewards", args.eval_rewards,
        "--operator", op,
        "--product_kind", pk,
        "--weights", args.weights,
        "--train_betas", args.train_betas,
        "--compose_space", args.compose_space,
        "--compose_beta", str(args.compose_beta),
        "--n_samples", str(args.n),
        "--seed", str(args.seed),
        "--out_dir", native_dir,
        "--tag", f"{args.bench_key}_{op}_{pk}",
        "--make_base",
    ]
    if args.use_logz:
        argv.append("--use_logz")
    if args.ref_smiles and os.path.exists(args.ref_smiles):
        argv += ["--fcd_ref_smiles", args.ref_smiles]
    print(f"[compose_native] {' '.join(argv[1:])}", flush=True)
    old = sys.argv
    try:
        sys.argv = argv
        runpy.run_module("gflow_multi", run_name="__main__")
    finally:
        sys.argv = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guide_ckpts", required=True, help="comma-separated .ckpt paths")
    ap.add_argument("--guide_labels", default="")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--train_betas", required=True)
    ap.add_argument("--eval_rewards", required=True,
                    help="compose.py eval spec; FIRST entry is the primary reward")
    ap.add_argument("--bench_key", default="osimertinib", help="for gcomp + naming")
    ap.add_argument("--operators", default="linear,product,harmonic")
    ap.add_argument("--compose_space", default="tilted")
    ap.add_argument("--compose_beta", type=float, default=0.0)
    ap.add_argument("--use_logz", action="store_true")
    ap.add_argument("--guide_source", default="ema")
    ap.add_argument("--make_singles", action="store_true")
    ap.add_argument("--quetzal_ckpt", default="geom.ckpt")
    ap.add_argument("--train_module", default="train.py")
    ap.add_argument("--dataset", default="geom")
    ap.add_argument("--diff_steps", type=int, default=18)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--sample_temp", type=float, default=1.0)
    ap.add_argument("--rand_eps", type=float, default=0.0)
    ap.add_argument("--ref_smiles", default=None)
    ap.add_argument("--ref_limit", type=int, default=None)
    ap.add_argument("--no_fcd", action="store_true")
    ap.add_argument("--skip_native_plots", action="store_true",
                    help="skip compose.py's own plot suite (only emit aggregator summary)")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--progress_every", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    if args.ref_limit is None:
        args.ref_limit = args.n

    for operator in [o.strip() for o in args.operators.split(",") if o.strip()]:
        if operator not in OPERATOR_FLAGS:
            print(f"[warn] unknown operator {operator!r}; skipping "
                  f"(valid: {list(OPERATOR_FLAGS)})")
            continue
        dump_one_operator(args, operator)

    print(f"[done] composed dumps -> {args.out_dir}")


if __name__ == "__main__":
    main()
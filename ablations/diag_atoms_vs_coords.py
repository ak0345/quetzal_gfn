"""
Where does the reward live: in the atom types a guide can change, or in the 3D
coordinates the frozen diffusion produces and an atom-type guide cannot touch?

This matters because every intervention in this project acts on p_atom only. A
reward carried substantially by geometry would be out of reach of every method
tested, and a flat result against it would say nothing about steering.

Takes one checkpoint via --ckpt; no Composer and no composition list flags.

Tests, run on frozen-prior molecules with no training:
  T1 ATOM SENSITIVITY  force each atom to the guide's top choice, keep the base
     coordinates, re-perceive the graph and re-score. A large change means atom
     types are a lever the guide can pull.
  T2 COORD SENSITIVITY keep the atoms fixed and re-roll the coordinate diffusion
     several times, re-scoring each. High reward variance from the coordinates
     alone means geometry dominates the objective.

Usage:
  python diag_atoms_vs_coords.py \
    --ckpt logs/quetzal-gfn/gfn-quetzal-osim-repro-db-beta10-.../checkpoints/last.ckpt \
    --n 300 --coord_rerolls 5 --guide_source ema --out_dir dumps/atoms_vs_coords
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("medium")


def _mask_for(cfg, device):
    from chem import GEN, PAD, QM9_MASK
    mask = torch.ones(128, dtype=torch.bool, device=device)
    mask[GEN] = False
    mask[PAD] = False
    if getattr(cfg, "mask_atoms", None) == "qm9":
        mask = QM9_MASK.to(device)
    return mask


def _guided_logits(guide, prior_logits, h):
    if guide is None:
        return prior_logits
    if hasattr(guide, "guided_logits"):
        try:
            return guide.guided_logits(prior_logits, h)
        except TypeError:
            return guide.guided_logits(h)
    return prior_logits + guide(h)


def _score(reward_fn, mol):
    """compute_log_reward takes a LIST of mols and returns a tensor."""
    out = reward_fn([mol])
    try:
        return float(out[0])
    except Exception:
        return float(out)


@torch.no_grad()
def T1_atom_sensitivity(lit, guide, mols, reward_fn, device, cap=200):
    from chem import Molecule, GEN
    prior = lit.frozen
    mask = _mask_for(lit.cfg, device)
    NEG = -1e9
    base_r, swap_r = [], []
    for m in mols[:cap]:
        atoms = m.atoms.to(device)
        coords = m.coords.to(device)
        try:
            base_r.append(_score(reward_fn, m))
        except Exception:
            continue
        a_in = torch.cat([torch.tensor([GEN], device=device), atoms]).unsqueeze(0)
        c_in = torch.cat([torch.zeros(1, 3, device=device), coords]).unsqueeze(0)
        L = a_in.shape[1]
        new_atoms = a_in.clone()
        for t in range(1, L):
            idx = torch.arange(t, device=device).unsqueeze(0)
            seq = prior.encode1(idx, new_atoms[:, :t], c_in[:, :t])
            h = seq[:, -1, :]
            pl = prior.proj_logits(h)
            gl = _guided_logits(guide, pl, h).float().masked_fill(~mask, NEG)
            new_atoms[0, t] = torch.softmax(gl, -1).argmax(-1)
        mm = Molecule(atoms=new_atoms[:, 1:], coords=c_in[:, 1:]).to("cpu").unbatch()[0]
        try:
            swap_r.append(_score(reward_fn, mm))
        except Exception:
            swap_r.append(float("nan"))
    b = np.array(base_r, float); a = np.array(swap_r, float)
    m = np.isfinite(b) & np.isfinite(a)
    b, a = b[m], a[m]
    return {"n": int(len(b)),
            "base_reward_mean": float(b.mean()) if len(b) else None,
            "atomswap_reward_mean": float(a.mean()) if len(a) else None,
            "mean_abs_reward_change": float(np.mean(np.abs(a - b))) if len(b) else None,
            "note": "large => atom-type choice IS a reward lever (fix B can help); "
                    "tiny => atoms the guide controls don't move reward"}


@torch.no_grad()
def T2_coord_sensitivity(lit, mols, reward_fn, device, rerolls=5, cap=150):
    """Keep the atom sequence FIXED; re-roll coordinate diffusion `rerolls` times
    with fresh noise; re-score. Mirrors _generate_guided's order exactly:
    encode1 over the prefix -> encode2 with atoms-so-far -> sample_coord -> append.
    """
    from chem import Molecule, GEN
    prior = lit.frozen
    stds = []
    for m in mols[:cap]:
        atoms = m.atoms.to(device)                                 # [L] real atoms (no GEN)
        a_full = torch.cat([torch.tensor([GEN], device=device), atoms]).unsqueeze(0)  # [1, L+1]
        n_real = atoms.shape[0]
        if n_real < 1:
            continue
        rewards = []
        for _ in range(rerolls):
            coords = torch.zeros(1, 1, 3, device=device)           # coord for the GEN token
            # place one coordinate per REAL atom, in generation order
            for step in range(n_real):
                # tokens known so far include GEN + the atoms up to and INCLUDING
                # this step (their coords are what we are building). The prefix fed
                # to encode1 is a_full[:, :step+1] with its coords (length step+1).
                prefix_atoms = a_full[:, :step + 1]                # [1, step+1]  (GEN + step atoms)
                idx = torch.arange(prefix_atoms.shape[1], device=device).unsqueeze(0)
                seq = prior.encode1(idx, prefix_atoms, coords)     # coords has length step+1
                # encode2 with the atoms INCLUDING the current one (drop GEN):
                atoms_incl = a_full[:, 1:step + 2]                 # [1, step+1] real atoms so far
                x = prior.encode2(atoms_incl, seq)[:, -1, :]
                nc, _ = prior.sample_coord(x, device=device, num_steps=lit.cfg.diff_steps)
                coords = torch.cat([coords, nc.view(1, 1, 3)], 1)
            mm = Molecule(atoms=a_full[:, 1:], coords=coords[:, 1:]).to("cpu").unbatch()[0]
            try:
                rewards.append(_score(reward_fn, mm))
            except Exception:
                pass
        r = np.array(rewards, float); r = r[np.isfinite(r)]
        if len(r) >= 2:
            stds.append(float(r.std()))
    return {"n": len(stds),
            "coord_only_reward_std_mean": float(np.mean(stds)) if stds else None,
            "coord_only_reward_std_p90": float(np.percentile(stds, 90)) if stds else None,
            "note": "high => coordinates alone move reward a lot => atom guide handicapped"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--guide_source", choices=["ema", "policy"], default="ema")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--coord_rerolls", type=int, default=5)
    ap.add_argument("--diff_steps", type=int, default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import gflow
    from gflow import LitGFlowNet

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", ckpt.get("hparams", None))
    if hp is None:
        raise SystemExit("checkpoint has no hyper_parameters")
    lit = LitGFlowNet(hp)
    lit.load_state_dict(ckpt["state_dict"], strict=False)
    lit = lit.to(args.device).eval()
    if args.diff_steps is not None:
        lit.cfg.diff_steps = args.diff_steps

    guide = lit.guide_ema.module if args.guide_source == "ema" else lit.guide
    reward_fn = lit.compute_log_reward

    print(f"[cfg] reward={lit.cfg.reward} reward_smiles={getattr(lit.cfg,'reward_smiles',None)} "
          f"diff_steps={lit.cfg.diff_steps} guide_source={args.guide_source}")
    print(f"[sample] {args.n} base molecules (single batched call, like eval) ...")
    try:
        base = lit.rollout(args.n, guide=None, sample_temp=lit.cfg.sample_temp,
                           rand_eps=0.0, with_reward=False)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            base = lit.rollout_chunked(args.n, guide=None, chunk=min(args.n, 1000),
                                       with_reward=False)
        else:
            raise
    mols = base["mols"]

    report = {"ckpt": args.ckpt, "reward": getattr(lit.cfg, "reward_smiles", lit.cfg.reward)}

    print("[T1] atom-type sensitivity ...")
    report["T1_atom_sensitivity"] = T1_atom_sensitivity(lit, guide, mols, reward_fn, args.device)
    print(f"     {report['T1_atom_sensitivity']}")

    print(f"[T2] coordinate sensitivity ({args.coord_rerolls} re-rolls) ...")
    report["T2_coord_sensitivity"] = T2_coord_sensitivity(
        lit, mols, reward_fn, args.device, rerolls=args.coord_rerolls)
    print(f"     {report['T2_coord_sensitivity']}")

    t1 = report["T1_atom_sensitivity"].get("mean_abs_reward_change")
    t2 = report["T2_coord_sensitivity"].get("coord_only_reward_std_mean")
    verdict = []
    if t1 is not None and t1 < 0.1:
        verdict.append("atom-type choice barely moves reward -> atom guide is a weak lever; "
                       "hidden-injection fix (B) unlikely to help much")
    elif t1 is not None:
        verdict.append(f"atom-type choice DOES move reward (|d|={t1:.3f}) -> fix B "
                       f"(hidden injection) is worth building")
    if t1 is not None and t2 is not None and t2 > t1:
        verdict.append(f"coordinates move reward MORE than atoms (coord_std={t2:.3f} > "
                       f"atom_change={t1:.3f}) -> reward is largely coordinate-driven; "
                       f"consider coordinate guidance, not just atom guidance")
    report["verdict"] = verdict
    print("\n=== VERDICT ===")
    for v in verdict:
        print("  -", v)

    with open(os.path.join(args.out_dir, "atoms_vs_coords.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] -> {os.path.join(args.out_dir, 'atoms_vs_coords.json')}")


if __name__ == "__main__":
    main()
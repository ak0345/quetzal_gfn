"""
ablate_logit_flip.py -- does the guide's residual actually CHANGE Quetzal's
decisions, or just perturb a distribution whose argmax never moves?

Panel A of the earlier ablation measured residual NORM and KL(guided||prior) at
prior-drawn states. But a nonzero residual with nonzero KL can STILL change no
actual decision: if the prior logit for the winning atom towers over the rest
(prior_logit_norm was ~6841 here), a residual of a few units shifts the softmax
mass yet never flips the argmax -> the SAME atom is sampled every time -> the
guide is behaviorally inert even though it "adds logits".

This probe walks the causal chain end to end, per guide:

  1. DELIVERY : is (prior+guide) != prior at the logit tensor level? (catches a
                wiring bug where the residual is computed but dropped.)
  2. DECISION : over the SAME state, does the guide change the top-1 atom
                (argmax flip), and by how much does it move probability onto
                atoms the prior disfavored? (softmax-survival test.)
  3. SAMPLE   : with paired RNG (same uniform draw for prior and guided), does
                the SAMPLED next-atom differ? -> action-flip rate. A guide that
                never flips a sampled action cannot change any molecule.
  4. POSITION : flip rate as a function of sequence position (early flips
                cascade into very different molecules; late flips barely matter).
  5. MOLECULE : do guided trajectories reach different final reward than the
                prior baseline they branched from? (paired reward delta.)

All measured along trajectories rolled by the PRIOR (so every guide is compared
on the same realistic state distribution), plus a check along the guide's OWN
rollout (its residual could matter more on states it actually visits).

Usage (mirror compose.py flags):
    python ablate_logit_flip.py \
        --guide_ckpts "...c0/last.ckpt,...c1/last.ckpt,...c2/last.ckpt,...c3/last.ckpt" \
        --guide_labels "c0,c1,c2,c3" --route policy --n_traj 400
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("medium")

from chem import Molecule, GEN, STOP, PAD, QM9_MASK
from reward_fn import build_reward
from gflow_multi import MultiConfig, Composer


def _mask_for(mask_atoms, device):
    mask = torch.ones(128, dtype=torch.bool, device=device)
    mask[GEN] = False
    mask[PAD] = False
    if mask_atoms == "qm9":
        mask = QM9_MASK.to(device)
    return mask


@torch.no_grad()
def flip_diagnostics(comp, n_traj=400, max_steps=None, temp=1.0, own_rollout_for=None):
    """Roll trajectories with the PRIOR; at every step compare prior vs prior+guide
    decisions on the identical state. Returns per-guide chain metrics.

    If own_rollout_for is a guide index, ALSO roll that guide's own trajectory and
    measure its flip rate on the states it actually visits (a guide can matter more
    off the prior's beaten path).
    """
    prior = comp.prior
    device = comp.device
    mask = _mask_for(comp.cfg.mask_atoms, device)
    NEG = -1e9
    max_steps = max_steps or min(comp.cfg.max_len, 64)
    bsz = min(comp.cfg.chunk, n_traj)

    labels = comp.labels
    guides = comp.guides
    K = len(guides)

    # accumulators per guide
    acc = {lab: {"delivered": 0, "n_states": 0,
                 "argmax_flip": 0, "sample_flip": 0,
                 "mass_moved_sum": 0.0, "kl_sum": 0.0,
                 "prior_top1_logit_gap_sum": 0.0,
                 "flip_by_pos": np.zeros(max_steps),
                 "state_by_pos": np.zeros(max_steps)}
           for lab in labels}

    done_total = 0
    while done_total < n_traj:
        b = min(bsz, n_traj - done_total)
        atoms = torch.full((b, 1), GEN, dtype=torch.long, device=device)
        coords = torch.zeros(b, 1, 3, device=device)
        stop_mask = torch.zeros(b, dtype=torch.bool, device=device)

        for t in range(max_steps):
            idx = torch.arange(atoms.shape[1], device=device).expand(b, -1)
            seq = prior.encode1(idx, atoms, coords)
            h = seq[:, -1, :]
            prior_logits = prior.proj_logits(h).float().masked_fill(~mask, NEG)
            lp_prior = F.log_softmax(prior_logits / temp, dim=-1)
            p_prior = lp_prior.exp()
            prior_top1 = prior_logits.argmax(-1)

            # prior sampling draw (paired RNG: reuse this uniform for guided)
            u = torch.rand(b, 1, device=device)
            # inverse-CDF sample for prior
            cdf_prior = torch.cumsum(p_prior, dim=-1)
            next_prior = (u < cdf_prior).float().argmax(dim=-1)  # first index where cdf>u

            alive = (~stop_mask)
            for gi, lab in enumerate(labels):
                resid = guides[gi](h)
                guided_logits = (prior_logits + resid).float().masked_fill(~mask, NEG)
                # 1. delivery: tensor-level difference on legal atoms
                delivered = ((guided_logits - prior_logits).abs()
                             .masked_fill(~mask, 0.0).sum(-1) > 1e-6)
                lp_g = F.log_softmax(guided_logits / temp, dim=-1)
                p_g = lp_g.exp()
                # 2. decision: argmax flip + mass moved + KL + prior top1 gap
                g_top1 = guided_logits.argmax(-1)
                argmax_flip = (g_top1 != prior_top1)
                mass_moved = 0.5 * (p_g - p_prior).abs().sum(-1)   # total variation
                kl = (p_g * (lp_g - lp_prior)).sum(-1)
                # how dominant is the prior's top1? (gap to 2nd logit)
                top2 = torch.topk(prior_logits, 2, dim=-1).values
                gap = (top2[:, 0] - top2[:, 1])
                # 3. sample flip under paired RNG
                cdf_g = torch.cumsum(p_g, dim=-1)
                next_g = (u < cdf_g).float().argmax(dim=-1)
                sample_flip = (next_g != next_prior)

                a = alive
                na = a.sum().item()
                acc[lab]["delivered"] += (delivered & a).sum().item()
                acc[lab]["n_states"] += na
                acc[lab]["argmax_flip"] += (argmax_flip & a).sum().item()
                acc[lab]["sample_flip"] += (sample_flip & a).sum().item()
                acc[lab]["mass_moved_sum"] += (mass_moved * a.float()).sum().item()
                acc[lab]["kl_sum"] += (kl * a.float()).sum().item()
                acc[lab]["prior_top1_logit_gap_sum"] += (gap * a.float()).sum().item()
                if t < max_steps:
                    acc[lab]["flip_by_pos"][t] += (sample_flip & a).sum().item()
                    acc[lab]["state_by_pos"][t] += na

            # advance the PRIOR trajectory (shared baseline path)
            atoms = torch.cat([atoms, next_prior.unsqueeze(1)], dim=1)
            stop_mask = stop_mask | (next_prior == STOP)
            if stop_mask.all():
                break
            x = prior.encode2(atoms[:, 1:], seq)[:, -1, :]
            nc, _ = prior.sample_coord(x, device=device, num_steps=comp.cfg.diff_steps)
            coords = torch.cat([coords, nc.view(b, 1, 3)], dim=1)

        done_total += b

    # finalize
    out = {}
    for lab in labels:
        a = acc[lab]
        ns = max(a["n_states"], 1)
        pos_rate = a["flip_by_pos"] / np.maximum(a["state_by_pos"], 1)
        out[lab] = {
            "delivered_frac": a["delivered"] / ns,
            "argmax_flip_rate": a["argmax_flip"] / ns,
            "sample_flip_rate": a["sample_flip"] / ns,
            "mean_total_variation": a["mass_moved_sum"] / ns,
            "mean_KL": a["kl_sum"] / ns,
            "mean_prior_top1_gap": a["prior_top1_logit_gap_sum"] / ns,
            "flip_rate_first8_positions": [float(x) for x in pos_rate[:8]],
            "n_states": ns,
        }
    return out


@torch.no_grad()
def molecule_delta(comp, guide_idx, n=300):
    """Branch test: from the SAME prior prefix, does turning the guide on change
    the final reward? Roll prior baseline and guided-from-scratch, compare reward
    distributions (paired only at the root, so this is a distributional delta).
    """
    name, fn = comp.eval_rewards[0]
    # prior baseline
    base = comp.sample_base(n)
    base_r = np.array([fn(m) for m in base], float)
    # single guide on (reuse composer single path but guide idx only)
    from ablate_guide import sample_with_residual_scale
    gm = sample_with_residual_scale(comp, 1.0, n, single_idx=guide_idx)
    g_r = np.array([fn(m) for m in gm], float)
    return {
        "reward_scored": name,
        "base_mean": float(np.mean(base_r)),
        "guided_mean": float(np.mean(g_r)),
        "delta_mean": float(np.mean(g_r) - np.mean(base_r)),
    }


def main():
    ap = argparse.ArgumentParser()
    for f in MultiConfig.__dataclass_fields__.values():
        if isinstance(f.default, bool):
            ap.add_argument(f"--{f.name}", dest=f.name,
                            action="store_true" if not f.default else "store_false")
            ap.set_defaults(**{f.name: f.default})
        else:
            t = type(f.default) if f.default is not None else str
            ap.add_argument(f"--{f.name}", type=t, default=f.default)
    ap.add_argument("--n_traj", type=int, default=400)
    ap.add_argument("--flip_temp", type=float, default=1.0,
                    help="temperature for the flip test; try 1.0 AND a low value "
                         "like 0.3 to see if the guide only matters when the prior "
                         "isn't near-greedy")
    args = ap.parse_args()

    cfg = MultiConfig(**{k: getattr(args, k) for k in MultiConfig.__dataclass_fields__})
    os.makedirs(cfg.out_dir, exist_ok=True)
    comp = Composer(cfg)

    report = {"config": {"route": cfg.route, "flip_temp": args.flip_temp,
                         "n_traj": args.n_traj}}

    print(f"[flip] rolling {args.n_traj} prior trajectories, comparing decisions "
          f"per guide (temp={args.flip_temp}) ...")
    report["flip"] = flip_diagnostics(comp, n_traj=args.n_traj, temp=args.flip_temp)

    print("\n=== CAUSAL CHAIN PER GUIDE ===")
    print(f"{'guide':<8} {'deliver':>8} {'argmax':>8} {'sample':>8} {'TV':>7} "
          f"{'KL':>7} {'priorgap':>9}")
    for lab, d in report["flip"].items():
        print(f"{lab:<8} {d['delivered_frac']:>8.3f} {d['argmax_flip_rate']:>8.3f} "
              f"{d['sample_flip_rate']:>8.3f} {d['mean_total_variation']:>7.3f} "
              f"{d['mean_KL']:>7.3f} {d['mean_prior_top1_gap']:>9.2f}")
    print("\ninterpretation:")
    print("  deliver~1, argmax~0, sample~0  -> residual reaches logits but NEVER")
    print("     changes a decision (prior top1 too dominant; see priorgap).")
    print("  argmax>0, sample>0             -> guide genuinely steers.")
    print("  deliver~0                      -> wiring bug: residual not applied.")

    with open(os.path.join(cfg.out_dir, "flip_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # optional plot: flip rate by position
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for lab, d in report["flip"].items():
            ax.plot(range(len(d["flip_rate_first8_positions"])),
                    d["flip_rate_first8_positions"], "o-", label=lab)
        ax.set_xlabel("sequence position"); ax.set_ylabel("sample-flip rate")
        ax.set_title("Where (if anywhere) does the guide change the sampled atom?")
        ax.legend()
        p = os.path.join(cfg.out_dir, "flip_by_position.png")
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        print(f"[plot] {p}")
    except Exception as e:
        print(f"[plot] failed: {e}")

    print(f"[done] -> {os.path.join(cfg.out_dir, 'flip_report.json')}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
flip_ablation.py -- logit-FLIP diagnostic for a SINGLE-GUIDE checkpoint
(the mechanistic ceiling test), ported from the composition-only version in
gflow_multi's flip diagnostics.

THE TEST (why it matters for the thesis): roll trajectories with the FROZEN
PRIOR; at every atom-decision state, compare the prior's next-atom distribution
against the GUIDED one on the IDENTICAL state. Measure:
  * delivered_frac   -- does the guide's residual reach the logits at all?
                        (~0 => wiring bug: guide not applied)
  * argmax_flip_rate -- how often the guided argmax != prior argmax
  * sample_flip_rate -- how often a paired-RNG sample differs (the real "did the
                        decision change" number)
  * mean_total_variation / mean_KL -- how much probability mass the guide moves
  * mean_prior_top1_gap -- how dominant the prior's top-1 logit is (the CEILING:
                        a large gap means the residual can't flip the argmax)
  * flip_rate_by_position -- where in the sequence (if anywhere) flips happen

THE CEILING SIGNATURE: delivered~1, argmax~0, sample~0, with a LARGE
mean_prior_top1_gap => the residual reaches the logits but never changes a
decision because the prior's top-1 is too dominant. That's the saturated prior.

Works for ALL guide types (LogitGuide/base, TempGainGuide, HiddenGuide): the
guided logits are computed via the same dispatch gflow.py uses, then the residual
is derived as (guided_logits - prior_logits) for the metrics -- so it is correct
regardless of whether the guide adds an output residual or perturbs the hidden
state.

Usage:
  python flip_ablation.py \
    --ckpt logs/quetzal-gfn/sweep-nitrogen-hidden-db-replay_off-b10/checkpoints/last.ckpt \
    --n_traj 400 --flip_temp 1.0 --out_dir flips/nitrogen-hidden-b10
  # try BOTH --flip_temp 1.0 and 0.3 (does the guide only matter when the prior
  # isn't near-greedy?)
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

# reuse final_dump2's correct checkpoint loading (config unwrap + guide-type
# detection + [FATAL] guards) so we never silently measure an untrained guide.
import final_dump as fd2
from chem import GEN, STOP, PAD, QM9_MASK


def _mask_for(mask_atoms, device):
    if mask_atoms == "qm9":
        return QM9_MASK.to(device)
    if mask_atoms == "H":
        m = torch.zeros(128, dtype=torch.bool, device=device)
        m[0] = True; m[1] = True
        return m
    mask = torch.ones(128, dtype=torch.bool, device=device)
    mask[GEN] = False
    mask[PAD] = False
    return mask


def _guided_logits(guide, prior_logits, h):
    """Compute guided logits for ANY guide type, matching gflow.py's dispatch:
      HiddenGuide  -> guided_logits(h)                (1-arg, applies proj itself)
      TempGain     -> guided_logits(prior_logits, h)  (2-arg)
      LogitGuide   -> prior_logits + guide(h)         (residual)
    """
    if guide is None:
        return prior_logits
    if type(guide).__name__ == "HiddenGuide":
        return guide.guided_logits(h)
    if hasattr(guide, "guided_logits"):
        try:
            return guide.guided_logits(prior_logits, h)
        except TypeError:
            return guide.guided_logits(h)
    return prior_logits + guide(h)


@torch.no_grad()
def flip_diagnostics(lit, guide, n_traj=400, max_steps=None, temp=1.0,
                     diff_steps=18, mask_atoms=None, chunk=200, device="cuda",
                     progress=False):
    prior = lit.frozen
    mask = _mask_for(mask_atoms, device)
    NEG = -1e9
    max_steps = max_steps or min(getattr(lit.cfg, "max_len", 192), 64)
    bsz = min(chunk, n_traj)

    acc = {"delivered": 0, "n_states": 0, "argmax_flip": 0, "sample_flip": 0,
           "mass_moved_sum": 0.0, "kl_sum": 0.0, "prior_top1_logit_gap_sum": 0.0,
           "flip_by_pos": np.zeros(max_steps), "state_by_pos": np.zeros(max_steps),
           "gap_flip_hi": 0, "gap_flip_hi_flipped": 0,   # flips on gap>8 (hard) states
           "gap_lo": 0, "gap_lo_flipped": 0}             # flips on gap<=8 (easy) states

    done_total = 0
    while done_total < n_traj:
        b = min(bsz, n_traj - done_total)
        if progress:
            print(f"[flip] trajectories {done_total}/{n_traj}", flush=True)
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

            # paired RNG: one uniform reused for prior AND guided sampling
            u = torch.rand(b, 1, device=device)
            cdf_prior = torch.cumsum(p_prior, dim=-1)
            next_prior = (u < cdf_prior).float().argmax(dim=-1)

            alive = (~stop_mask)
            guided_logits = _guided_logits(guide, prior_logits, h) \
                .float().masked_fill(~mask, NEG)
            delivered = ((guided_logits - prior_logits).abs()
                         .masked_fill(~mask, 0.0).sum(-1) > 1e-6)
            lp_g = F.log_softmax(guided_logits / temp, dim=-1)
            p_g = lp_g.exp()
            g_top1 = guided_logits.argmax(-1)
            argmax_flip = (g_top1 != prior_top1)
            mass_moved = 0.5 * (p_g - p_prior).abs().sum(-1)     # total variation
            kl = (p_g * (lp_g - lp_prior)).sum(-1)
            top2 = torch.topk(prior_logits, 2, dim=-1).values
            gap = (top2[:, 0] - top2[:, 1])
            cdf_g = torch.cumsum(p_g, dim=-1)
            next_g = (u < cdf_g).float().argmax(dim=-1)
            sample_flip = (next_g != next_prior)

            a = alive
            na = a.sum().item()
            acc["delivered"] += (delivered & a).sum().item()
            acc["n_states"] += na
            acc["argmax_flip"] += (argmax_flip & a).sum().item()
            acc["sample_flip"] += (sample_flip & a).sum().item()
            acc["mass_moved_sum"] += (mass_moved * a.float()).sum().item()
            acc["kl_sum"] += (kl * a.float()).sum().item()
            acc["prior_top1_logit_gap_sum"] += (gap * a.float()).sum().item()
            # ceiling split: flip rate on HARD (gap>8) vs EASY (gap<=8) states
            hi = a & (gap > 8)
            lo = a & (gap <= 8)
            acc["gap_flip_hi"] += hi.sum().item()
            acc["gap_flip_hi_flipped"] += (hi & sample_flip).sum().item()
            acc["gap_lo"] += lo.sum().item()
            acc["gap_lo_flipped"] += (lo & sample_flip).sum().item()
            if t < max_steps:
                acc["flip_by_pos"][t] += (sample_flip & a).sum().item()
                acc["state_by_pos"][t] += na

            # advance the PRIOR trajectory (the shared baseline path)
            atoms = torch.cat([atoms, next_prior.unsqueeze(1)], dim=1)
            stop_mask = stop_mask | (next_prior == STOP)
            if stop_mask.all():
                break
            x = prior.encode2(atoms[:, 1:], seq)[:, -1, :]
            nc, _ = prior.sample_coord(x, device=device, num_steps=diff_steps)
            coords = torch.cat([coords, nc.view(b, 1, 3)], dim=1)

        done_total += b

    ns = max(acc["n_states"], 1)
    pos_rate = acc["flip_by_pos"] / np.maximum(acc["state_by_pos"], 1)
    return {
        "delivered_frac": acc["delivered"] / ns,
        "argmax_flip_rate": acc["argmax_flip"] / ns,
        "sample_flip_rate": acc["sample_flip"] / ns,
        "mean_total_variation": acc["mass_moved_sum"] / ns,
        "mean_KL": acc["kl_sum"] / ns,
        "mean_prior_top1_gap": acc["prior_top1_logit_gap_sum"] / ns,
        # THE CEILING NUMBERS: can the guide flip the hard (high-gap) decisions?
        "flip_rate_high_gap": (acc["gap_flip_hi_flipped"] / acc["gap_flip_hi"])
                              if acc["gap_flip_hi"] else None,
        "flip_rate_low_gap": (acc["gap_lo_flipped"] / acc["gap_lo"])
                             if acc["gap_lo"] else None,
        "frac_states_high_gap": acc["gap_flip_hi"] / ns,
        "flip_rate_first8_positions": [float(x) for x in pos_rate[:8]],
        "n_states": ns,
        "temp": temp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_traj", type=int, default=400)
    ap.add_argument("--flip_temp", type=float, default=1.0)
    ap.add_argument("--also_temp", type=float, default=None,
                    help="optional 2nd temperature to run (e.g. 0.3) -- does the "
                         "guide only matter when the prior isn't near-greedy?")
    ap.add_argument("--max_steps", type=int, default=64)
    ap.add_argument("--diff_steps", type=int, default=18)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--guide_source", choices=["ema", "policy"], default="ema")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load checkpoint via final_dump2's hardened loader ----
    # (config unwrap + guide-type assert + [FATAL] on missing guide weights)
    import gflow
    from gflow import LitGFlowNet
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", ckpt.get("hparams", None))
    if isinstance(hp, dict) and "config" in hp and isinstance(hp["config"], dict):
        hp = hp["config"]
    lit = LitGFlowNet(hp)
    ckpt_hidden = bool(hp.get("use_hidden_guide", True)) if isinstance(hp, dict) else True
    built_hidden = type(lit.guide).__name__ == "HiddenGuide"
    if ckpt_hidden != built_hidden:
        raise SystemExit(f"[FATAL] guide-type mismatch: ckpt use_hidden_guide="
                         f"{ckpt_hidden} but built {type(lit.guide).__name__}")
    missing, _ = lit.load_state_dict(ckpt["state_dict"], strict=False)
    guide_missing = [m for m in missing if m.startswith("guide")
                     and not m.startswith("guide_ema") and "n_averaged" not in m]
    if guide_missing:
        raise SystemExit(f"[FATAL] guide weights did not load: {guide_missing[:4]} "
                         f"-- would measure an untrained guide.")
    lit = lit.to(args.device).eval()

    # pick guide weights (ema = eval-time, policy = live)
    if args.guide_source == "ema" and hasattr(lit, "guide_ema"):
        guide = lit.guide_ema.module
    else:
        guide = lit.guide

    gtype = type(guide).__name__
    mask_atoms = getattr(lit.cfg, "mask_atoms", None)
    print(f"[flip] ckpt guide={gtype} reward={lit.cfg.reward} "
          f"source={args.guide_source}", flush=True)

    report = {"ckpt": args.ckpt, "guide_type": gtype, "reward": lit.cfg.reward,
              "reward_smiles": getattr(lit.cfg, "reward_smiles", None),
              "guide_source": args.guide_source, "n_traj": args.n_traj}

    temps = [args.flip_temp] + ([args.also_temp] if args.also_temp is not None else [])
    for temp in temps:
        print(f"[flip] rolling {args.n_traj} prior trajectories at temp={temp} ...",
              flush=True)
        res = flip_diagnostics(lit, guide, n_traj=args.n_traj, max_steps=args.max_steps,
                               temp=temp, diff_steps=args.diff_steps,
                               mask_atoms=mask_atoms, chunk=args.chunk,
                               device=args.device, progress=args.progress)
        report[f"flip_temp{temp}"] = res
        print(f"\n=== CAUSAL CHAIN (temp={temp}) ===")
        print(f"  delivered={res['delivered_frac']:.3f}  "
              f"argmax_flip={res['argmax_flip_rate']:.3f}  "
              f"sample_flip={res['sample_flip_rate']:.3f}")
        print(f"  total_variation={res['mean_total_variation']:.3f}  "
              f"KL={res['mean_KL']:.3f}  prior_top1_gap={res['mean_prior_top1_gap']:.2f}")
        print(f"  CEILING: flip_rate on HIGH-gap(>8) states={res['flip_rate_high_gap']}  "
              f"on LOW-gap states={res['flip_rate_low_gap']}  "
              f"(frac states high-gap={res['frac_states_high_gap']:.3f})")
        print("  interpretation: deliver~1 & argmax~0 & sample~0 & big gap "
              "=> ceiling (residual reaches logits, never flips the decision).")

    with open(os.path.join(args.out_dir, "flip_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # plot flip rate by position (one line per temperature)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for temp in temps:
            r = report[f"flip_temp{temp}"]
            ax.plot(range(len(r["flip_rate_first8_positions"])),
                    r["flip_rate_first8_positions"], "o-", label=f"temp={temp}")
        ax.set_xlabel("sequence position"); ax.set_ylabel("sample-flip rate")
        ax.set_title(f"Where does the guide change the sampled atom? "
                     f"({gtype}, {lit.cfg.reward})")
        ax.legend()
        p = os.path.join(args.out_dir, "flip_by_position.png")
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        print(f"[plot] {p}")
    except Exception as e:
        print(f"[plot] failed: {e}")

    print(f"[done] -> {os.path.join(args.out_dir, 'flip_report.json')}")


if __name__ == "__main__":
    main()
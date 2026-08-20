"""
Localise where a hidden guide's effect stops, along the causal chain

    delta magnitude -> logit change -> atom flip -> reward movement.

Separates two explanations for a hidden-guide run that leaves the guided sampler
at the prior: the delta never grew, which is a training problem, or the delta is
large but reward-neutral, meaning the guide moves the logits in an unhelpful
direction.

Takes one trained hidden-guide checkpoint via --ckpt.

Probes:
  A1. DELTA MAGNITUDE      ||delta(h)|| over sampled states, relative to ||h||.
                           Near zero points at training; large sends you to A2.
  A2. LOGIT CHANGE         ||guided_logits(h) - proj_logits(h)|| over states, and
                           the change on the runner-up logit, which is what has
                           to move for a decision to change.
  A3. ATOM FLIP RATE       whether the guide changes the sampled atom, binned by
                           the prior's top-1 margin.
  A4. REWARD MOVEMENT      guided against prior top-k reward on freshly sampled
                           molecules.
  A5. TERMINAL-TARGET SCALE the spread of beta*logR at the checkpoint's beta. A
                           spread in the hundreds makes the DB terminal target
                           unfittable by the flow head, which floors the loss.

Usage:
  python ablations/ablate_hidden_guide.py \
      --ckpt logs/quetzal-gfn/sweep-osim-hidden-db-replay_off-b10/checkpoints/last.ckpt \
      --out_dir results/ablations/hidden-guide
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


@torch.no_grad()
def collect_states(lit, guide, n_states=4000, device="cuda"):
    """Sample base-prior trajectories and collect hidden states h + prior logits."""
    from chem import GEN, STOP
    prior = lit.frozen
    mask = _mask_for(lit.cfg, device)
    NEG = -1e9
    # sample a modest batch of trajectories, collect per-step h
    bsz = 128
    atoms = torch.full((bsz, 1), GEN, dtype=torch.long, device=device)
    coords = torch.zeros(bsz, 1, 3, device=device)
    stop_mask = torch.zeros(bsz, dtype=torch.bool, device=device)
    Hs, PL = [], []
    for t in range(lit.cfg.max_len):
        idx = torch.arange(atoms.shape[1], device=device).expand(bsz, -1)
        seq = prior.encode1(idx, atoms, coords)
        h = seq[:, -1, :]
        pl = prior.proj_logits(h)
        alive = ~stop_mask
        if alive.any():
            Hs.append(h[alive].cpu()); PL.append(pl[alive].cpu())
        # sample next atom from the prior (we want prior-distributed states)
        plm = pl.float().masked_fill(~mask, NEG)
        nxt = torch.multinomial(F.softmax(plm, -1), 1)
        stop_mask = stop_mask | (nxt.squeeze(-1) == STOP)
        if stop_mask.all() or sum(x.shape[0] for x in Hs) >= n_states:
            break
        atoms = torch.cat([atoms, nxt], 1)
        x = prior.encode2(atoms[:, 1:], seq)[:, -1, :]
        nc, _ = prior.sample_coord(x, device=device, num_steps=lit.cfg.diff_steps)
        coords = torch.cat([coords, nc.view(bsz, 1, 3)], 1)
    H = torch.cat(Hs)[:n_states]
    P = torch.cat(PL)[:n_states]
    return H.to(device), P.to(device)


@torch.no_grad()
def run(lit, guide, out_dir, device):
    from chem import STOP
    mask = _mask_for(lit.cfg, device)
    NEG = -1e9
    report = {}

    H, PL = collect_states(lit, guide, device=device)
    n = H.shape[0]
    report["n_states"] = int(n)

    # ---- A1: delta magnitude ----
    if not hasattr(guide, "delta"):
        report["ERROR"] = "guide has no .delta -- not a HiddenGuide?"
        return report
    d = guide.delta(H)
    dn = d.norm(dim=-1)
    hn = H.norm(dim=-1)
    report["A1_delta"] = {
        "delta_norm_mean": float(dn.mean()), "delta_norm_p90": float(dn.quantile(0.9)),
        "h_norm_mean": float(hn.mean()),
        "delta_rel_to_h_mean": float((dn / hn.clamp(min=1e-6)).mean()),
        "verdict": ("delta ~0 -> UNTRAINED (H1: gradient never grew it)"
                    if float(dn.mean()) < 1e-3 else
                    "delta is nonzero -> proceed to check if it helps reward"),
    }

    # ---- A2: logit change (esp. runner-up) ----
    guided = guide.guided_logits(H)          # applies proj(h+delta)
    dlogit = (guided - PL)
    # runner-up index per state under the prior
    plm = PL.float().masked_fill(~mask, NEG)
    top2 = plm.topk(2, dim=-1)
    winner = top2.indices[:, 0]; runner = top2.indices[:, 1]
    gap = (top2.values[:, 0] - top2.values[:, 1])
    ru_change = dlogit.gather(-1, runner.unsqueeze(-1)).squeeze(-1)
    report["A2_logit_change"] = {
        "dlogit_norm_mean": float(dlogit.norm(dim=-1).mean()),
        "runnerup_logit_change_mean": float(ru_change.mean()),
        "runnerup_logit_change_p90": float(ru_change.quantile(0.9)),
        "prior_gap_mean": float(gap.mean()),
        "note": "runner-up change must exceed the prior gap to flip a decision",
    }

    # ---- A3: atom flip rate by prior-gap bin ----
    guided_m = guided.float().masked_fill(~mask, NEG)
    g_arg = guided_m.argmax(-1)
    flipped = (g_arg != winner)
    bins = [(0, 2), (2, 4), (4, 8), (8, 1e9)]
    flip_by_gap = {}
    for lo, hi in bins:
        sel = (gap >= lo) & (gap < hi)
        if sel.any():
            flip_by_gap[f"gap_{lo}_{hi}"] = {
                "n": int(sel.sum()),
                "flip_rate": float(flipped[sel].float().mean()),
            }
    report["A3_flip_by_gap"] = flip_by_gap
    report["A3_overall_flip_rate"] = float(flipped.float().mean())
    report["A3_verdict"] = (
        "flips concentrated in low-gap bins only -> still ceiling-bound"
        if flip_by_gap.get("gap_8_1000000000.0", {}).get("flip_rate", 0) < 0.02
        else "flips reach high-gap bins -> hidden injection IS overcoming the ceiling")

    # ---- A4: reward movement (guided vs base top-k) ----
    gh = lit.rollout(400, guide=guide, sample_temp=1.0, rand_eps=0.0, with_reward=True)
    bh = lit.rollout(400, guide=None, sample_temp=1.0, rand_eps=0.0, with_reward=True)
    g_lr = gh["log_reward"].cpu().numpy(); b_lr = bh["log_reward"].cpu().numpy()
    def topk(a, k): return float(np.mean(np.sort(a)[-k:])) if len(a) >= k else None
    report["A4_reward"] = {
        "guided_mean": float(g_lr.mean()), "base_mean": float(b_lr.mean()),
        "guided_top10": topk(g_lr, 10), "base_top10": topk(b_lr, 10),
        "top10_delta": (topk(g_lr, 10) - topk(b_lr, 10)) if topk(g_lr,10) and topk(b_lr,10) else None,
        "verdict": "near 0 => not steering; positive => steering",
    }

    # ---- A5: terminal-target scale (is beta*logR unfittable?) ----
    beta = lit.cfg.reward_beta
    valid = b_lr[b_lr > lit.cfg.invalid_logr + 0.1]
    if len(valid):
        spread = float(np.percentile(valid, 95) - np.percentile(valid, 5))
        report["A5_terminal_scale"] = {
            "beta": beta, "logR_spread_5_95": spread,
            "beta_logR_spread": beta * spread,
            "squared_target_scale": (beta * spread) ** 2,
            "verdict": ("terminal target spread ~hundreds+ -> UNFITTABLE, lower beta"
                        if (beta * spread) ** 2 > 300 else
                        "terminal target scale is fittable; beta is not the main issue"),
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--guide_source", choices=["ema", "policy"], default="ema")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import gflow
    from gflow import LitGFlowNet
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", ckpt.get("hparams"))
    lit = LitGFlowNet(hp); lit.load_state_dict(ckpt["state_dict"], strict=False)
    lit = lit.to(args.device).eval()
    guide = lit.guide_ema.module if args.guide_source == "ema" else lit.guide

    print(f"[cfg] reward_smiles={getattr(lit.cfg,'reward_smiles',None)} "
          f"beta={lit.cfg.reward_beta} objective={lit.cfg.objective} "
          f"use_hidden_guide={getattr(lit.cfg,'use_hidden_guide',False)}")

    rep = run(lit, guide, args.out_dir, args.device)
    print(json.dumps(rep, indent=2))
    with open(os.path.join(args.out_dir, "ablate_hidden.json"), "w") as f:
        json.dump(rep, f, indent=2)

    # one-line overall diagnosis
    print("\n=== DIAGNOSIS ===")
    a1 = rep.get("A1_delta", {})
    a4 = rep.get("A4_reward", {})
    a5 = rep.get("A5_terminal_scale", {})
    if a1.get("delta_norm_mean", 1) < 1e-3:
        print("  delta ~0: the guide never trained. This is a TRAINING problem, not the")
        print("  architecture. Most likely beta too high (see A5) -> lower beta to 1-2.")
    elif a4.get("top10_delta") is not None and abs(a4["top10_delta"]) < 0.05:
        print("  delta is nonzero but reward flat: the guide moves logits but toward")
        print("  reward-NEUTRAL directions. The gradient signal is weak/mis-scaled.")
        if a5.get("squared_target_scale", 0) > 300:
            print("  A5 confirms beta*logR target is unfittable -> lower beta.")
    else:
        print("  reward IS moving -- check A4 top10_delta; the guide may be working now.")


if __name__ == "__main__":
    main()
"""
Instrumented rollout diagnostics, re-running guided rollouts to measure three
things the aggregate metrics cannot distinguish. The companion,
diag_training_logs.py, reads the logged training curves instead.

  REWARD SPREAD OVER THE TRAINING DISTRIBUTION.
    A component can vary over GEOM-Drugs and still be flat over the molecules the
    guide actually sampled while training. If every rollout molecule scored about
    the same, the terminal loss had no gradient regardless of whether it
    converged. Rolls both the guide and the prior and reports the reward spread
    over what each actually samples, which is the distribution the training
    gradient saw.

  RESIDUAL AGAINST LOGIT HEADROOM.
    Per guide, the distribution of residual norms over sampled states, against
    the headroom: how large the residual would have to be to change the median
    decision, i.e. the prior's top-1 margin there. A residual far below the
    headroom everywhere is structurally under-scaled, which more training does
    not fix.

  TRAINING VERSUS EVALUATION SAMPLING SETTINGS.
    Rolling out at sample_temp 2 and rand_eps 0.2 trains the guide largely on
    near-random trajectories. Compares the reward the policy reaches under the
    training settings against the evaluation settings (temperature 1, no
    epsilon). Much lower reward under the training settings means the guide
    learned from a region away from where the reward is, so its terminal targets
    were mostly flat.

Usage:
    python ablations/diag_rollout.py \
        --guide_ckpts "...c0,...c1,...c2,...c3" --guide_labels "c0,c1,c2,c3" \
        --eval_rewards "gcomp:osimertinib:0=c0,gcomp:osimertinib:1=c1,gcomp:osimertinib:2=c2,gcomp:osimertinib:3=c3" \
        --route policy --n_samples 1000 \
        --train_temp 2.0 --train_eps 0.2 --out_dir .../rollout-diag
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("medium")

from chem import Molecule, GEN, STOP, PAD, QM9_MASK
from gflow_multi import MultiConfig, Composer


def _mask_for(mask_atoms, device):
    mask = torch.ones(128, dtype=torch.bool, device=device)
    mask[GEN] = False
    mask[PAD] = False
    if mask_atoms == "qm9":
        mask = QM9_MASK.to(device)
    return mask


@torch.no_grad()
def rollout_guide(comp, gi, n, temp, eps, collect_resid=False):
    """Sample the single guide's policy at given temp/eps; optionally collect
    residual norms and prior top-1 gaps over the states visited."""
    prior = comp.prior
    device = comp.device
    mask = _mask_for(comp.cfg.mask_atoms, device)
    NEG = -1e9
    guide = comp.guides[gi] if gi is not None else None
    uniform = mask.float(); uniform = uniform / uniform.sum()

    mols_all, resid_norms, gaps = [], [], []
    done = 0
    while done < n:
        b = min(comp.cfg.chunk, n - done)
        atoms = torch.full((b, 1), GEN, dtype=torch.long, device=device)
        coords = torch.zeros(b, 1, 3, device=device)
        stop_mask = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(comp.cfg.max_len):
            idx = torch.arange(atoms.shape[1], device=device).expand(b, -1)
            seq = prior.encode1(idx, atoms, coords)
            h = seq[:, -1, :]
            pl = prior.proj_logits(h)
            if guide is None:
                guided = pl.float().masked_fill(~mask, NEG)
            else:
                resid = guide(h)
                guided = (pl + resid).float().masked_fill(~mask, NEG)
                if collect_resid:
                    rn = resid.masked_fill(~mask, 0.0).norm(dim=-1)
                    top2 = torch.topk(pl.float().masked_fill(~mask, NEG), 2, -1).values
                    resid_norms.append(rn.cpu().numpy())
                    gaps.append((top2[:, 0] - top2[:, 1]).cpu().numpy())
            behav = F.softmax(guided / temp, -1)
            if eps > 0:
                behav = (1 - eps) * behav + eps * uniform
            nxt = torch.multinomial(behav, 1)
            atoms = torch.cat([atoms, nxt], 1)
            stop_mask = stop_mask | (nxt.squeeze(-1) == STOP)
            if stop_mask.all():
                break
            x = prior.encode2(atoms[:, 1:], seq)[:, -1, :]
            nc, _ = prior.sample_coord(x, device=device, num_steps=comp.cfg.diff_steps)
            coords = torch.cat([coords, nc.view(b, 1, 3)], 1)
        mols_all.extend(Molecule(atoms=atoms[:, 1:], coords=coords[:, 1:]).to("cpu").unbatch())
        done += b
    extra = {}
    if collect_resid and resid_norms:
        extra["resid_norm"] = np.concatenate(resid_norms)
        extra["prior_gap"] = np.concatenate(gaps)
    return mols_all, extra


def spread_stats(vals):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std()),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "iqr": float(np.percentile(v, 75) - np.percentile(v, 25))}


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
    ap.add_argument("--train_temp", type=float, default=2.0)
    ap.add_argument("--train_eps", type=float, default=0.2)
    args = ap.parse_args()

    cfg = MultiConfig(**{k: getattr(args, k) for k in MultiConfig.__dataclass_fields__})
    os.makedirs(cfg.out_dir, exist_ok=True)
    comp = Composer(cfg)
    n = cfg.n_samples
    # score each guide on its OWN component (i-th eval reward), else first reward
    rewards = comp.eval_rewards
    report = {"config": {"route": cfg.route, "train_temp": args.train_temp,
                         "train_eps": args.train_eps}}

    for gi, lab in enumerate(comp.labels):
        rname, rfn = rewards[gi] if gi < len(rewards) else rewards[0]
        entry = {"scored_on": rname}

        # M2 + M4: reward spread at EVAL vs TRAINING sampling settings
        mols_eval, extra = rollout_guide(comp, gi, n, temp=1.0, eps=0.0, collect_resid=True)
        r_eval = np.array([rfn(m) for m in mols_eval], float)
        mols_tr, _ = rollout_guide(comp, gi, max(n // 2, 300),
                                   temp=args.train_temp, eps=args.train_eps)
        r_tr = np.array([rfn(m) for m in mols_tr], float)

        entry["reward_spread_eval"] = spread_stats(r_eval)      # M2: is reward flat?
        entry["reward_spread_train_settings"] = spread_stats(r_tr)  # M4
        entry["reward_mean_eval_minus_train"] = (
            float(np.nanmean(r_eval) - np.nanmean(r_tr)))       # M4: did training see lower reward?

        # M3: residual norm vs headroom needed to flip
        if "resid_norm" in extra:
            rn = extra["resid_norm"]; gap = extra["prior_gap"]
            entry["residual_norm"] = spread_stats(rn)
            entry["prior_gap"] = spread_stats(gap)
            # crude headroom test: residual per-logit must ~ gap to flip; compare
            # median residual norm to median gap (both are logit-scale quantities)
            med_rn = float(np.median(rn)); med_gap = float(np.median(gap))
            entry["resid_over_gap_median"] = med_rn / max(med_gap, 1e-9)
            entry["headroom_verdict"] = (
                "residual << gap: structurally under-scaled (needs gain/inject-point fix)"
                if med_rn < 0.3 * med_gap else
                "residual comparable to gap: can flip when aligned (training/alignment issue)")

        # verdicts
        v = []
        se = entry["reward_spread_eval"]
        if se.get("std", 9) < 0.05:
            v.append("M2: reward FLAT over sampled mols (no training gradient signal)")
        if entry["reward_mean_eval_minus_train"] > 1.0:
            v.append("M4: training-settings reward much lower (guide trained off-target region)")
        if "resid_over_gap_median" in entry and entry["resid_over_gap_median"] < 0.3:
            v.append("M3: residual too small vs prior gap (saturated-prior ceiling)")
        if not v:
            v.append("no single-mechanism flag; combine with training-log diag")
        entry["verdict"] = "; ".join(v)
        report[lab] = entry

        print(f"\n=== {lab} (scored on {rname}) ===")
        print(f"  reward std (eval sampling)  = {se.get('std', float('nan')):.4f}  "
              f"[flat if <0.05]")
        print(f"  reward mean eval - train    = {entry['reward_mean_eval_minus_train']:+.3f}  "
              f"[>1 => trained off-target]")
        if "resid_over_gap_median" in entry:
            print(f"  median resid / prior gap    = {entry['resid_over_gap_median']:.4f}  "
                  f"[<0.3 => under-scaled]")
        print(f"  VERDICT: {entry['verdict']}")

    with open(os.path.join(cfg.out_dir, "rollout_diag.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] -> {os.path.join(cfg.out_dir, 'rollout_diag.json')}")


if __name__ == "__main__":
    main()
"""
Effect size of a logit-injection guide, measured against the levers that
control it.

The guided sampler puts a little extra mass on the best molecules while the bulk
of the distribution stays at the prior. This harness isolates where that comes
from, reusing the building blocks from gflow.py and gflow_multi.py so the
behaviour matches training.

Four ablations:

  A. RESIDUAL MAGNITUDE. Is guide(h) large enough to move the softmax? Reports
     ||guide(h)|| against ||prior_logits|| and the per-step KL(guided||prior) at
     evaluation time. A residual orders of magnitude below the prior's logits
     leaves the policy at the prior whatever the guide learned.

  B. RESIDUAL SCALE. Multiply the trained residual by a constant factor s at
     sampling time, as a proxy for retraining at higher beta. Effect size rises
     to a maximum near 4x and then falls, with the mean reward shift turning
     negative: past that point the sampler leaves the region where its outputs
     remain valid faster than it gains reward.

  C. SINGLE VERSUS COMPOSED. Each guide's effect size sampled alone, then the
     composed sampler's. Strong singles with a weak composition mean the product
     operator is being dragged toward the prior by its least-active component.

  D. SAMPLING SETTINGS. Re-sample at several (sample_temp, rand_eps) pairs, to
     confirm evaluation-time settings are not themselves washing the guide out.

Usage (the guide flags mirror gflow_multi's):
    python ablate_guide.py \
        --guide_ckpts "logs/.../comp0/last.ckpt,logs/.../comp1/last.ckpt,..." \
        --guide_labels "c0,c1,c2,c3" \
        --route flow --n_samples 1000 --residual_scales "0.5,1,2,4,8"

python ablate_guide.py --guide_ckpts "logs/quetzal-gfn/sweep-osim-base-db-replay_off-b10/checkpoints/last.ckpt" --guide_labels "db" --route flow --n_samples 1000 --residual_scales "0.5,2,8"

Outputs: a JSON + a bar/line summary PNG per ablation in --out_dir.
"""
import os
import json
import argparse
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("medium")

from chem import Molecule, GEN, STOP, PAD, QM9_MASK
from metrics import compute_valid_unique
from reward_fn import build_reward
from gflow_multi import (
    MultiConfig, Composer, generate_composed,
)


# ------------------------------------------------------------------ helpers
def _mask_for(mask_atoms, device):
    if mask_atoms is None or mask_atoms == "None":
        mask = torch.ones(128, dtype=torch.bool, device=device)
        mask[GEN] = False; mask[PAD] = False
    elif mask_atoms == "qm9":
        mask = QM9_MASK.to(device)
    else:
        mask = torch.ones(128, dtype=torch.bool, device=device)
        mask[GEN] = False; mask[PAD] = False
    return mask


@torch.no_grad()
def residual_diagnostics(comp, n_states=512):
    """Ablation A: at eval-time states drawn from the PRIOR, measure per guide
    ||guide(h)|| vs ||prior_logits||, and KL(softmax(prior+guide) || softmax(prior)).
    A guide that can't move the softmax has KL ~ 0 and residual_ratio << 1.
    """
    prior = comp.prior
    device = comp.device
    mask = _mask_for(comp.cfg.mask_atoms, device)
    NEG = -1e9

    # roll the PRIOR forward a few steps to collect realistic hidden states h
    bsz = min(128, n_states)
    atoms = torch.full((bsz, 1), GEN, dtype=torch.long, device=device)
    coords = torch.zeros(bsz, 1, 3, device=device)
    H = []
    for t in range(8):
        idx = torch.arange(atoms.shape[1], device=device).expand(bsz, -1)
        seq = prior.encode1(idx, atoms, coords)
        h = seq[:, -1, :]
        H.append(h)
        prior_logits = prior.proj_logits(h).float().masked_fill(~mask, NEG)
        nxt = torch.multinomial(F.softmax(prior_logits, -1), 1)
        if (nxt.squeeze(-1) == STOP).all():
            break
        atoms = torch.cat([atoms, nxt], 1)
        x = prior.encode2(atoms[:, 1:], seq)[:, -1, :]
        nc, _ = prior.sample_coord(x, device=device, num_steps=comp.cfg.diff_steps)
        coords = torch.cat([coords, nc.view(bsz, 1, 3)], 1)
    H = torch.cat(H, 0)  # [~bsz*T, d]

    out = {}
    prior_logits = prior.proj_logits(H).float().masked_fill(~mask, NEG)
    lp_prior = F.log_softmax(prior_logits, -1)
    prior_norm = prior_logits.masked_fill(~mask, 0.0).norm(dim=-1).mean().item()
    for lab, g in zip(comp.labels, comp.guides):
        resid = g(H)                                    # [N, V]
        guided = (prior_logits + resid).float().masked_fill(~mask, NEG)
        lp_guided = F.log_softmax(guided, -1)
        # KL(guided || prior), averaged over states
        kl = (lp_guided.exp() * (lp_guided - lp_prior)).sum(-1)
        kl = kl[torch.isfinite(kl)]
        resid_norm = resid.masked_fill(~mask, 0.0).norm(dim=-1).mean().item()
        out[lab] = {
            "residual_norm": resid_norm,
            "prior_logit_norm": prior_norm,
            "residual_ratio": resid_norm / max(prior_norm, 1e-9),
            "mean_KL_guided_vs_prior": float(kl.mean().item()),
            "median_KL": float(kl.median().item()),
        }
    return out


def effect_size(base_logr, guided_logr):
    """Scalar summaries of how far the guided reward dist moved off base."""
    b = np.asarray(base_logr, float); g = np.asarray(guided_logr, float)
    b = b[np.isfinite(b)]; g = g[np.isfinite(g)]
    # mean shift, plus a distributional distance (1D Wasserstein via sorted diff)
    def w1(a, c):
        a = np.sort(a); c = np.sort(c)
        n = min(len(a), len(c))
        if n == 0:
            return 0.0
        aa = np.interp(np.linspace(0, 1, 512), np.linspace(0, 1, len(a)), a)
        cc = np.interp(np.linspace(0, 1, 512), np.linspace(0, 1, len(c)), c)
        return float(np.mean(np.abs(aa - cc)))
    return {
        "mean_shift": float(g.mean() - b.mean()),
        "median_shift": float(np.median(g) - np.median(b)),
        "top10pct_shift": float(np.mean(np.sort(g)[-max(1, len(g)//10):])
                                - np.mean(np.sort(b)[-max(1, len(b)//10):])),
        "wasserstein1": w1(b, g),
    }


@torch.no_grad()
def sample_with_residual_scale(comp, scale, n, single_idx=None):
    """Sample the composed (or a single) guide with every guide residual scaled
    by `scale`. Implemented by temporarily wrapping each guide's forward.

    scale acts as a proxy for tilt strength: the trained guide residual r(h) is
    what produces p_prior * exp(r). Scaling r by s ~ raising the effective tilt.
    """
    class _Scaled(torch.nn.Module):
        def __init__(self, g, s): super().__init__(); self.g = g; self.s = s
        def forward(self, h): return self.s * self.g(h)

    if single_idx is None:
        guides = [ _Scaled(g, scale) for g in comp.guides ]
        log_weights = comp.log_weights
        log_Z = comp.log_Z
        train_betas = comp.train_betas
        flow_heads = comp.flow_heads
    else:
        guides = [ _Scaled(comp.guides[single_idx], scale) ]
        log_weights = torch.zeros(1)
        log_Z = comp.log_Z[single_idx:single_idx+1]
        train_betas = comp.train_betas[single_idx:single_idx+1]
        flow_heads = [comp.flow_heads[single_idx]]

    mols_all = []
    done = 0
    while done < n:
        b = min(comp.cfg.chunk, n - done)
        mols, _ = generate_composed(
            comp.prior, guides, log_weights, log_Z, train_betas,
            comp.cfg.operator, comp.cfg.product_kind, comp.cfg.compose_space,
            comp.compose_beta, b, comp.cfg.max_len, comp.device,
            mask_atoms=comp.cfg.mask_atoms, sample_temp=comp.cfg.sample_temp,
            rand_eps=comp.cfg.rand_eps, flow_heads=flow_heads,
            route=comp.cfg.route, num_steps=comp.cfg.diff_steps)
        mols_all.extend(mols.unbatch())
        done += b
    return mols_all


def score_primary(comp, mols):
    """Score on the FIRST eval reward (the aggregate MPO by convention)."""
    name, fn = comp.eval_rewards[0]
    return name, np.asarray([fn(m) for m in mols], float)


def main():
    ap = argparse.ArgumentParser()
    # mirror the compose flags we need
    for f in MultiConfig.__dataclass_fields__.values():
        if isinstance(f.default, bool):
            ap.add_argument(f"--{f.name}", dest=f.name, action="store_true" if not f.default else "store_false")
            ap.set_defaults(**{f.name: f.default})
        else:
            t = type(f.default) if f.default is not None else str
            ap.add_argument(f"--{f.name}", type=t, default=f.default)
    ap.add_argument("--residual_scales", default="0.5,1,2,4,8",
                    help="scale factors applied to the guide residual (beta proxy)")
    ap.add_argument("--do_singles", action="store_true",
                    help="also measure each single guide's effect size (dilution test)")
    args = ap.parse_args()

    cfg = MultiConfig(**{k: getattr(args, k) for k in MultiConfig.__dataclass_fields__})
    os.makedirs(cfg.out_dir, exist_ok=True)
    comp = Composer(cfg)
    n = cfg.n_samples
    scales = [float(x) for x in args.residual_scales.split(",")]

    report = {"config": {k: getattr(cfg, k) for k in
                         ["route","operator","product_kind","weights","train_betas",
                          "compose_space","sample_temp","rand_eps","n_samples"]}}

    # ---- baseline: pure prior ----
    print("[base] sampling prior ...")
    base_mols = comp.sample_base(n)
    rname, base_logr = score_primary(comp, base_mols)
    bvalid, buniq = compute_valid_unique(base_mols)
    report["reward_scored"] = rname
    report["base"] = {"mean_logr": float(np.mean(base_logr)),
                      "validity": bvalid, "uniqueness": buniq}
    print(f"[base] mean_logr={np.mean(base_logr):.3f} valid={bvalid:.3f}")

    # ---- Ablation A: residual magnitude ----
    print("[A] residual diagnostics ...")
    report["A_residual"] = residual_diagnostics(comp)
    for lab, d in report["A_residual"].items():
        print(f"   {lab}: resid/prior={d['residual_ratio']:.4f}  "
              f"KL(guided||prior)={d['mean_KL_guided_vs_prior']:.4f}")

    # ---- Ablation B: residual-scale (beta proxy) sweep on the COMPOSED guide ----
    print("[B] residual-scale sweep (composed) ...")
    report["B_scale_sweep"] = {}
    for s in scales:
        mols = sample_with_residual_scale(comp, s, n)
        _, logr = score_primary(comp, mols)
        es = effect_size(base_logr, logr)
        v, u = compute_valid_unique(mols)
        es.update({"mean_logr": float(np.mean(logr)), "validity": v, "uniqueness": u})
        report["B_scale_sweep"][f"scale_{s:g}"] = es
        print(f"   scale={s:g}: mean_shift={es['mean_shift']:+.3f}  "
              f"W1={es['wasserstein1']:.3f}  valid={v:.3f} uniq={u:.3f}")

    # ---- Ablation C: single-guide effect sizes (dilution test) ----
    if args.do_singles:
        print("[C] single-guide effect sizes ...")
        report["C_singles"] = {}
        for i, lab in enumerate(comp.labels):
            mols = sample_with_residual_scale(comp, 1.0, n, single_idx=i)
            _, logr = score_primary(comp, mols)
            es = effect_size(base_logr, logr)
            report["C_singles"][lab] = es
            print(f"   {lab}: mean_shift={es['mean_shift']:+.3f}  W1={es['wasserstein1']:.3f}")

    # ---- Ablation D: temp / eps ----
    print("[D] temp/eps effect ...")
    report["D_tempeps"] = {}
    for st, re_ in [(1.0, 0.0), (1.0, 0.1), (2.0, 0.0), (2.0, 0.2)]:
        old_t, old_e = comp.cfg.sample_temp, comp.cfg.rand_eps
        comp.cfg.sample_temp, comp.cfg.rand_eps = st, re_
        mols = sample_with_residual_scale(comp, 1.0, max(n // 2, 200))
        comp.cfg.sample_temp, comp.cfg.rand_eps = old_t, old_e
        _, logr = score_primary(comp, mols)
        es = effect_size(base_logr, logr)
        report["D_tempeps"][f"temp{st}_eps{re_}"] = es
        print(f"   temp={st} eps={re_}: mean_shift={es['mean_shift']:+.3f}  W1={es['wasserstein1']:.3f}")

    # ---- save + plot ----
    with open(os.path.join(cfg.out_dir, "ablation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        # B: effect size vs scale
        ss = [float(k.split("_")[1]) for k in report["B_scale_sweep"]]
        w1 = [report["B_scale_sweep"][k]["wasserstein1"] for k in report["B_scale_sweep"]]
        ms = [report["B_scale_sweep"][k]["mean_shift"] for k in report["B_scale_sweep"]]
        axes[0].plot(ss, w1, "o-", label="Wasserstein-1(guided,base)")
        axes[0].plot(ss, ms, "s--", label="mean logR shift")
        axes[0].set_xlabel("residual scale (beta proxy)")
        axes[0].set_ylabel("effect size")
        axes[0].set_title("B: does stronger tilt separate from prior?")
        axes[0].axhline(0, color="k", lw=0.6); axes[0].legend(fontsize=8)
        # A: residual ratio + KL per guide
        labs = list(report["A_residual"].keys())
        ratios = [report["A_residual"][l]["residual_ratio"] for l in labs]
        kls = [report["A_residual"][l]["mean_KL_guided_vs_prior"] for l in labs]
        xp = np.arange(len(labs))
        axes[1].bar(xp - 0.2, ratios, 0.4, label="resid/prior norm")
        axes[1].bar(xp + 0.2, kls, 0.4, label="KL(guided||prior)")
        axes[1].set_xticks(xp); axes[1].set_xticklabels(labs, rotation=30, fontsize=8)
        axes[1].set_title("A: is the residual big enough to steer?")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(cfg.out_dir, "ablation_summary.png")
        fig.savefig(p, dpi=130); plt.close(fig)
        print(f"[plot] {p}")
    except Exception as e:
        print(f"[plot] failed: {e}")

    print(f"[done] report -> {os.path.join(cfg.out_dir, 'ablation_report.json')}")


if __name__ == "__main__":
    main()
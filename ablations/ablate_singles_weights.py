"""
ablate_singles_weights.py -- two ablations:

  #3 SINGLE-GUIDE EFFECT (fixed).
     The earlier C_singles numbers were degenerate (identical -2.679 across all
     guides) because the single-guide path went through the HARMONIC operator with
     one component: harmonic on a single guide gives lognum == logden -> comp == 0
     -> UNIFORM distribution -> garbage samples. Here we sample each guide alone
     with a correct single-guide policy (prior + guide, softmax-normalized), no
     operator, so the per-guide effect size is real.

  #4 WEIGHT-SKEW SWEEP.
     Equal 0.25 weights let three near-prior guides dilute the one that steers.
     Sweep composition weights (e.g. all mass on c3, or skewed mixes) and measure
     composed effect size + validity/uniqueness, to quantify how much dropping the
     dead guides recovers.

Usage:
    python ablate_singles_weights.py \
        --guide_ckpts "...c0,...c1,...c2,...c3" --guide_labels "c0,c1,c2,c3" \
        --route policy --product_kind poe --train_betas "50,50,50,50" \
        --n_samples 1000 \
        --weight_sets "0.25,0.25,0.25,0.25;1,0,0,0;0,0,0,1;0.1,0.1,0.1,0.7;0.4,0,0,0.6"
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("medium")

from chem import Molecule, GEN, STOP, PAD, QM9_MASK
from gflow_multi import MultiConfig, Composer, generate_composed


def _mask_for(mask_atoms, device):
    mask = torch.ones(128, dtype=torch.bool, device=device)
    mask[GEN] = False
    mask[PAD] = False
    if mask_atoms == "qm9":
        mask = QM9_MASK.to(device)
    return mask


@torch.no_grad()
def sample_single_guide(comp, gi, n):
    """Correct single-guide sampler: policy = softmax(prior_logits + guide(h)),
    normalized directly. NO composition operator (harmonic/poe on one element is
    degenerate). This is what the guide alone actually samples.
    """
    prior = comp.prior
    device = comp.device
    mask = _mask_for(comp.cfg.mask_atoms, device)
    NEG = -1e9
    guide = comp.guides[gi]
    max_len = comp.cfg.max_len
    st = comp.cfg.sample_temp

    mols_all = []
    done = 0
    while done < n:
        b = min(comp.cfg.chunk, n - done)
        atoms = torch.full((b, 1), GEN, dtype=torch.long, device=device)
        coords = torch.zeros(b, 1, 3, device=device)
        stop_mask = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_len):
            idx = torch.arange(atoms.shape[1], device=device).expand(b, -1)
            seq = prior.encode1(idx, atoms, coords)
            h = seq[:, -1, :]
            pl = prior.proj_logits(h)
            guided = (pl + guide(h)).float().masked_fill(~mask, NEG)
            behav = F.softmax(guided / st, -1)
            nxt = torch.multinomial(behav, 1)
            atoms = torch.cat([atoms, nxt], 1)
            stop_mask = stop_mask | (nxt.squeeze(-1) == STOP)
            if stop_mask.all():
                break
            x = prior.encode2(atoms[:, 1:], seq)[:, -1, :]
            nc, _ = prior.sample_coord(x, device=device, num_steps=comp.cfg.diff_steps)
            coords = torch.cat([coords, nc.view(b, 1, 3)], 1)
        m = Molecule(atoms=atoms[:, 1:], coords=coords[:, 1:]).to("cpu").unbatch()
        mols_all.extend(m)
        done += b
    return mols_all


def effect_size(base, guided):
    b = np.asarray(base, float); g = np.asarray(guided, float)
    b = b[np.isfinite(b)]; g = g[np.isfinite(g)]
    def w1(a, c):
        if len(a) == 0 or len(c) == 0:
            return 0.0
        aa = np.interp(np.linspace(0, 1, 512), np.linspace(0, 1, len(a)), np.sort(a))
        cc = np.interp(np.linspace(0, 1, 512), np.linspace(0, 1, len(c)), np.sort(c))
        return float(np.mean(np.abs(aa - cc)))
    return {"mean_shift": float(g.mean() - b.mean()),
            "median_shift": float(np.median(g) - np.median(b)),
            "wasserstein1": w1(b, g)}


@torch.no_grad()
def sample_composed_weights(comp, weights, n):
    """Composed sample with a given weight vector (rest of config unchanged)."""
    lw = torch.tensor(np.log(np.clip(weights, 1e-12, None)), dtype=torch.float32)
    mols_all = []
    done = 0
    while done < n:
        b = min(comp.cfg.chunk, n - done)
        mols, _ = generate_composed(
            comp.prior, comp.guides, lw, comp.log_Z, comp.train_betas,
            comp.cfg.operator, comp.cfg.product_kind, comp.cfg.compose_space,
            comp.compose_beta, b, comp.cfg.max_len, comp.device,
            mask_atoms=comp.cfg.mask_atoms, sample_temp=comp.cfg.sample_temp,
            rand_eps=comp.cfg.rand_eps, flow_heads=comp.flow_heads,
            route=comp.cfg.route, num_steps=comp.cfg.diff_steps)
        mols_all.extend(mols.unbatch())
        done += b
    return mols_all


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
    ap.add_argument("--weight_sets",
                    default="0.25,0.25,0.25,0.25;1,0,0,0;0,0,0,1;0.1,0.1,0.1,0.7",
                    help="semicolon-separated weight vectors for the #4 sweep")
    args = ap.parse_args()

    cfg = MultiConfig(**{k: getattr(args, k) for k in MultiConfig.__dataclass_fields__})
    os.makedirs(cfg.out_dir, exist_ok=True)
    comp = Composer(cfg)
    from metrics import compute_valid_unique
    n = cfg.n_samples
    report = {"config": {"route": cfg.route, "product_kind": cfg.product_kind,
                         "train_betas": cfg.train_betas}}

    name0, fn0 = comp.eval_rewards[0]
    report["reward_scored"] = name0

    print("[base] sampling prior ...")
    base = comp.sample_base(n)
    base_r = np.array([fn0(m) for m in base], float)
    bv, bu = compute_valid_unique(base)
    report["base"] = {"mean_logr": float(np.mean(base_r)), "validity": bv, "uniqueness": bu}
    print(f"[base] mean_logr={np.mean(base_r):.3f} valid={bv:.3f}")

    # ---- #3 single-guide effect (FIXED) ----
    print("\n[#3] single-guide effect sizes (correct single-guide sampler) ...")
    report["singles"] = {}
    for i, lab in enumerate(comp.labels):
        mols = sample_single_guide(comp, i, n)
        r = np.array([fn0(m) for m in mols], float)
        es = effect_size(base_r, r)
        v, u = compute_valid_unique(mols)
        es.update({"mean_logr": float(np.mean(r)), "validity": v, "uniqueness": u})
        report["singles"][lab] = es
        print(f"   {lab}: mean_shift={es['mean_shift']:+.3f}  W1={es['wasserstein1']:.3f}  "
              f"valid={v:.3f} uniq={u:.3f}")

    # ---- #4 weight-skew sweep ----
    print("\n[#4] weight-skew sweep (composed) ...")
    report["weight_sweep"] = {}
    for ws in args.weight_sets.split(";"):
        ws = ws.strip()
        if not ws:
            continue
        w = np.array([float(x) for x in ws.split(",")], float)
        if len(w) != len(comp.guides):
            print(f"   [skip] {ws}: has {len(w)} weights, need {len(comp.guides)}")
            continue
        mols = sample_composed_weights(comp, w, n)
        r = np.array([fn0(m) for m in mols], float)
        es = effect_size(base_r, r)
        v, u = compute_valid_unique(mols)
        es.update({"mean_logr": float(np.mean(r)), "validity": v, "uniqueness": u})
        report["weight_sweep"][ws] = es
        print(f"   w=[{ws}]: mean_shift={es['mean_shift']:+.3f}  W1={es['wasserstein1']:.3f}  "
              f"valid={v:.3f} uniq={u:.3f}")

    with open(os.path.join(cfg.out_dir, "singles_weights_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
        labs = list(report["singles"].keys())
        w1s = [report["singles"][l]["wasserstein1"] for l in labs]
        ms = [report["singles"][l]["mean_shift"] for l in labs]
        x = np.arange(len(labs))
        ax1.bar(x - 0.2, w1s, 0.4, label="W1(single,base)")
        ax1.bar(x + 0.2, ms, 0.4, label="mean shift")
        ax1.set_xticks(x); ax1.set_xticklabels(labs)
        ax1.axhline(0, color="k", lw=0.6)
        ax1.set_title("#3: single-guide effect (which guides actually steer?)")
        ax1.legend(fontsize=8)
        ws_labels = list(report["weight_sweep"].keys())
        ws_w1 = [report["weight_sweep"][w]["wasserstein1"] for w in ws_labels]
        ws_uq = [report["weight_sweep"][w]["uniqueness"] for w in ws_labels]
        xw = np.arange(len(ws_labels))
        ax2.bar(xw - 0.2, ws_w1, 0.4, label="W1(composed,base)")
        ax2.bar(xw + 0.2, ws_uq, 0.4, label="uniqueness")
        ax2.set_xticks(xw); ax2.set_xticklabels(ws_labels, rotation=25, fontsize=7)
        ax2.set_title("#4: weight-skew (does dropping dead guides recover effect?)")
        ax2.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(cfg.out_dir, "singles_weights_summary.png")
        fig.savefig(p, dpi=130); plt.close(fig)
        print(f"[plot] {p}")
    except Exception as e:
        print(f"[plot] failed: {e}")

    print(f"[done] -> {os.path.join(cfg.out_dir, 'singles_weights_report.json')}")


if __name__ == "__main__":
    main()
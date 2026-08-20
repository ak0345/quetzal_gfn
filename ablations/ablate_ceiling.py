"""
Two diagnostics that share one rollout of the frozen prior.

  MARGIN BINNING.
     At each decision the prior has a top-1 margin, logit_1 - logit_2. A residual
     of a given norm can only change the decision where that margin is small.
     This bins states by the prior's margin and computes the sampled-flip rate
     within each bin, per guide. The flip rate falls to zero above a margin of
     roughly 4, while the majority of decisions lie above 8.

     Output: a flip-rate against margin curve per guide, plus the fraction of
     decisions falling in each bin.

  COMPONENT VARIANCE.
     Scores frozen-prior samples on each leaf component of the MPO objective. A
     component with near-zero variance over reachable molecules is a dead axis:
     no guide can steer it, so a flat curve there is expected rather than a
     training failure. A component with real variance but a flat guide is
     genuinely under-trained. This separates the two, and is why the benchmark
     runs train against the assembled objective rather than its components.

Usage (the guide flags mirror gflow_multi's; give the per-component eval
rewards):
    python ablate_ceiling.py \
        --guide_ckpts "...c0/last.ckpt,...c1,...c2,...c3" \
        --guide_labels "c0,c1,c2,c3" \
        --eval_rewards "gcomp:osimertinib:0=c0,gcomp:osimertinib:1=c1,gcomp:osimertinib:2=c2,gcomp:osimertinib:3=c3" \
        --route policy --n_traj 400 --n_score 1500
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


# ------------------------------------------------------------ #2 saturation bins
@torch.no_grad()
def saturation_binning(comp, n_traj=400, temp=1.0,
                       gap_edges=(0, 1, 2, 4, 8, 16, 32, 1e9)):
    """Roll prior trajectories; per guide, bucket every decision by the prior's
    top-1 logit gap and record the paired-RNG sampled-flip rate in each bucket.
    Also record how the guide residual norm compares to the gap it would need to
    overcome.
    """
    prior = comp.prior
    device = comp.device
    mask = _mask_for(comp.cfg.mask_atoms, device)
    NEG = -1e9
    max_steps = min(comp.cfg.max_len, 64)
    bsz = min(comp.cfg.chunk, n_traj)
    labels, guides = comp.labels, comp.guides
    nb = len(gap_edges) - 1

    flips = {lab: np.zeros(nb) for lab in labels}
    counts = {lab: np.zeros(nb) for lab in labels}
    resid_norm_in_bin = {lab: np.zeros(nb) for lab in labels}

    done = 0
    while done < n_traj:
        b = min(bsz, n_traj - done)
        atoms = torch.full((b, 1), GEN, dtype=torch.long, device=device)
        coords = torch.zeros(b, 1, 3, device=device)
        stop_mask = torch.zeros(b, dtype=torch.bool, device=device)

        for t in range(max_steps):
            idx = torch.arange(atoms.shape[1], device=device).expand(b, -1)
            seq = prior.encode1(idx, atoms, coords)
            h = seq[:, -1, :]
            prior_logits = prior.proj_logits(h).float().masked_fill(~mask, NEG)
            p_prior = F.softmax(prior_logits / temp, -1)
            top2 = torch.topk(prior_logits, 2, dim=-1).values
            gap = (top2[:, 0] - top2[:, 1])                      # [b]
            gap_bin = torch.bucketize(gap.cpu(),
                                      torch.tensor(gap_edges[1:-1], dtype=torch.float))
            u = torch.rand(b, 1, device=device)
            next_prior = (u < torch.cumsum(p_prior, -1)).float().argmax(-1)
            alive = (~stop_mask)

            for gi, lab in enumerate(labels):
                resid = guides[gi](h)
                guided = (prior_logits + resid).float().masked_fill(~mask, NEG)
                p_g = F.softmax(guided / temp, -1)
                next_g = (u < torch.cumsum(p_g, -1)).float().argmax(-1)
                flip = (next_g != next_prior) & alive
                rn = resid.masked_fill(~mask, 0.0).norm(dim=-1)  # [b]
                gb = gap_bin.numpy()
                al = alive.cpu().numpy()
                fl = flip.cpu().numpy()
                rnn = rn.cpu().numpy()
                for k in range(nb):
                    sel = (gb == k) & al
                    counts[lab][k] += sel.sum()
                    flips[lab][k] += (fl & sel).sum()
                    resid_norm_in_bin[lab][k] += rnn[sel].sum()

            atoms = torch.cat([atoms, next_prior.unsqueeze(1)], 1)
            stop_mask = stop_mask | (next_prior == STOP)
            if stop_mask.all():
                break
            x = prior.encode2(atoms[:, 1:], seq)[:, -1, :]
            nc, _ = prior.sample_coord(x, device=device, num_steps=comp.cfg.diff_steps)
            coords = torch.cat([coords, nc.view(b, 1, 3)], 1)
        done += b

    out = {"gap_bins": [f"[{gap_edges[i]:g},{gap_edges[i+1]:g})" for i in range(nb)]}
    # overall: fraction of decisions in each gap bin (guide-independent)
    total_counts = counts[labels[0]]
    tot = total_counts.sum()
    out["decision_frac_by_gap"] = (total_counts / max(tot, 1)).tolist()
    out["per_guide"] = {}
    for lab in labels:
        c = counts[lab]
        fr = flips[lab] / np.maximum(c, 1)
        rn = resid_norm_in_bin[lab] / np.maximum(c, 1)
        out["per_guide"][lab] = {
            "flip_rate_by_gap": fr.tolist(),
            "mean_resid_norm_by_gap": rn.tolist(),
            "overall_flip_rate": float(flips[lab].sum() / max(c.sum(), 1)),
            # flippable fraction = decisions where gap is small enough that the
            # guide actually flipped at least sometimes
            "frac_decisions_gap_lt_4": float(c[:3].sum() / max(c.sum(), 1)),
        }
    return out


# ---------------------------------------------------- #5 component variance
@torch.no_grad()
def component_variance(comp, n_score=1500):
    """Score Quetzal BASE samples on each eval reward (the per-component scorers)
    and report variance. Near-zero variance -> DEAD AXIS (unsteerable);
    real variance + a KL~0 guide -> UNDER-TRAINED guide.
    """
    mols = comp.sample_base(n_score)
    out = {}
    for name, fn in comp.eval_rewards:
        vals = np.array([fn(m) for m in mols], float)
        finite = vals[np.isfinite(vals)]
        # exclude the invalid floor when judging variance of the VALID range
        valid = finite[finite > comp.cfg.invalid_logr + 0.1]
        out[name] = {
            "n": int(len(finite)),
            "n_valid": int(len(valid)),
            "mean_logr": float(np.mean(finite)) if len(finite) else float("nan"),
            "std_logr_all": float(np.std(finite)) if len(finite) else float("nan"),
            "std_logr_valid": float(np.std(valid)) if len(valid) else float("nan"),
            "min": float(np.min(finite)) if len(finite) else float("nan"),
            "max": float(np.max(finite)) if len(finite) else float("nan"),
            "verdict": ("DEAD AXIS (std<0.05 over reachable mols; unsteerable)"
                        if len(valid) and np.std(valid) < 0.05
                        else "has variance (steerable in principle)"),
        }
    return out


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
    ap.add_argument("--n_score", type=int, default=1500)
    ap.add_argument("--sat_temp", type=float, default=1.0)
    args = ap.parse_args()

    cfg = MultiConfig(**{k: getattr(args, k) for k in MultiConfig.__dataclass_fields__})
    os.makedirs(cfg.out_dir, exist_ok=True)
    comp = Composer(cfg)
    report = {}

    print(f"[#5] scoring {args.n_score} base Quetzal samples per component ...")
    report["component_variance"] = component_variance(comp, n_score=args.n_score)
    print(f"\n{'component':<22} {'std_valid':>10} {'mean':>8} {'verdict'}")
    for name, d in report["component_variance"].items():
        print(f"{name:<22} {d['std_logr_valid']:>10.4f} {d['mean_logr']:>8.3f}  {d['verdict']}")

    print(f"\n[#2] prior-saturation binning over {args.n_traj} trajectories "
          f"(temp={args.sat_temp}) ...")
    report["saturation"] = saturation_binning(comp, n_traj=args.n_traj, temp=args.sat_temp)

    bins = report["saturation"]["gap_bins"]
    dfrac = report["saturation"]["decision_frac_by_gap"]
    print("\ndecision fraction by prior top-1 gap:")
    for bn, fr in zip(bins, dfrac):
        print(f"   gap {bn:<12} {fr*100:5.1f}% of decisions")
    print("\nsampled-flip rate by prior gap (per guide):")
    print(f"{'guide':<8} " + " ".join(f"{bn:>12}" for bn in bins))
    for lab, d in report["saturation"]["per_guide"].items():
        row = " ".join(f"{fr:>12.3f}" for fr in d["flip_rate_by_gap"])
        print(f"{lab:<8} {row}")
    print("\nflippable fraction (gap<4) & overall flip rate per guide:")
    for lab, d in report["saturation"]["per_guide"].items():
        print(f"   {lab}: frac_gap<4={d['frac_decisions_gap_lt_4']:.3f}  "
              f"overall_flip={d['overall_flip_rate']:.4f}")

    with open(os.path.join(cfg.out_dir, "ceiling_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
        x = np.arange(len(bins))
        for lab, d in report["saturation"]["per_guide"].items():
            ax1.plot(x, d["flip_rate_by_gap"], "o-", label=lab)
        ax1.set_xticks(x); ax1.set_xticklabels(bins, rotation=35, fontsize=8)
        ax1.set_xlabel("prior top-1 logit gap"); ax1.set_ylabel("sampled-flip rate")
        ax1.set_title("#2: guide only steers where prior is UNsure (small gap)")
        ax1.legend(fontsize=8)
        ax1b = ax1.twinx()
        ax1b.bar(x, dfrac, alpha=0.15, color="gray")
        ax1b.set_ylabel("frac of decisions (gray)")
        names = list(report["component_variance"].keys())
        stds = [report["component_variance"][n]["std_logr_valid"] for n in names]
        ax2.bar(range(len(names)), stds, color="#4C72B0")
        ax2.axhline(0.05, color="r", ls="--", lw=1, label="dead-axis threshold")
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=30, fontsize=8)
        ax2.set_ylabel("std of log-reward (valid)")
        ax2.set_title("#5: component variance on Quetzal (dead vs steerable)")
        ax2.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(cfg.out_dir, "ceiling_summary.png")
        fig.savefig(p, dpi=130); plt.close(fig)
        print(f"[plot] {p}")
    except Exception as e:
        print(f"[plot] failed: {e}")

    print(f"[done] -> {os.path.join(cfg.out_dir, 'ceiling_report.json')}")


if __name__ == "__main__":
    main()
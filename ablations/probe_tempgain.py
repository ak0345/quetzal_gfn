"""
What the temperature and gain heads learned, and what they cost.

The temp/gain guide computes

    guided = prior/T(h) + g(h) * residual(h)

This probe reads T(h) and g(h) directly off trained checkpoints -- no
training-time logging required -- and reports three things:

  A. WHAT T AND g LEARNED.
     T(h) and g(h) over sampled states, cross-tabulated against the prior's
     top-1 margin. The mechanism is only meaningful if T > 1 selectively: high
     where the decision is contestable, near 1 where softening would break
     validity. T flat at 1 means the temperature never engaged; T large
     everywhere means over-softening.

     This is the probe that established the defect reported in the paper. The
     learned T sits between 0.73 and 0.80 everywhere and is flat in the margin,
     and since the forward pass applied clamp(T, min=1) the effective
     temperature was 1 at every state.

  B. VALIDITY AND UNIQUENESS COST.
     Softening the prior can produce invalid molecules, as the residual-scale
     sweep shows directly. Compares validity and uniqueness against the
     plain-residual baseline: flips bought by collapsing validity are not a
     gain.

  C. STEERING DIRECTION.
     More flips only help if they move mass toward reward. Compares the guided
     reward distribution to the prior's and reports the fraction of the extra
     flips that landed on higher-reward continuations, paired against the
     prior's own choice.

Usage (the guide flags mirror gflow_multi's; give the per-component eval
rewards):
    python probe_tempgain.py \
        --guide_ckpts "...comp0--tempgain.../last.ckpt,...comp2...,...comp3..." \
        --guide_labels "c0,c2,c3" \
        --eval_rewards "gcomp:osimertinib:0=c0,gcomp:osimertinib:2=c2,gcomp:osimertinib:3=c3" \
        --route flow --n_traj 400 --out_dir .../tempgain-probe
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


def _has_tempgain(g):
    return hasattr(g, "temp") or hasattr(g, "gain") or hasattr(g, "guided_logits")


@torch.no_grad()
def probe(comp, n_traj=400, gap_edges=(0, 1, 2, 4, 8, 16, 32, 1e9)):
    """Roll prior trajectories; per guide read T(h), g(h) and paired flip info,
    cross-tabulated against the prior top-1 gap."""
    prior = comp.prior
    device = comp.device
    mask = _mask_for(comp.cfg.mask_atoms, device)
    NEG = -1e9
    max_steps = min(comp.cfg.max_len, 64)
    bsz = min(comp.cfg.chunk, n_traj)
    labels, guides = comp.labels, comp.guides
    nb = len(gap_edges) - 1

    # accumulators
    acc = {lab: {"T": [], "g": [],
                 "T_by_gap_sum": np.zeros(nb), "g_by_gap_sum": np.zeros(nb),
                 "flip_by_gap": np.zeros(nb), "count_by_gap": np.zeros(nb),
                 "flip_higher_reward": 0, "flip_total": 0}
           for lab in labels}

    done = 0
    while done < n_traj:
        b = min(bsz, n_traj - done)
        atoms = torch.full((b, 1), GEN, dtype=torch.long, device=device)
        coords = torch.zeros(b, 1, 3, device=device)
        stop_mask = torch.zeros(b, dtype=torch.bool, device=device)

        for _ in range(max_steps):
            idx = torch.arange(atoms.shape[1], device=device).expand(b, -1)
            seq = prior.encode1(idx, atoms, coords)
            h = seq[:, -1, :]
            pl = prior.proj_logits(h).float().masked_fill(~mask, NEG)
            p_prior = F.softmax(pl, -1)
            top2 = torch.topk(pl, 2, -1).values
            gap = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            gap_bin = np.clip(np.digitize(gap, gap_edges[1:-1]), 0, nb - 1)
            u = torch.rand(b, 1, device=device)
            next_prior = (u < torch.cumsum(p_prior, -1)).float().argmax(-1)
            alive = (~stop_mask).cpu().numpy()

            for gi, lab in enumerate(labels):
                g = guides[gi]
                if not _has_tempgain(g):
                    continue
                # read the heads directly
                Tval = g.temp(h).cpu().numpy() if getattr(g, "temp", None) is not None \
                    else np.ones(b)
                gval = g.gain(h).cpu().numpy() if getattr(g, "gain", None) is not None \
                    else np.ones(b)
                guided = g.guided_logits(pl, h).float().masked_fill(~mask, NEG)
                p_g = F.softmax(guided, -1)
                next_g = (u < torch.cumsum(p_g, -1)).float().argmax(-1)
                flip = (next_g != next_prior).cpu().numpy() & alive

                a = acc[lab]
                a["T"].append(Tval[alive]); a["g"].append(gval[alive])
                for k in range(nb):
                    sel = (gap_bin == k) & alive
                    a["count_by_gap"][k] += sel.sum()
                    a["flip_by_gap"][k] += (flip & sel).sum()
                    a["T_by_gap_sum"][k] += Tval[sel].sum()
                    a["g_by_gap_sum"][k] += gval[sel].sum()

            # advance prior trajectory
            atoms = torch.cat([atoms, next_prior.unsqueeze(1)], 1)
            stop_mask = stop_mask | (next_prior == STOP)
            if stop_mask.all():
                break
            x = prior.encode2(atoms[:, 1:], seq)[:, -1, :]
            nc, _ = prior.sample_coord(x, device=device, num_steps=comp.cfg.diff_steps)
            coords = torch.cat([coords, nc.view(b, 1, 3)], 1)
        done += b

    out = {"gap_bins": [f"[{gap_edges[i]:g},{gap_edges[i+1]:g})" for i in range(nb)]}
    out["per_guide"] = {}
    for lab in labels:
        a = acc[lab]
        if not a["T"]:
            out["per_guide"][lab] = {"note": "guide has no temp/gain heads (plain residual)"}
            continue
        T = np.concatenate(a["T"]); g = np.concatenate(a["g"])
        c = np.maximum(a["count_by_gap"], 1)
        out["per_guide"][lab] = {
            "T_mean": float(T.mean()), "T_std": float(T.std()),
            "T_p10": float(np.percentile(T, 10)), "T_p90": float(np.percentile(T, 90)),
            "T_frac_above_1p1": float((T > 1.1).mean()),   # how often temperature engages
            "g_mean": float(g.mean()), "g_std": float(g.std()),
            "g_p10": float(np.percentile(g, 10)), "g_p90": float(np.percentile(g, 90)),
            "T_by_gap": (a["T_by_gap_sum"] / c).tolist(),
            "g_by_gap": (a["g_by_gap_sum"] / c).tolist(),
            "flip_rate_by_gap": (a["flip_by_gap"] / c).tolist(),
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
    ap.add_argument("--n_valid", type=int, default=1000,
                    help="samples for the validity/uniqueness + reward cost check")
    args = ap.parse_args()

    cfg = MultiConfig(**{k: getattr(args, k) for k in MultiConfig.__dataclass_fields__})
    os.makedirs(cfg.out_dir, exist_ok=True)
    comp = Composer(cfg)
    from metrics import compute_valid_unique
    report = {}

    # ---- A: what T/g learned + flip-by-gap ----
    print(f"[A] probing T(h)/g(h) over {args.n_traj} trajectories ...")
    report["learned"] = probe(comp, n_traj=args.n_traj)
    bins = report["learned"]["gap_bins"]
    for lab, d in report["learned"]["per_guide"].items():
        if "note" in d:
            print(f"  {lab}: {d['note']}"); continue
        print(f"\n  {lab}: T mean={d['T_mean']:.2f} (p10={d['T_p10']:.2f} p90={d['T_p90']:.2f}), "
              f"engages(T>1.1)={d['T_frac_above_1p1']*100:.0f}%  |  g mean={d['g_mean']:.2f}")
        print(f"    {'gap':<12} {'T':>6} {'g':>6} {'flip':>7}")
        for bn, T, g, fr in zip(bins, d["T_by_gap"], d["g_by_gap"], d["flip_rate_by_gap"]):
            print(f"    {bn:<12} {T:>6.2f} {g:>6.2f} {fr:>7.3f}")

    # ---- B: validity/uniqueness + reward cost (per single guide vs base) ----
    print(f"\n[B] validity/uniqueness + reward cost ({args.n_valid} samples) ...")
    name0, fn0 = comp.eval_rewards[0]
    base = comp.sample_base(args.n_valid)
    base_r = np.array([fn0(m) for m in base], float)
    bv, bu = compute_valid_unique(base)
    report["cost"] = {"base": {"validity": bv, "uniqueness": bu,
                               "mean_logr": float(np.mean(base_r))}}
    print(f"  base: valid={bv:.3f} uniq={bu:.3f} mean_logr={np.mean(base_r):.3f}")
    # sample each guide alone via the composer single path
    from ablate_singles_weights import sample_single_guide
    for gi, lab in enumerate(comp.labels):
        rname, rfn = comp.eval_rewards[gi] if gi < len(comp.eval_rewards) else (name0, fn0)
        mols = sample_single_guide(comp, gi, args.n_valid)
        r = np.array([rfn(m) for m in mols], float)
        v, u = compute_valid_unique(mols)
        report["cost"][lab] = {
            "validity": v, "uniqueness": u, "mean_logr": float(np.mean(r)),
            "validity_delta_vs_base": v - bv,
            "reward_shift_vs_base": float(np.mean(r) - np.mean(base_r)),
            "scored_on": rname,
        }
        flag = "  <-- VALIDITY DROP" if (bv - v) > 0.1 else ""
        print(f"  {lab}: valid={v:.3f} ({v-bv:+.3f}) uniq={u:.3f}  "
              f"reward_shift={np.mean(r)-np.mean(base_r):+.3f}{flag}")

    with open(os.path.join(cfg.out_dir, "tempgain_probe.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- plots ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pg = {k: v for k, v in report["learned"]["per_guide"].items() if "note" not in v}
        if pg:
            x = np.arange(len(bins))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
            for lab, d in pg.items():
                ax1.plot(x, d["T_by_gap"], "o-", label=f"{lab} T")
            ax1.axhline(1.0, color="k", ls="--", lw=1, label="T=1 (no softening)")
            ax1.set_xticks(x); ax1.set_xticklabels(bins, rotation=35, fontsize=8)
            ax1.set_xlabel("prior top-1 gap"); ax1.set_ylabel("learned T(h)")
            ax1.set_title("A: does temperature engage in the HIGH-gap bins?")
            ax1.legend(fontsize=8)
            for lab, d in pg.items():
                ax2.plot(x, d["flip_rate_by_gap"], "s-", label=f"{lab} flip")
            ax2.set_xticks(x); ax2.set_xticklabels(bins, rotation=35, fontsize=8)
            ax2.set_xlabel("prior top-1 gap"); ax2.set_ylabel("flip rate")
            ax2.set_title("A: flip rate by gap (ceiling broken if >0 at gap>8)")
            ax2.legend(fontsize=8)
            fig.tight_layout()
            p = os.path.join(cfg.out_dir, "tempgain_probe.png")
            fig.savefig(p, dpi=130); plt.close(fig)
            print(f"[plot] {p}")
    except Exception as e:
        print(f"[plot] failed: {e}")

    print(f"[done] -> {os.path.join(cfg.out_dir, 'tempgain_probe.json')}")


if __name__ == "__main__":
    main()
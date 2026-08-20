#!/usr/bin/env python3
"""
Coupled-flip diagnostic for a single-guide checkpoint.

Aggregate reward differences cannot separate a guide that is inert from one that
acts but is mis-directed, so this measures the effect on the decision itself.

Trajectories are rolled by the FROZEN PRIOR; at every atom-decision state the
prior's next-atom distribution is compared against the guided one on the
IDENTICAL state. Because the trajectories belong to the prior, every guide
architecture is compared on the same state distribution. Reported:
  * delivered_frac   -- does the guide's residual reach the logits at all?
                        (~0 => wiring bug: guide not applied)
  * argmax_flip_rate -- how often the guided argmax != prior argmax
  * sample_flip_rate -- how often a paired-RNG sample differs (the real "did the
                        decision change" number)
  * mean_total_variation / mean_KL -- how much probability mass the guide moves
  * mean_prior_top1_gap -- how dominant the prior's top-1 logit is, i.e. how
                        large a residual would have to be to change the decision
  * flip_rate_by_position -- where in the sequence (if anywhere) flips happen.
                        Reported over --n_report_pos positions (default 64).
                        Molecules terminate at different lengths, so positions
                        no trajectory ever reached come back as null, NOT 0.0 --
                        "no molecule was ever this long" is not "the guide never
                        flipped here", and averaging the two together would
                        drag the tail of the curve to a false zero.

Reading the result: delivered ~1 with argmax ~0 and sample ~0, alongside a large
mean_prior_top1_gap, means the residual reaches the logits at every state but
never changes a decision, because the prior's top-1 is too dominant. Delivery
near 0 instead means the residual was computed and not applied -- a wiring
failure rather than a bound, which is why the two are measured separately.

Works for every guide type. The guided logits are computed through the same
dispatch gflow.py uses, and the residual is then derived as
(guided_logits - prior_logits), so the metrics are correct whether the guide
adds an output residual or perturbs the hidden state.

Usage:
  python single_flip_ablation.py \
    --ckpt logs/quetzal-gfn/sweep-nitrogen-hidden-db-replay_off-b10/checkpoints/last.ckpt \
    --n_traj 400 --flip_temp 1.0 --also_temp 0.3 --out_dir flips/nitrogen-hidden-b10
  # try BOTH --flip_temp 1.0 and 0.3 (does the guide only matter when the prior
  # isn't near-greedy?)
"""
import os
import sys
import json
import argparse
import datetime as _dt

import numpy as np
import torch
import torch.nn.functional as F

# final_dump's checkpoint loading (config unwrap + guide-type
# detection + [FATAL] guards) so we never silently measure an untrained guide.
import final_dump as fd2
from chem import GEN, STOP, PAD, QM9_MASK

# default number of sequence positions to report the flip rate for. Molecules
# shorter than this contribute no states to the tail positions; those are
# reported as null (see _pos_rates).
N_REPORT_POS = 256


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


def _label_for(ckpt_path):
    """Name a run after its run dir, skipping generic container dirs.

    …/sweep-fexo-base-db-replay_on-b10/checkpoints/epoch=4-step=489.ckpt
      -> 'sweep-fexo-base-db-replay_on-b10'
    Using the checkpoint's own parent would label every run 'checkpoints'.
    """
    GENERIC = {"last", "final", "best", "checkpoint", "checkpoints", "ckpt",
               "ckpts", "weights", "models"}
    parts = os.path.abspath(ckpt_path).split(os.sep)
    parts[-1] = os.path.splitext(parts[-1])[0]
    for name in reversed(parts):
        if not name or name.lower() in GENERIC:
            continue
        if name.lower().startswith(("epoch=", "step=")):   # epoch=4-step=489
            continue
        return name
    return os.path.splitext(os.path.basename(ckpt_path))[0]


def _pos_mean(sum_by_pos, state_by_pos, n_report=N_REPORT_POS):
    """Per-position mean of a summed quantity, None where no state was reached.

    Same null convention as _pos_rates: a position no trajectory ever reached
    is absent, not zero.
    """
    n = int(min(n_report, len(sum_by_pos)))
    sums = np.asarray(sum_by_pos, dtype=float)[:n]
    states = np.asarray(state_by_pos, dtype=float)[:n]
    return [None if s <= 0 else float(v / s) for v, s in zip(sums, states)]


def _pos_rates(flip_by_pos, state_by_pos, n_report=N_REPORT_POS):
    """flips/states per position, safe against positions with zero states.

    Trajectories terminate at different lengths (and max_steps may itself be
    < n_report if cfg.max_len is small), so the tail of these arrays can be all
    zeros over a zero denominator. Those positions come back as None instead of
    0.0, and the lists are truncated to what was actually simulated -- so asking
    for 64 positions on a run that only ever reached 20 yields 20 entries rather
    than raising or padding with fiction.
    """
    n = int(min(n_report, len(flip_by_pos)))
    flips = np.asarray(flip_by_pos, dtype=float)[:n]
    states = np.asarray(state_by_pos, dtype=float)[:n]
    rates = np.divide(flips, states, out=np.full(n, np.nan), where=states > 0)
    return ([None if np.isnan(r) else float(r) for r in rates],
            [int(s) for s in states])


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
                     progress=False, n_report_pos=N_REPORT_POS):
    prior = lit.frozen
    mask = _mask_for(mask_atoms, device)
    NEG = -1e9
    max_steps = max_steps or min(getattr(lit.cfg, "max_len", 192), n_report_pos)
    bsz = min(chunk, n_traj)

    acc = {"delivered": 0, "n_states": 0, "argmax_flip": 0, "sample_flip": 0,
           "mass_moved_sum": 0.0, "kl_sum": 0.0, "prior_top1_logit_gap_sum": 0.0,
           "flip_by_pos": np.zeros(max_steps), "state_by_pos": np.zeros(max_steps),
           "gap_by_pos": np.zeros(max_steps),
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
            # t < max_steps by construction and both arrays are sized max_steps,
            # so this index is always in range; positions past the longest
            # molecule simply keep a zero denominator and report as null.
            acc["flip_by_pos"][t] += (sample_flip & a).sum().item()
            acc["state_by_pos"][t] += na
            # the prior's margin at this position, summed so it can be divided
            # by state_by_pos later. Figure 2 plots the flip decay against this
            # on a twin axis, and the decay is only interpretable next to it.
            acc["gap_by_pos"][t] += (gap * a.float()).sum().item()

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
    pos_rate, pos_states = _pos_rates(acc["flip_by_pos"], acc["state_by_pos"],
                                      n_report=n_report_pos)
    reached = [i for i, s in enumerate(pos_states) if s > 0]
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
        # null at positions no trajectory ever reached
        "flip_rate_by_position": pos_rate,
        "states_by_position": pos_states,
        "mean_gap_by_position": _pos_mean(acc["gap_by_pos"], acc["state_by_pos"],
                                          n_report),
        "n_positions_reported": len(pos_rate),
        "deepest_position_reached": (reached[-1] + 1) if reached else 0,
        "max_steps": int(max_steps),
        "n_states": ns,
        "temp": temp,
        # RAW counts so several runs can be pooled correctly: sum numerators and
        # denominators, THEN divide. Averaging the rate fields across runs would
        # weight a 5-state position the same as a 5000-state one.
        "raw": {
            "n_states": int(acc["n_states"]),
            "delivered": int(acc["delivered"]),
            "argmax_flip": int(acc["argmax_flip"]),
            "sample_flip": int(acc["sample_flip"]),
            "mass_moved_sum": float(acc["mass_moved_sum"]),
            "kl_sum": float(acc["kl_sum"]),
            "prior_top1_logit_gap_sum": float(acc["prior_top1_logit_gap_sum"]),
            "gap_hi_states": int(acc["gap_flip_hi"]),
            "gap_hi_flipped": int(acc["gap_flip_hi_flipped"]),
            "gap_lo_states": int(acc["gap_lo"]),
            "gap_lo_flipped": int(acc["gap_lo_flipped"]),
            "flip_by_position": [int(x) for x in acc["flip_by_pos"]],
            "state_by_position": [int(x) for x in acc["state_by_pos"]],
            "gap_sum_by_position": [float(x) for x in acc["gap_by_pos"]],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_traj", type=int, default=400)
    ap.add_argument("--flip_temp", type=float, default=1.0)
    ap.add_argument("--also_temp", type=float, default=None,
                    help="optional 2nd temperature to run (e.g. 0.3) -- does the "
                         "guide only matter when the prior isn't near-greedy?")
    ap.add_argument("--n_report_pos", type=int, default=N_REPORT_POS,
                    help="how many sequence positions to report the flip rate "
                         "for (default 64). Positions no molecule ever reached "
                         "come back as null, and the list is truncated to what "
                         "was actually simulated -- asking for more positions "
                         "than the molecules are long is safe.")
    ap.add_argument("--max_steps", type=int, default=0,
                    help="rollout length; 0 = auto = min(cfg.max_len, "
                         "--n_report_pos)")
    ap.add_argument("--diff_steps", type=int, default=18)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--guide_source", choices=["ema", "policy"], default="ema")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label", default=None,
                    help="name for this run in the report (default: derived "
                         "from the checkpoint's run directory)")
    ap.add_argument("--report_tag", default="",
                    help="suffix for the output files, e.g. --report_tag b10_t03 "
                         "writes flip_report_b10_t03.json; keeps runs from "
                         "overwriting each other when they share an --out_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.n_report_pos < 1:
        raise SystemExit("[cfg] --n_report_pos must be >= 1")
    max_steps = args.max_steps or None            # None => auto in flip_diagnostics
    label = args.label or _label_for(args.ckpt)

    # ---- load the checkpoint through final_dump's loader ----
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
    print(f"[flip] {label}: guide={gtype} reward={lit.cfg.reward} "
          f"source={args.guide_source}", flush=True)

    report = {"schema": "single_flip_report/2",
              "label": label, "ckpt": os.path.abspath(args.ckpt),
              "guide_type": gtype, "reward": lit.cfg.reward,
              "reward_smiles": getattr(lit.cfg, "reward_smiles", None),
              "guide_source": args.guide_source, "n_traj": args.n_traj,
              "n_report_pos": args.n_report_pos,
              "max_len": getattr(lit.cfg, "max_len", None),
              "mask_atoms": mask_atoms, "diff_steps": args.diff_steps,
              "report_tag": args.report_tag,
              "run": {"timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                      "argv": sys.argv[1:]}}

    temps = [args.flip_temp] + ([args.also_temp] if args.also_temp is not None else [])
    report["temps"] = temps
    for temp in temps:
        print(f"[flip] rolling {args.n_traj} prior trajectories at temp={temp} ...",
              flush=True)
        res = flip_diagnostics(lit, guide, n_traj=args.n_traj, max_steps=max_steps,
                               temp=temp, diff_steps=args.diff_steps,
                               mask_atoms=mask_atoms, chunk=args.chunk,
                               device=args.device, progress=args.progress,
                               n_report_pos=args.n_report_pos)
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
        print(f"  positions: reported={res['n_positions_reported']} "
              f"deepest_reached={res['deepest_position_reached']} "
              f"(beyond that: null, no molecule was that long)")
        print("  interpretation: deliver~1 & argmax~0 & sample~0 & big gap "
              "=> ceiling (residual reaches logits, never flips the decision).")

    suffix = f"_{args.report_tag}" if args.report_tag else ""
    report_path = os.path.join(args.out_dir, f"flip_report{suffix}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[json] {report_path}")

    # plot flip rate by position (one line per temperature). Positions with no
    # states are skipped rather than plotted as zero.
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for temp in temps:
            r = report[f"flip_temp{temp}"]
            xs = [i for i, v in enumerate(r["flip_rate_by_position"]) if v is not None]
            ys = [r["flip_rate_by_position"][i] for i in xs]
            if not xs:
                continue
            # markers get unreadable past ~16 points
            ax.plot(xs, ys, "o-" if len(xs) <= 16 else "-", label=f"temp={temp}")
        ax.set_xlabel("sequence position"); ax.set_ylabel("sample-flip rate")
        ax.set_title(f"Where does the guide change the sampled atom? "
                     f"({gtype}, {lit.cfg.reward})")
        if ax.has_data():
            ax.legend()
        p = os.path.join(args.out_dir, f"flip_by_position{suffix}.png")
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        print(f"[plot] {p}")
    except Exception as e:
        print(f"[plot] failed: {e}")

    print(f"[done] -> {report_path}")


if __name__ == "__main__":
    main()
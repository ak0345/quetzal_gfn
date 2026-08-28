#!/usr/bin/env python3
"""
Seeded molecule dump and metric suite from a trained guide checkpoint.

Generates --n molecules from the guided policy and, unless --base_from points at
an existing base dump, the same number from the frozen prior, then scores both
and writes everything the aggregator needs.

Outputs (in --out_dir):
  guided_smiles.txt, base_smiles.txt
  guided_rewards.npy, base_rewards.npy
  guided_molecules.pt, base_molecules.pt        (atoms+coords, for re-analysis)
  reward_hist.png            base vs guided reward overlay
  descriptor_grid.png        MW/logP/TPSA/natoms/nrings, guided/base/GEOM overlay
  stability_bar.png          atom & mol stability, guided vs base
  per_molecule.csv           smiles, log_reward, source
  dump_summary.json          every number: seed, valid/unique rates, reward
                             statistics, FCD, stability, descriptor Wasserstein
                             distances. This is what aggregate_dumps.py reads.

Usage:
  python final_dump.py --ckpt .../last.ckpt --n 5000 --seed 0 \
      --ref_smiles reference/geom_drugs_smiles.txt --dataset geom \
      --out_dir results/dumps/<name>/seed0
"""
import os
import json
import random
import argparse

import numpy as np
import numpy as np, scipy
if not hasattr(scipy, "histogram"): scipy.histogram = np.histogram
import torch


class _MolView:
    """Duck-typed stand-in for a chem.Molecule carrying numpy arrays.

    Worker processes need `.atoms` and `.coords` and nothing else: mol_to_rdkit
    reaches them through _atoms_coords, which accepts numpy via np.asarray. The
    parent converts the torch tensors before dispatch so no CUDA tensor is ever
    pickled into a child, and so the real mol_to_rdkit runs unmodified in the
    worker rather than being duplicated here.
    """
    __slots__ = ("atoms", "coords")

    def __init__(self, atoms, coords):
        self.atoms = atoms
        self.coords = coords


# Reward kinds whose score is a pure function of the SMILES string. For these,
# build_reward() itself derives the SMILES by running mol_to_rdkit and then
# MolToSmiles(RemoveHs(...)), so scoring the SMILES we already computed gives
# the identical number while doing bond perception ONCE instead of twice.
#
# Everything else (nitrogen_count, isomer, force) is scored from the molecule.
# That costs nothing extra: none of those three run bond perception at all,
# they read atomic numbers or coordinates directly.
_SMILES_SCORABLE = ("guacamol", "guacamol_component", "qed", "logp", "tpsa",
                    "similarity")

# Fields build_reward / build_reward_smiles read, plus the guard settings
# LitGFlowNet applies on top. Passed to workers as a plain dict so nothing
# model-shaped has to be pickled.
_REWARD_FIELDS = ("reward", "invalid_logr", "reward_smiles", "reward_target",
                  "reward_sigma", "reward_formula", "reward_benchmark",
                  "reward_component", "reward_beta", "force_method",
                  "guard_reward_timeout", "guard_max_atoms")

_W = {}          # per-worker cache, populated once by _worker_init


def reward_payload(cfg):
    """The subset of a GFNConfig the workers need, as a picklable dict."""
    return {k: getattr(cfg, k, None) for k in _REWARD_FIELDS}


def _worker_init(payload):
    """Build the scorer once per worker rather than once per molecule."""
    import types as _types
    import reward_fn as _rf
    from hang_guard import guarded_reward
    cfg = _types.SimpleNamespace(**payload)
    _W["floor"] = float(cfg.invalid_logr if cfg.invalid_logr is not None else -5.0)
    if cfg.reward in _SMILES_SCORABLE:
        _W["smi_scorer"] = _rf.build_reward_smiles(cfg)
        _W["mol_scorer"] = None
    else:
        # same wrapping LitGFlowNet applies, so the numbers match what
        # compute_log_reward would have produced
        _W["smi_scorer"] = None
        _W["mol_scorer"] = guarded_reward(
            _rf.build_reward(cfg),
            seconds=float(cfg.guard_reward_timeout or 0),
            floor=_W["floor"],
            max_atoms=(cfg.guard_max_atoms or None))


def _convert_one(task):
    """(index, atoms, coords) -> (index, smiles or None, log_reward).

    One bond perception per molecule, feeding both outputs. Runs in a worker.
    """
    idx, a, c = task
    floor = _W.get("floor", -5.0)
    lr = floor
    try:
        from reward_fn import mol_to_rdkit
        from rdkit import Chem
        m = _MolView(a, c)
        if _W.get("mol_scorer") is not None:
            lr = float(_W["mol_scorer"](m))
        rd = mol_to_rdkit(m)
        smi = None
        if rd is not None:
            try:
                smi = Chem.MolToSmiles(Chem.RemoveHs(rd)) or None
            except Exception:
                smi = None
        if _W.get("smi_scorer") is not None:
            lr = float(_W["smi_scorer"](smi)) if smi else floor
        return idx, smi, lr
    except Exception:
        return idx, None, floor


def _convert_parallel(mols, payload, procs, timeout, progress_every=0, tag=""):
    """Score and convert 3D molecules across a process pool.

    Returns (smiles_by_index, logr_by_index, n_lost). An entry is None for a
    molecule that failed; n_lost counts molecules abandoned when the pool was
    torn down, and those keep the invalid floor as their reward.

    Order is preserved by index, which matters: the harvest reads generation
    order to compute AUC at an oracle budget.

    WHY A POOL AND NOT A TIMEOUT. rdDetermineBonds searches bond orders and
    charges inside a C++ extension that never returns to the interpreter, so
    SIGALRM cannot interrupt it (see hang_guard.call_with_timeout). Process
    isolation is the only thing that can: a wedged worker is killed with its
    molecule, and the rest of the pool keeps going.

    The timeout is on the interval between COMPLETIONS, not per molecule. While
    any worker is still returning results the pool is making progress; silence
    for `timeout` seconds means every remaining worker is stuck, and the ones
    still outstanding are written off.

    spawn, not fork: the parent has an initialised CUDA context and torch's
    thread pool by this point, and forking either is a known source of deadlock.
    """
    import multiprocessing as _mp

    tasks = []
    for i, m in enumerate(mols):
        a = m.atoms
        c = m.coords
        a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        c = c.detach().cpu().numpy() if hasattr(c, "detach") else np.asarray(c)
        tasks.append((i, a, c))

    floor = float(payload.get("invalid_logr") or -5.0)
    smis = [None] * len(tasks)
    logrs = [floor] * len(tasks)
    done = 0
    ctx = _mp.get_context("spawn")
    pool = ctx.Pool(processes=procs, initializer=_worker_init, initargs=(payload,))
    try:
        # chunksize=1 so one pathological molecule takes down one task rather
        # than the whole chunk it was batched into
        it = pool.imap_unordered(_convert_one, tasks, chunksize=1)
        while True:
            try:
                idx, smi, lr = it.next(timeout=timeout)
            except StopIteration:
                break
            except _mp.TimeoutError:
                print(f"[convert{tag}] no completion for {timeout}s with "
                      f"{len(tasks) - done} outstanding; abandoning them to a "
                      f"wedged worker and moving on", flush=True)
                break
            smis[idx] = smi
            logrs[idx] = lr
            done += 1
            # also print the final tally: with a fixed stride the last partial
            # batch prints nothing, so a healthy run goes silent for its last
            # few hundred molecules and looks exactly like a stall
            if progress_every and (done % progress_every == 0
                                   or done == len(tasks)):
                print(f"[convert{tag}]   {done}/{len(tasks)} scored+converted",
                      flush=True)
            if done == len(tasks):
                break
    finally:
        pool.terminate()
        pool.join()
    return smis, logrs, len(tasks) - done


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ref_smiles(path, limit=None):
    if path is None or not os.path.exists(path):
        return None
    out = []
    with open(path) as f:
        for ln in f:
            s = ln.strip().split()[0] if ln.strip() else ""
            if s:
                out.append(s)
            if limit and len(out) >= limit:
                break
    return out or None


# ------------------------- descriptor helpers -------------------------

_DESCRIPTORS = ["MW", "logP", "TPSA", "n_heavy", "n_rings"]


def descriptors_for_smiles(smiles_list):
    """Return dict descriptor -> np.array over the parseable SMILES."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    vals = {k: [] for k in _DESCRIPTORS}
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        try:
            vals["MW"].append(Descriptors.MolWt(m))
            vals["logP"].append(Descriptors.MolLogP(m))
            vals["TPSA"].append(Descriptors.TPSA(m))
            vals["n_heavy"].append(m.GetNumHeavyAtoms())
            vals["n_rings"].append(Chem.rdMolDescriptors.CalcNumRings(m))
        except Exception:
            continue
    return {k: np.asarray(v, dtype=float) for k, v in vals.items()}


def wasserstein_safe(a, b):
    """1D Wasserstein distance; returns None if either side is too small."""
    if a is None or b is None or len(a) < 5 or len(b) < 5:
        return None
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except Exception:
        return None


# ------------------------- EDM stability -------------------------

def edm_stability_for_mols(mols, dataset="geom", progress_every=0, tag=""):
    """(atom_stability, mol_stability) over a list of Quetzal Molecules, using
    the repo's own edm_metrics. Returns (None, None) if edm_metrics is missing.
    progress_every>0 prints a flushed line every N mols (O(N^2)/mol -> slow)."""
    try:
        import edm_metrics as _edm
    except Exception as e:
        print(f"[stability] edm_metrics import failed ({e}); skipping", flush=True)
        return None, None
    info = _edm.geom_with_h if dataset == "geom" else _edm.qm9_with_h
    mapping = info["mapping"]
    check = _edm.check_stability
    n_atom_stable = n_atom_total = 0
    n_mol_stable = n_mol_total = 0
    for _k, m in enumerate(mols):
        if progress_every and _k > 0 and _k % progress_every == 0:
            print(f"[stability{tag}]   {_k}/{len(mols)} mols checked", flush=True)
        a = m.atoms
        a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        c = m.coords
        c = c.detach().cpu().numpy() if hasattr(c, "detach") else np.asarray(c)
        a = a.reshape(-1).astype(int)
        # trim padding / non-physical
        keep = 0
        for z in a:
            if z <= 0 or z > 118:
                break
            keep += 1
        a = a[:keep]
        c = c.reshape(-1, 3)[:keep]
        if len(a) == 0:
            continue
        try:
            atom_type = [mapping[int(z)] for z in a]
        except KeyError:
            n_mol_total += 1        # unmapped atom -> counts as an (unstable) mol
            continue
        try:
            mol_stable, nr_stable, total = check(np.asarray(c, dtype=float),
                                                 atom_type, info)
        except Exception:
            n_mol_total += 1
            continue
        n_atom_stable += nr_stable
        n_atom_total += total
        n_mol_stable += int(mol_stable)
        n_mol_total += 1
    atom_stab = (n_atom_stable / n_atom_total) if n_atom_total else None
    mol_stab = (n_mol_stable / n_mol_total) if n_mol_total else None
    return atom_stab, mol_stab


# ------------------------- main -------------------------

def _load_base_from(args, summary, keep, lit):
    """Load a precomputed base dump from args.base_from and populate
    keep['base'] + summary['base'], instead of regenerating base. Base is
    identical across guides for the same (reward, seed), so one base dump is
    reused by every guided run of that reward. ABORTS if the cached base was
    made under a different reward (its log-rewards would be meaningless here)."""
    import json as _json
    d = args.base_from
    smi_p = os.path.join(d, "base_smiles.txt")
    rew_p = os.path.join(d, "base_rewards.npy")
    mol_p = os.path.join(d, "base_molecules.pt")
    sum_p = os.path.join(d, "dump_summary.json")

    # ---- reward-match guard: a cached base is only valid for the SAME reward ----
    if os.path.exists(sum_p):
        try:
            cached = _json.load(open(sum_p))
            cached_reward = cached.get("reward")
            cached_bench = cached.get("reward_benchmark")
            cached_smiles = cached.get("reward_smiles")
            this_reward = lit.cfg.reward
            this_bench = getattr(lit.cfg, "reward_benchmark", None)
            this_smiles = getattr(lit.cfg, "reward_smiles", None)
            # match on reward kind AND its selector (benchmark/smiles), since e.g.
            # osim vs fexo are both 'guacamol'/'guacamol_component' but differ by selector.
            mism = (cached_reward != this_reward
                    or (this_reward in ("guacamol", "guacamol_component")
                        and (cached_bench != this_bench or cached_smiles != this_smiles)))
            if mism:
                raise SystemExit(
                    f"[FATAL] --base_from reward mismatch: cached base is "
                    f"reward={cached_reward}/bench={cached_bench}/smiles={cached_smiles} "
                    f"but this run is reward={this_reward}/bench={this_bench}/"
                    f"smiles={this_smiles}. A base dump is only reusable within the "
                    f"same reward. Point --base_from at the matching base.")
            if cached.get("seed") != args.seed:
                print(f"[base_from] NOTE cached base seed={cached.get('seed')} != "
                      f"this seed={args.seed}; base sampling differs slightly. "
                      f"Acceptable if you accept base as seed-shared, else regenerate.")
        except SystemExit:
            raise
        except Exception as e:
            print(f"[base_from] WARNING could not verify reward match ({e}); proceeding")
    else:
        print(f"[base_from] WARNING no dump_summary.json in {d}; cannot verify reward match")

    if not (os.path.exists(smi_p) and os.path.exists(rew_p)):
        raise SystemExit(f"[FATAL] --base_from {d} missing base_smiles.txt/base_rewards.npy")

    smiles = [ln.strip() for ln in open(smi_p) if ln.strip()]
    valid_logr = np.load(rew_p)

    # recompute EDM stability from the saved molecules if available (cheap-ish),
    # else pull it from the cached summary so the base bar/columns stay populated.
    atom_stab = mol_stab = None
    mols = None
    if os.path.exists(mol_p):
        try:
            payload = torch.load(mol_p, map_location="cpu", weights_only=False)
            mols = payload.get("mols")
        except Exception as e:
            print(f"[base_from] could not load base_molecules.pt ({e})")
    if mols is not None:
        atom_stab, mol_stab = edm_stability_for_mols(
            mols, args.dataset,
            progress_every=(args.progress_every if args.progress else 0), tag=" base")
    elif os.path.exists(sum_p):
        try:
            cb = _json.load(open(sum_p)).get("base", {}) or {}
            atom_stab = cb.get("atom_stability"); mol_stab = cb.get("mol_stability")
        except Exception:
            pass

    st = {
        "n_generated": None,
        "n_valid_smiles": len(smiles),
        "parse_rate": None,
        "n_parse_fail": None,
        "n_size_skipped": None,
        "n_conv_lost": None,
        "max_atoms": None,
        "uniqueness": (len(set(smiles)) / len(smiles)) if smiles else 0.0,
        "log_reward_mean": float(valid_logr.mean()) if len(valid_logr) else None,
        "log_reward_top1": float(np.max(valid_logr)) if len(valid_logr) else None,
        "log_reward_top10": float(np.mean(np.sort(valid_logr)[-10:])) if len(valid_logr) >= 10 else None,
        "log_reward_top100": float(np.mean(np.sort(valid_logr)[-100:])) if len(valid_logr) >= 100 else None,
        "atom_stability": atom_stab,
        "mol_stability": mol_stab,
        "reused_from": d,
    }
    summary["base"] = st
    keep["base"] = {"smiles": smiles, "logr": valid_logr, "mols": mols or []}
    summary["base_reused_from"] = d
    print(f"[base] REUSED from {d}: n_valid={len(smiles)} "
          f"logR_mean={st['log_reward_mean']} top10={st['log_reward_top10']} "
          f"atom_stab={atom_stab}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guide_source", choices=["ema", "policy"], default="ema")
    ap.add_argument("--diff_steps", type=int, default=None,
                    help="default: checkpoint's value (18). Do NOT lower for speed.")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--sample_temp", type=float, default=None)
    ap.add_argument("--rand_eps", type=float, default=0.0)
    ap.add_argument("--ref_smiles", default=None,
                    help="GEOM reference SMILES .txt for FCD + descriptor comparison")
    ap.add_argument("--ref_limit", type=int, default=10000,
                    help="cap reference SMILES loaded (FCD is fine with ~10k)")
    ap.add_argument("--dataset", default="geom")
    # Bond perception from 3D coordinates searches over bond orders and charges,
    # and the cost grows sharply with size. A dense or oversized geometry can
    # wedge rdDetermineBonds inside a C++ extension, where no Python-level
    # timeout can reach it, and stall the whole dump. This short-circuits before
    # the call. Measured over the 1.03M molecules dumped so far, a cap of 120
    # excludes ~0.002% of the converted set (p99.9 is 83 atoms), so it costs
    # roughly nothing; `n_size_skipped` records the exact count per dump so the
    # loss is reportable rather than assumed. 0 disables the cap.
    ap.add_argument("--max_atoms", type=int, default=120,
                    help="skip 3D->SMILES conversion above this atom count "
                         "(counted to the first STOP); 0 disables")
    # The conversion is the dump's bottleneck and is embarrassingly parallel.
    # conv_procs x DUMP_PARALLEL should stay at or under the vCPU count: on a
    # 16-vCPU box, 8 workers with two concurrent dumps saturates it.
    ap.add_argument("--conv_procs", type=int, default=0,
                    help="worker processes for 3D->SMILES; 0 = auto "
                         "(half the vCPUs), 1 = serial, no pool")
    # This is silence across EVERY worker, not a per-molecule budget. A healthy
    # molecule converts in milliseconds and a full pool completes many per
    # second, so total silence is unambiguous well before a minute. It is paid
    # once per dump that has a wedged molecule, at the very end when the healthy
    # workers have drained the queue, so an over-generous value is dead time
    # multiplied by the number of dumps.
    ap.add_argument("--conv_timeout", type=float, default=90.0,
                    help="seconds with NO conversion completing anywhere before "
                         "the outstanding molecules are written off as wedged")
    ap.add_argument("--skip_base", action="store_true")
    ap.add_argument("--skip_guided", action="store_true")
    ap.add_argument("--base_from", default=None,
                    help="DIR of a precomputed base dump (from a --skip_guided run). "
                         "Loads base_molecules.pt/base_rewards.npy/base_smiles.txt and "
                         "REUSES them instead of regenerating base. The base dump's "
                         "reward MUST match this run's reward (checked via its "
                         "dump_summary.json); mismatched reward -> abort.")
    ap.add_argument("--no_fcd", action="store_true", help="disable FCD even if available")
    ap.add_argument("--progress", action="store_true",
                    help="print flushed progress for generation + post-processing "
                         "(log-file friendly: periodic lines, not a redrawing bar)")
    ap.add_argument("--progress_every", type=int, default=500,
                    help="print a post-processing progress line every N molecules")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    set_seed(args.seed)

    import gflow
    from gflow import LitGFlowNet, mol_to_rdkit
    from hang_guard import default_n_atoms
    from rdkit import Chem

    print(f"[load] {args.ckpt} (seed={args.seed})")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", ckpt.get("hparams", None))
    if hp is None:
        raise SystemExit("checkpoint has no hyper_parameters; cannot rebuild config")

    # save_hyperparameters() captured the __init__ arg named `config`, so Lightning
    # stores hp = {"config": {...real dict...}}. Passing that whole thing back as
    # `config` would nest it one level too deep -> GFNConfig() falls back to ALL
    # DEFAULTS (incl. use_hidden_guide=True), silently rebuilding the WRONG guide
    # for base/tempgain checkpoints and mismatching every key. Unwrap it here.
    if isinstance(hp, dict) and "config" in hp and isinstance(hp["config"], dict):
        hp = hp["config"]
    elif not (isinstance(hp, dict) and any(
            k in hp for k in ("use_hidden_guide", "objective", "reward"))):
        # Lightning AttributeDict or similar -> coerce to plain dict
        try:
            hp = dict(hp)
            if "config" in hp and isinstance(hp["config"], dict):
                hp = hp["config"]
        except Exception:
            pass

    lit = LitGFlowNet(hp)

    # sanity: the rebuilt guide type MUST match the checkpoint, or we'd dump an
    # untrained (identity) guide == base, silently invalidating the run.
    ckpt_hidden = bool(hp.get("use_hidden_guide", True)) if isinstance(hp, dict) else True
    built_hidden = type(lit.guide).__name__ == "HiddenGuide"
    if ckpt_hidden != built_hidden:
        raise SystemExit(
            f"[FATAL] guide-type mismatch: checkpoint use_hidden_guide={ckpt_hidden} "
            f"but rebuilt guide is {type(lit.guide).__name__}. Config did not round-trip; "
            f"refusing to dump an untrained guide.")

    missing, unexpected = lit.load_state_dict(ckpt["state_dict"], strict=False)
    real_missing = [m for m in missing if "n_averaged" not in m and "frozen" not in m]
    # HARD FAIL if any GUIDE weight is missing -- that means the trained guide
    # didn't load and guided == base (the exact bug this run hit before).
    guide_missing = [m for m in real_missing if m.startswith("guide")
                     and not m.startswith("guide_ema")]
    if guide_missing:
        raise SystemExit(
            f"[FATAL] {len(guide_missing)} guide weights did NOT load from the "
            f"checkpoint (e.g. {guide_missing[:4]}). The guide is untrained -> guided "
            f"would equal base. Aborting. Check the checkpoint / config round-trip.")
    if real_missing:
        print(f"[warn] missing (non-guide) keys on load: {real_missing[:8]}")
    lit = lit.to(args.device).eval()

    if args.diff_steps is not None:
        if args.diff_steps < 8:
            print(f"[WARN] diff_steps={args.diff_steps} very low -- degenerate mols / "
                  f"bond-perception hangs likely. Training used {lit.cfg.diff_steps}.")
        lit.cfg.diff_steps = args.diff_steps
    if args.sample_temp is not None:
        lit.cfg.sample_temp = args.sample_temp

    print(f"[cfg] reward={lit.cfg.reward} reward_smiles={getattr(lit.cfg,'reward_smiles',None)} "
          f"diff_steps={lit.cfg.diff_steps} sample_temp={lit.cfg.sample_temp} "
          f"guide_source={args.guide_source} dataset={args.dataset}")

    ref_smiles = load_ref_smiles(args.ref_smiles, limit=args.ref_limit)
    if args.ref_smiles and ref_smiles is None:
        print(f"[ref] WARNING: could not load ref smiles from {args.ref_smiles}")

    summary = {
        "ckpt": args.ckpt, "name": os.path.basename(os.path.dirname(os.path.dirname(args.ckpt))),
        "seed": args.seed, "n_requested": args.n, "diff_steps": lit.cfg.diff_steps,
        "guide_source": args.guide_source, "reward": lit.cfg.reward,
        "reward_smiles": getattr(lit.cfg, "reward_smiles", None),
        "reward_benchmark": getattr(lit.cfg, "reward_benchmark", None),
        "sample_temp": lit.cfg.sample_temp,
    }
    all_rows = []
    keep = {}   # source -> dict(smiles=[], logr=np, mols=[])

    @torch.no_grad()
    def dump(source_name, guide_arg):
        print(f"[{source_name}] generating {args.n} (single batched call) ...", flush=True)
        if args.progress:
            print(f"[{source_name}] rollout started: {args.n} mols x "
                  f"~{lit.cfg.max_len} atom-steps x {lit.cfg.diff_steps} diff-steps "
                  f"-- this is the slow part, progress bar below", flush=True)
        try:
            if args.progress:
                # call generate_guided directly with pbar=True so we get a live
                # bar over the atom-generation steps (the real progress axis).
                # tqdm mininterval keeps it log-friendly (few lines, not thousands).
                import tqdm as _tqdm
                _orig_trange = _tqdm.trange
                def _slow_trange(*a, **k):
                    k.setdefault("mininterval", 2.0)   # <=1 line every 2s -> clean log
                    k.setdefault("desc", f"{source_name} gen")
                    return _orig_trange(*a, **k)
                _tqdm.trange = _slow_trange
                try:
                    guide_mod = (lit.guide if guide_arg == "policy"
                                 else lit.guide_ema.module if guide_arg == "ema"
                                 else guide_arg)
                    mols_b, _, _info = lit.frozen.generate_guided(
                        args.n, guide=guide_mod, sample_temp=lit.cfg.sample_temp,
                        rand_eps=args.rand_eps, max_len=lit.cfg.max_len,
                        device=lit.device, pbar=True, mask_atoms=lit.cfg.mask_atoms,
                        num_steps=lit.cfg.diff_steps)
                    mols = mols_b.unbatch()
                finally:
                    _tqdm.trange = _orig_trange   # restore
            else:
                res = lit.rollout(args.n, guide=guide_arg, sample_temp=lit.cfg.sample_temp,
                                  rand_eps=args.rand_eps, with_reward=False)
                mols = res["mols"]
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[{source_name}] OOM at n={args.n}; chunk={args.chunk}", flush=True)
                torch.cuda.empty_cache()
                res = lit.rollout_chunked(args.n, guide=guide_arg, chunk=args.chunk,
                                          with_reward=False)
                mols = res["mols"]
            else:
                raise
        if args.progress:
            print(f"[{source_name}] generation done ({len(mols)} mols); scoring reward ...",
                  flush=True)
        # ---- size cap, before anything expensive -------------------------
        # Counted separately from n_parse_fail: "too large to attempt" and
        # "attempted and failed" are different populations, and only the first
        # is a choice this script made.
        n_size_skipped = 0
        todo_idx, todo_mols = [], []
        for _i, m in enumerate(mols):
            if args.max_atoms > 0:
                try:
                    if default_n_atoms(m) > args.max_atoms:
                        n_size_skipped += 1
                        continue
                except Exception:
                    pass          # unrecognised molecule type: convert as before
            todo_idx.append(_i); todo_mols.append(m)
        todo_set = set(todo_idx)

        # ---- score + convert, in ONE bond-perception pass -----------------
        # These used to be two passes: compute_log_reward ran mol_to_rdkit over
        # every molecule to score it, then this loop ran mol_to_rdkit again to
        # record the SMILES. For a GuacaMol reward that was the dump's single
        # largest cost, paid twice.
        floor = float(getattr(lit.cfg, "invalid_logr", -5.0))
        procs = args.conv_procs or max(1, (os.cpu_count() or 4) // 2)
        prog = args.progress_every if args.progress else 0
        n_conv_lost = 0
        if procs > 1 and len(todo_mols) > 1:
            if args.progress:
                print(f"[{source_name}] scoring + converting {len(todo_mols)} "
                      f"mols across {procs} workers ...", flush=True)
            sub_smi, sub_lr, n_conv_lost = _convert_parallel(
                todo_mols, reward_payload(lit.cfg), procs, args.conv_timeout,
                progress_every=prog, tag=f" {source_name}")
        else:
            # serial fallback (--conv_procs 1): the original two-pass path,
            # kept verbatim so a comparison run is always available
            if args.progress:
                print(f"[{source_name}] scoring reward (serial) ...", flush=True)
            sub_lr = [float(v) for v in
                      lit.compute_log_reward(todo_mols).cpu().numpy()]
            sub_smi = []
            for _k, m in enumerate(todo_mols):
                if prog and _k > 0 and _k % prog == 0:
                    print(f"[{source_name}]   processed {_k}/{len(todo_mols)}",
                          flush=True)
                rd = mol_to_rdkit(m)
                if rd is None:
                    sub_smi.append(None); continue
                try:
                    sub_smi.append(Chem.MolToSmiles(Chem.RemoveHs(rd)) or None)
                except Exception:
                    sub_smi.append(None)

        # ---- reassemble in GENERATION order ------------------------------
        # The harvest reads generation order to compute AUC at an oracle
        # budget, so the surviving molecules must keep their original sequence.
        # A molecule skipped by the size cap keeps the invalid floor, the same
        # value an unparseable one gets, and is recorded in n_size_skipped.
        smi_by_idx = [None] * len(mols)
        logr = np.full(len(mols), floor, dtype=np.float32)
        for _j, _i in enumerate(todo_idx):
            smi_by_idx[_i] = sub_smi[_j]
            logr[_i] = sub_lr[_j]

        smiles, valid_logr = [], []
        n_attempted_fail = 0
        for _i, lr in enumerate(logr):
            smi = smi_by_idx[_i]
            if smi:
                smiles.append(smi)
                valid_logr.append(float(lr))
                all_rows.append({"source": source_name, "smiles": smi,
                                 "log_reward": float(lr)})
            elif _i in todo_set:
                n_attempted_fail += 1
        # a molecule the pool never returned is "lost", not "failed": it was
        # never scored, so folding it into n_parse_fail would misreport bond
        # perception as the cause.
        n_parse_fail = n_attempted_fail - n_conv_lost
        valid_logr = np.array(valid_logr)

        with open(os.path.join(args.out_dir, f"{source_name}_smiles.txt"), "w") as f:
            f.write("\n".join(smiles) + ("\n" if smiles else ""))
        np.save(os.path.join(args.out_dir, f"{source_name}_rewards.npy"), valid_logr)
        torch.save({"mols": mols, "log_reward": logr},
                   os.path.join(args.out_dir, f"{source_name}_molecules.pt"))

        # EDM stability over ALL generated mols (not just SMILES-valid ones)
        atom_stab, mol_stab = edm_stability_for_mols(
            mols, args.dataset,
            progress_every=(args.progress_every if args.progress else 0),
            tag=f" {source_name}")

        parse_rate = len(smiles) / max(len(mols), 1)
        st = {
            "n_generated": len(mols),
            "n_valid_smiles": len(smiles),
            "parse_rate": parse_rate,
            "n_parse_fail": n_parse_fail,
            "n_size_skipped": n_size_skipped,
            "n_conv_lost": n_conv_lost,
            "max_atoms": args.max_atoms,
            "uniqueness": (len(set(smiles)) / len(smiles)) if smiles else 0.0,
            "log_reward_mean": float(valid_logr.mean()) if len(valid_logr) else None,
            "log_reward_top1": float(np.max(valid_logr)) if len(valid_logr) else None,
            "log_reward_top10": float(np.mean(np.sort(valid_logr)[-10:])) if len(valid_logr) >= 10 else None,
            "log_reward_top100": float(np.mean(np.sort(valid_logr)[-100:])) if len(valid_logr) >= 100 else None,
            "atom_stability": atom_stab,
            "mol_stability": mol_stab,
        }
        summary[source_name] = st
        keep[source_name] = {"smiles": smiles, "logr": valid_logr, "mols": mols}
        print(f"[{source_name}] valid={len(smiles)}/{len(mols)} ({parse_rate:.3f}) "
              f"logR_mean={st['log_reward_mean']} top10={st['log_reward_top10']} "
              f"atom_stab={atom_stab} mol_stab={mol_stab}")
        if parse_rate < 0.5:
            print(f"[{source_name}] WARNING low parse rate -- 3D->2D dropping mols.")
        return valid_logr

    if not args.skip_guided:
        dump("guided", args.guide_source)

    if args.base_from:
        _load_base_from(args, summary, keep, lit)
    elif not args.skip_base:
        dump("base", None)

    guided = keep.get("guided")
    base = keep.get("base")

    # ---------- FCD ----------
    if not args.no_fcd:
        try:
            from gflow import _get_fcd
            fcd_fn = _get_fcd()
            if fcd_fn is None:
                print("[fcd] no backend (pip install fcd_torch); skipping")
            else:
                fcd = {}
                gs = guided["smiles"] if guided else []
                bs = base["smiles"] if base else []
                if len(gs) > 10 and len(bs) > 10:
                    fcd["guided_vs_base"] = fcd_fn(gs, bs)
                if ref_smiles and len(ref_smiles) > 10:
                    if len(gs) > 10:
                        fcd["guided_vs_ref"] = fcd_fn(gs, ref_smiles)
                    if len(bs) > 10:
                        fcd["base_vs_ref"] = fcd_fn(bs, ref_smiles)
                summary["fcd"] = fcd
                print(f"[fcd] {fcd}")
        except Exception as e:
            print(f"[fcd] failed: {e}")

    # ---------- descriptor distributions vs GEOM ----------
    desc = {}
    try:
        g_desc = descriptors_for_smiles(guided["smiles"]) if guided else None
        b_desc = descriptors_for_smiles(base["smiles"]) if base else None
        r_desc = descriptors_for_smiles(ref_smiles) if ref_smiles else None
        # Wasserstein distance of guided/base to the GEOM ref, per descriptor
        wass = {}
        if r_desc is not None:
            for k in _DESCRIPTORS:
                wass[k] = {
                    "guided_vs_ref": wasserstein_safe(g_desc[k] if g_desc else None, r_desc[k]),
                    "base_vs_ref": wasserstein_safe(b_desc[k] if b_desc else None, r_desc[k]),
                }
        summary["descriptor_wasserstein"] = wass
        desc = {"guided": g_desc, "base": b_desc, "ref": r_desc}
    except Exception as e:
        print(f"[descriptors] failed: {e}")

    # ---------- plots ----------
    _make_plots(args, summary, guided, base, desc)

    # ---------- per-molecule csv + summary ----------
    import csv
    with open(os.path.join(args.out_dir, "per_molecule.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "smiles", "log_reward"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    if guided and base and len(guided["logr"]) >= 10 and len(base["logr"]) >= 10:
        gt = float(np.mean(np.sort(guided["logr"])[-10:]))
        bt = float(np.mean(np.sort(base["logr"])[-10:]))
        summary["top10_delta_guided_minus_base"] = gt - bt
        print(f"\n[STEERING] top10 delta (guided-base) = {gt-bt:+.4f} "
              f"(near 0 => ceiling; large positive => steering)")

    with open(os.path.join(args.out_dir, "dump_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] -> {args.out_dir}")


def _make_plots(args, summary, guided, base, desc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable: {e}")
        return

    # 1) reward histogram (the ceiling figure)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        bins = np.linspace(-15, 0.5, 60)
        if base and len(base["logr"]):
            ax.hist(base["logr"], bins=bins, density=True, alpha=0.6, label="base")
        if guided and len(guided["logr"]):
            ax.hist(guided["logr"], bins=bins, density=True, alpha=0.6, label="guided")
        ax.set_xlabel("log reward"); ax.set_ylabel("density")
        ax.set_title(f"reward dist ({summary.get('name','')}, seed {args.seed})")
        ax.legend()
        if guided and base and len(guided["logr"]) >= 10 and len(base["logr"]) >= 10:
            gt = np.mean(np.sort(guided["logr"])[-10:])
            bt = np.mean(np.sort(base["logr"])[-10:])
            ax.text(0.02, 0.97, f"top10 base={bt:.3f} guided={gt:.3f} d={gt-bt:+.3f}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        p = os.path.join(args.out_dir, "reward_hist.png")
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        print(f"[plot] {p}")
    except Exception as e:
        print(f"[plot] reward_hist failed: {e}")

    # 2) descriptor grid: guided/base/GEOM overlay per descriptor
    try:
        import matplotlib.pyplot as plt
        g = desc.get("guided"); b = desc.get("base"); r = desc.get("ref")
        if any(x is not None for x in (g, b, r)):
            fig, axes = plt.subplots(1, len(_DESCRIPTORS), figsize=(4 * len(_DESCRIPTORS), 3.5))
            for ax, k in zip(axes, _DESCRIPTORS):
                series = []
                if r is not None and len(r[k]): series.append(r[k])
                if b is not None and len(b[k]): series.append(b[k])
                if g is not None and len(g[k]): series.append(g[k])
                if not series:
                    continue
                lo = min(s.min() for s in series); hi = max(s.max() for s in series)
                bins = np.linspace(lo, hi, 40)
                if r is not None and len(r[k]):
                    ax.hist(r[k], bins=bins, density=True, alpha=0.5, label="GEOM")
                if b is not None and len(b[k]):
                    ax.hist(b[k], bins=bins, density=True, alpha=0.5, label="base")
                if g is not None and len(g[k]):
                    ax.hist(g[k], bins=bins, density=True, alpha=0.5, label="guided")
                ax.set_title(k); ax.set_yticks([])
            axes[0].legend(fontsize=8)
            fig.suptitle(f"descriptors vs GEOM ({summary.get('name','')}, seed {args.seed})")
            p = os.path.join(args.out_dir, "descriptor_grid.png")
            fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
            print(f"[plot] {p}")
    except Exception as e:
        print(f"[plot] descriptor_grid failed: {e}")

    # 3) stability bar: atom & mol stability, guided vs base
    try:
        import matplotlib.pyplot as plt
        gS = summary.get("guided", {}); bS = summary.get("base", {})
        labels = ["atom_stability", "mol_stability"]
        gv = [gS.get(k) or 0 for k in labels]
        bv = [bS.get(k) or 0 for k in labels]
        x = np.arange(len(labels)); w = 0.35
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(x - w/2, bv, w, label="base")
        ax.bar(x + w/2, gv, w, label="guided")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1); ax.set_ylabel("fraction"); ax.legend()
        ax.set_title(f"EDM stability ({summary.get('name','')}, seed {args.seed})")
        for i, (bb, gg) in enumerate(zip(bv, gv)):
            ax.text(i - w/2, bb + 0.01, f"{bb:.3f}", ha="center", fontsize=8)
            ax.text(i + w/2, gg + 0.01, f"{gg:.3f}", ha="center", fontsize=8)
        p = os.path.join(args.out_dir, "stability_bar.png")
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        print(f"[plot] {p}")
    except Exception as e:
        print(f"[plot] stability_bar failed: {e}")


if __name__ == "__main__":
    main()
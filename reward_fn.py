"""
Reward functions for GFlowNet fine-tuning of Quetzal.

Two families, selected by cfg.reward:

  (1) GuacaMol  (arXiv:1811.09621) -- 2D molecular-graph scores in [0, 1].
      These depend ONLY on the perceived molecular graph (atom types +
      connectivity), so they pair naturally with discrete-logit guidance.
      Objectives: "qed", "logp", "tpsa", "isomer", "similarity", and an
      optional "guacamol" passthrough to the installed guacamol package.

  (2) RLPF      (arXiv:2508.16521) -- 3D force-field reward.
      r = -RMS atomic force (eV/A) from GFN2-xTB (paper) or an MMFF surrogate.
      Lower force => closer to a relaxed/equilibrium structure.

CONVENTION: build_reward(cfg) returns a callable `log_reward(mol) -> float`
giving the *base* log reward (beta = 1). The trainer multiplies by
cfg.reward_beta, so the sampled distribution is  p_prior(x) * exp(beta * base).
  - GuacaMol:  base = log(score)        -> target ~ p_prior * score^beta
  - RLPF:      base = -RMSF             -> target ~ p_prior * exp(-beta * RMSF)
Invalid molecules return cfg.invalid_logr (RLPF paper uses -5).

Quetzal outputs RAW ATOMIC NUMBERS (C=6, N=7, O=8, F=9, H=1), NOT model
indices. So no atom-type remapping is applied anywhere in this file.
"""

import math
import re
import numpy as np
import numpy as np, scipy
if not hasattr(scipy, "histogram"): scipy.histogram = np.histogram

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs, QED
from rdkit.Geometry import Point3D

try:
    from rdkit.Chem import rdDetermineBonds  # RDKit >= 2022.09
    _HAS_DETERMINE_BONDS = True
except Exception:
    _HAS_DETERMINE_BONDS = False


# Toggle this to see *why* molecules fail (atom conversion, bonds, xTB, etc.)
DEBUG_REWARD = False


def _dbg(msg):
    if DEBUG_REWARD:
        print(f"[reward] {msg}")


# ----------------------- Molecule -> (Z, coords) -----------------------

def _atoms_coords(mol):
    """Return (atomic_numbers: int list, coords: (N,3) float array), truncated
    at the first STOP/pad (atomic number 0 or anything non-physical)."""
    a = mol.atoms
    c = mol.coords
    a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    c = c.detach().cpu().numpy() if hasattr(c, "detach") else np.asarray(c)
    a = a.reshape(-1).astype(int)
    c = c.reshape(-1, 3).astype(float)
    # cut at first non-physical token (0 = STOP/pad, or anything > 118)
    keep = 0
    for z in a:
        if z <= 0 or z > 118:
            break
        keep += 1
    return a[:keep].tolist(), c[:keep]


# ----------------------- Molecule -> RDKit Mol -----------------------

def mol_to_rdkit(molecule):
    """
    Convert a Quetzal Molecule (raw atomic numbers + 3D coords) to a sanitized
    RDKit Mol with inferred bonds. Returns None on any failure.

    Quetzal already outputs atomic numbers, so they are used directly.
    """
    Z, coords = _atoms_coords(molecule)
    if len(Z) == 0:
        _dbg("empty molecule after truncation")
        return None

    try:
        rw = Chem.RWMol()
        for z in Z:
            rw.AddAtom(Chem.Atom(int(z)))

        conf = Chem.Conformer(len(Z))
        for i, (x, y, zc) in enumerate(coords):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(zc)))
        rw.AddConformer(conf)

        mol = rw.GetMol()

        # Infer bonds from 3D geometry
        if _HAS_DETERMINE_BONDS:
            try:
                rdDetermineBonds.DetermineBonds(mol, charge=0)
            except Exception as e:
                # Fall back to connectivity-only if full perception fails
                _dbg(f"DetermineBonds failed ({e}); trying connectivity only")
                rdDetermineBonds.DetermineConnectivity(mol)
        else:
            _dbg("rdDetermineBonds unavailable in this RDKit build")
            return None

        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        _dbg(f"conversion failed: {type(e).__name__}: {e}")
        return None


# ----------------------- GuacaMol modifiers -----------------------

def _gaussian(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _min_gaussian(x, mu, sigma):   # full score below mu
    return 1.0 if x <= mu else _gaussian(x, mu, sigma)


def _max_gaussian(x, mu, sigma):   # full score above mu
    return 1.0 if x >= mu else _gaussian(x, mu, sigma)


def _thresholded(x, t):            # full score above t, linear to 0 below
    if t <= 0:
        return 1.0
    return min(max(x / t, 0.0), 1.0)


_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_SYMBOL_TO_Z = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Se": 34, "Br": 35, "I": 53,
}


def _parse_formula(formula):
    counts = {}
    for sym, num in _FORMULA_RE.findall(formula):
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0) + (int(num) if num else 1)
    return counts


# ----------------------- GuacaMol scorers (score in [0,1]) -----------------------

def _score_qed(mol):
    m = Chem.RemoveHs(mol)
    return float(QED.qed(m))


def _score_logp(mol, target, sigma):
    m = Chem.RemoveHs(mol)
    return _gaussian(Descriptors.MolLogP(m), target, sigma)


def _score_tpsa(mol, target, sigma):
    m = Chem.RemoveHs(mol)
    return _gaussian(Descriptors.TPSA(m), target, sigma)


def _score_isomer(molecule, formula):
    """GuacaMol isomer score: geometric mean of per-element Gaussians
    (sigma=1, mean = target count) and a total-atom Gaussian (sigma=2). Counts
    include hydrogens, so use the raw atomic numbers from the Molecule."""
    Z, _ = _atoms_coords(molecule)
    if len(Z) == 0:
        return 0.0
    target = _parse_formula(formula)
    z_of = {_SYMBOL_TO_Z[s]: s for s in target if s in _SYMBOL_TO_Z}
    counts = {s: 0 for s in target}
    for z in Z:
        if z in z_of:
            counts[z_of[z]] += 1
    terms = [_gaussian(counts[s], target[s], 1.0) for s in target]
    n_target = sum(target.values())
    terms.append(_gaussian(len(Z), n_target, 2.0))
    prod = 1.0
    for t in terms:
        prod *= t
    return prod ** (1.0 / len(terms))

def _score_nitrogen_fraction(molecule):
    """EASY / Ablation-X reward: fraction of heavy atoms that are nitrogen.
    Reads raw atomic numbers directly (no bond perception, so a chemically
    valid geometry never fails here). score in (0, 1]; higher = more N-rich.

    Deliberately easy, monotone, atom-level target: the guide raises it purely
    by emitting N (Z=7) instead of C/O/F at each step, the exact discrete
    decision it controls. If hidden/logit guidance cannot steer even this, the
    training loop is broken independent of the saturated-prior ceiling; if it
    can, osimertinib is simply too hard for logit guidance.

    Empty molecule -> 0.0 (caller maps to the invalid floor). A small epsilon
    gives all-carbon molecules a tiny nonzero score so log() stays finite and
    the gradient still points 'add N' instead of sitting flat at the floor.
    """
    Z, _ = _atoms_coords(molecule)
    heavy = [z for z in Z if z > 1]        # exclude explicit H
    if len(heavy) == 0:
        return 0.0
    n_count = sum(1 for z in heavy if z == 7)
    eps = 0.05
    return (n_count + eps) / (len(heavy) + eps)

# --- atom stability (EDM, via Quetzal's own edm_metrics.check_stability) ---
# Calls the repo's check_stability so the reward is IDENTICAL to the reported
# GEOM atom-stability metric. Atoms are mapped to EDM indices via the dataset
# 'mapping'; an atom outside GEOM's 16 elements makes the molecule invalid
# (caller -> floor), matching edm_metrics (which skips such molecules).
_EDM_STATE = {"tried": False, "check": None, "info": {}}

def _get_edm_stability(dataset="geom"):
    """Resolve (check_stability, dataset_info) from Quetzal's edm_metrics once.
    Returns (fn, info) or (None, None) if the module can't be imported."""
    if not _EDM_STATE["tried"]:
        _EDM_STATE["tried"] = True
        try:
            import edm_metrics as _edm
            _EDM_STATE["check"] = _edm.check_stability
            _EDM_STATE["info"] = {
                "geom": _edm.geom_with_h,
                "qm9": _edm.qm9_with_h,
            }
        except Exception as e:
            _dbg(f"atom_stability: edm_metrics import failed ({e})")
            _EDM_STATE["check"] = None
    info = _EDM_STATE["info"].get(dataset)
    return _EDM_STATE["check"], info

def _score_atom_stability(molecule, dataset="geom"):
    """Fraction of atoms with correct valence (EDM atom stability), in [0, 1],
    using Quetzal's check_stability. Returns None on empty / unmapped-atom /
    failure so the caller maps it to the invalid floor."""
    Z, coords = _atoms_coords(molecule)
    if len(Z) == 0:
        return None
    check, info = _get_edm_stability(dataset)
    if check is None or info is None:
        _dbg("atom_stability: edm_metrics/check_stability unavailable")
        return None
    mapping = info["mapping"]
    try:
        atom_type = [mapping[int(z)] for z in Z]   # atomic number -> EDM index
    except KeyError:
        # atom outside this dataset's element set -> invalid (as edm_metrics does)
        return None
    positions = np.asarray(coords, dtype=float)
    try:
        _mol_stable, nr_stable, total = check(positions, atom_type, info)
    except Exception as e:
        _dbg(f"check_stability failed: {e}")
        return None
    if total == 0:
        return None
    return nr_stable / total


def _morgan(mol, radius=2, nbits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(
        Chem.RemoveHs(mol), radius, nBits=nbits)


def _score_similarity(mol, target_fp, threshold):
    sim = DataStructs.TanimotoSimilarity(_morgan(mol), target_fp)
    return _thresholded(sim, threshold) if threshold and threshold > 0 else sim


# ----------------------- Force-field reward (RLPF) -----------------------

def _rmsf_from_forces(forces):
    f = np.asarray(forces, dtype=float).reshape(-1, 3)
    return float(np.sqrt((f ** 2).sum() / (3 * len(f))))


def _force_rmsf_xtb(molecule, method="GFN2-xTB"):
    """RMS atomic force (eV/A) via GFN2-xTB through ASE.
    Requires `xtb` python bindings (pip install xtb) and `ase`."""
    from ase import Atoms                       # lazy import
    from xtb.ase.calculator import XTB
    Z, xyz = _atoms_coords(molecule)
    if len(Z) == 0:
        return None
    atoms = Atoms(numbers=Z, positions=xyz)
    atoms.calc = XTB(method=method)
    forces = atoms.get_forces()                 # eV/A
    return _rmsf_from_forces(forces)


def _force_rmsf_mmff(molecule):
    """RMS atomic force from MMFF94 at the given geometry (kcal/mol/A).
    Cheap RDKit surrogate -- different units/scale than xTB, so tune beta."""
    m = mol_to_rdkit(molecule)
    if m is None:
        return None
    m = Chem.AddHs(m, addCoords=True)
    props = AllChem.MMFFGetMoleculeProperties(m)
    if props is None:
        return None
    ff = AllChem.MMFFGetMoleculeForceField(m, props)
    if ff is None:
        return None
    grad = np.asarray(ff.CalcGrad(), dtype=float)   # dE/dx, kcal/mol/A
    return _rmsf_from_forces(-grad)


# ----------------------- Optional: full guacamol package -----------------------

# ============================================================================
# DROP-IN REPLACEMENT for _guacamol_scoring_fn in reward_fn.py
#
# The old version did `getattr(sb, name)()`, which only works for benchmarks that
# happen to be NULLARY functions in guacamol.standard_benchmarks (hard_osimertinib,
# amlodipine_rings, ...). It CANNOT reach the similarity/rediscovery benchmarks
# (albuterol, celecoxib, mestranol, thiothixene, troglitazone), which the v2 suite
# builds by calling similarity(...) with arguments -- so those raised AttributeError.
#
# This version resolves ANY v2 goal-directed benchmark by matching against the
# canonical suite (goal_directed_benchmark_suite('v2')), which contains the fully
# constructed GoalDirectedBenchmark objects. It accepts:
#   * the guacamol suite display name         : "Osimertinib", "Albuterol", ...
#   * the friendly / heatmap label            : "osimertinib_mpo", "albuterol_similarity", ...
#   * the raw standard_benchmarks fn name     : "hard_osimertinib", "amlodipine_rings", ...
# so existing calls that passed a raw fn name keep working.
# ============================================================================

# Friendly heatmap label  ->  guacamol v2 benchmark .name (display name).
# Only labels whose display name differs from a simple title-case are listed;
# everything else is matched by normalisation below.
_GUACAMOL_LABEL_TO_NAME = {
    "osimertinib_mpo": "Osimertinib",
    "fexofenadine_mpo": "Fexofenadine",
    "ranolazine_mpo": "Ranolazine",
    "perindopril_mpo": "Perindopril",
    "amlodipine_mpo": "Amlodipine",
    "sitagliptin_mpo": "Sitagliptin",
    "zaleplon_mpo": "Zaleplon",
    "valsartan_smarts": "Valsartan",
    "deco_hop": "Deco Hop",
    "scaffold_hop": "Scaffold Hop",
    "albuterol_similarity": "Albuterol",
    "celecoxib_rediscovery": "Celecoxib",
    "mestranol_similarity": "Mestranol",
    "thiothixene_rediscovery": "Thiothixene",
    "troglitazone_rediscovery": "Troglitazone",
    "median1": "Median molecules 1",
    "median2": "Median molecules 2",
    "isomers_c7h8n2o2": "C7H8N2O2",
    "isomers_c9h10n2o2pf2cl": "C9H10N2O2PF2Cl",
    "isomers_c11h24": "C11H24",
    # raw sb fn names -> display name (so old callers keep working)
    "hard_osimertinib": "Osimertinib",
    "hard_fexofenadine": "Fexofenadine",
    "amlodipine_rings": "Amlodipine",
    "perindopril_rings": "Perindopril",
    "sitagliptin_replacement": "Sitagliptin",
    "zaleplon_with_other_formula": "Zaleplon",
    "decoration_hop": "Deco Hop",
    "median_camphor_menthol": "Median molecules 1",
    "median_tadalafil_sildenafil": "Median molecules 2",
}


def _norm(s):
    """Normalise a name for tolerant matching: lowercase, strip non-alphanumerics."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


# cache the built suite so we don't reconstruct scorers on every call
_GUACAMOL_V2_SUITE = None


def _guacamol_suite_v2():
    global _GUACAMOL_V2_SUITE
    if _GUACAMOL_V2_SUITE is None:
        from guacamol.benchmark_suites import goal_directed_benchmark_suite
        _GUACAMOL_V2_SUITE = goal_directed_benchmark_suite("v2")
    return _GUACAMOL_V2_SUITE


def _guacamol_scoring_fn(name):
    """Return a scoring object with a .score(smiles)->float method for the named
    GuacaMol v2 goal-directed benchmark. Accepts display name, friendly/heatmap
    label, or raw standard_benchmarks function name.

    Resolution order:
      1. If `name` is a nullary function on standard_benchmarks, use it directly
         (preserves the original behaviour for those benchmarks exactly).
      2. Otherwise, match `name` (or its alias) against the v2 suite by name.
    """
    # 1. original fast path: a real nullary sb function (keeps old behaviour)
    try:
        from guacamol import standard_benchmarks as sb
        fn = getattr(sb, name, None)
        if callable(fn):
            bench = fn()
            for attr in ("objective", "wrapped_objective"):
                obj = getattr(bench, attr, None)
                if obj is not None and hasattr(obj, "score"):
                    return obj
    except Exception:
        pass  # fall through to suite resolution

    # 2. resolve against the canonical v2 suite (handles similarity/rediscovery)
    target = _GUACAMOL_LABEL_TO_NAME.get(name, name)
    tnorm = _norm(target)
    suite = _guacamol_suite_v2()

    # exact-ish match on the benchmark display name
    for bench in suite:
        if _norm(bench.name) == tnorm:
            obj = getattr(bench, "objective", None) or getattr(bench, "wrapped_objective", None)
            if obj is not None and hasattr(obj, "score"):
                return obj

    # last resort: substring match, but only if unambiguous
    hits = [b for b in suite if tnorm in _norm(b.name)]
    if len(hits) == 1:
        obj = getattr(hits[0], "objective", None)
        if obj is not None and hasattr(obj, "score"):
            return obj

    avail = ", ".join(sorted(b.name for b in suite))
    raise AttributeError(
        f"Could not resolve guacamol benchmark {name!r}. "
        f"Available v2 benchmarks: {avail}")


# ----------------------- GuacaMol component decomposition -----------------------
#
# An assembled GuacaMol MPO benchmark (e.g. hard_osimertinib) wraps a single
# objective that is a mean over K sub-scorers, each of which is a raw RDKit
# descriptor / fingerprint similarity passed through a score modifier. To train
# one GFlowNet *per component*, we reach into that assembled objective and pull
# out the individual sub-scorers, WITHOUT re-specifying any params ourselves --
# so modifiers, fingerprint types, mus/sigmas and thresholds are identical to
# the benchmark by construction (no hand-copied numbers to drift).
#
# Layout of guacamol objects (guacamol >= 0.5):
#   standard_benchmarks.hard_osimertinib() -> GoalDirectedBenchmark
#       .objective  -> ArithmeticMean/GeometricMean ScoringFunction
#           .scoring_functions -> List[ScoringFunctionWrapper-ish]
#   Each element exposes .score(smiles) -> float in [0, 1]. Some guacamol
#   versions nest the list one level deeper under .scoring_function; we handle
#   both. We treat each element's .score(smiles) as the component reward.

# Named benchmark builders in guacamol.standard_benchmarks. Add more as needed.
_GUACAMOL_BENCH_BUILDERS = {
    "osimertinib":  "hard_osimertinib",
    "fexofenadine": "hard_fexofenadine",
    "ranolazine":   "ranolazine_mpo",
    "perindopril":  "perindopril_rings",
    "amlodipine":   "amlodipine_rings",
    "sitagliptin":  "sitagliptin_replacement",
    "zaleplon":     "zaleplon_with_other_formula",
    "pioglitazone": "pioglitazone_mpo",
}


def _load_guacamol_benchmark(bench_key):
    """Return the assembled GoalDirectedBenchmark for a friendly key or a raw
    guacamol standard_benchmarks function name."""
    from guacamol import standard_benchmarks as sb
    fn_name = _GUACAMOL_BENCH_BUILDERS.get(bench_key, bench_key)
    builder = getattr(sb, fn_name, None)
    if builder is None:
        raise ValueError(
            f"No guacamol benchmark {bench_key!r} (resolved to {fn_name!r}). "
            f"Known keys: {sorted(_GUACAMOL_BENCH_BUILDERS)}")
    return builder()


def _mean_kind(objective):
    """Best-effort detection of arithmetic vs geometric aggregation, so the
    caller knows what product/sum the trained teachers should compose into."""
    name = type(objective).__name__.lower()
    if "geometric" in name:
        return "geometric"
    if "arithmetic" in name:
        return "arithmetic"
    return "unknown"


def _extract_component_scorers(objective):
    """Return a list of leaf scorer objects, each with a .score(smiles)->float.
    Handles the couple of attribute names guacamol has used for the sub-scorer
    list across versions."""
    subs = getattr(objective, "scoring_functions", None)
    if subs is None:
        subs = getattr(objective, "scoring_function", None)
    if subs is None:
        # Objective is itself a single leaf scorer.
        return [objective]
    if not isinstance(subs, (list, tuple)):
        subs = [subs]
    return list(subs)


def guacamol_components(bench_key):
    """Introspect a benchmark and return (mean_kind, [(index, label, scorer)]).
    Use this to see how many teachers to train and what each one is, e.g.:

        kind, comps = guacamol_components("osimertinib")
        # kind == "geometric"; comps has 4 entries (2x sim, TPSA, logP)
    """
    bench = _load_guacamol_benchmark(bench_key)
    objective = bench.objective
    scorers = _extract_component_scorers(objective)
    out = []
    for i, s in enumerate(scorers):
        # Label: try to describe the modifier + descriptor for sanity-checking
        label = _describe_scorer(s)
        out.append((i, label, s))
    return _mean_kind(objective), out


def _describe_scorer(scorer):
    """Human-readable one-liner for a guacamol leaf scorer, for logging/asserts.
    Purely diagnostic -- never used in the reward value itself."""
    parts = [type(scorer).__name__]
    mod = getattr(scorer, "score_modifier", None)
    if mod is not None:
        mp = type(mod).__name__
        for attr in ("mu", "sigma", "threshold"):
            v = getattr(mod, attr, None)
            if v is not None:
                mp += f" {attr}={v}"
        parts.append(mp)
    desc = getattr(scorer, "descriptor", None)
    if desc is not None:
        parts.append(getattr(desc, "__name__", str(desc)))
    fp = getattr(scorer, "fp_type", None)
    if fp is not None:
        parts.append(f"fp={fp}")
    return " | ".join(parts)


# ----------------------- Factory -----------------------

def build_reward(cfg):
    """Return a callable log_reward(mol) -> float (base log reward, beta=1)."""
    kind = cfg.reward
    floor = cfg.invalid_logr

    def _log(score):
        return math.log(score) if score > 0 else floor

    # ---- EASY reward for Ablation X: nitrogen fraction (atom-level, monotone) ----
    if kind == "nitrogen_count":
        def log_reward(molecule):
            try:
                s = _score_nitrogen_fraction(molecule)
            except Exception as e:
                _dbg(f"nitrogen_count scoring failed: {e}")
                return floor
            return _log(s)
        return log_reward

    # ---- GuacaMol 2D graph scores ----
    if kind in ("qed", "logp", "tpsa", "isomer", "similarity"):
        target_fp = None
        if kind == "similarity":
            tmol = Chem.MolFromSmiles(cfg.reward_smiles)
            if tmol is None:
                raise ValueError(f"Bad reward_smiles: {cfg.reward_smiles!r}")
            target_fp = AllChem.GetMorganFingerprintAsBitVect(
                tmol, 2, nBits=2048)

        def log_reward(molecule):
            if kind == "isomer":
                # isomer works directly on atomic numbers, no bond perception
                try:
                    s = _score_isomer(molecule, cfg.reward_formula)
                except Exception as e:
                    _dbg(f"isomer scoring failed: {e}")
                    return floor
                return _log(s)

            m = mol_to_rdkit(molecule)
            if m is None:
                return floor
            try:
                if kind == "qed":
                    s = _score_qed(m)
                elif kind == "logp":
                    s = _score_logp(m, cfg.reward_target, cfg.reward_sigma)
                elif kind == "tpsa":
                    s = _score_tpsa(m, cfg.reward_target, cfg.reward_sigma)
                else:  # similarity
                    s = _score_similarity(m, target_fp, cfg.reward_target)
            except Exception as e:
                _dbg(f"{kind} scoring failed: {e}")
                return floor
            return _log(s)
        return log_reward

    # ---- Full guacamol package passthrough ----
    if kind == "guacamol":
        scorer = _guacamol_scoring_fn(cfg.reward_smiles)

        def log_reward(molecule):
            m = mol_to_rdkit(molecule)
            if m is None:
                return floor
            try:
                smi = Chem.MolToSmiles(Chem.RemoveHs(m))
                s = float(scorer.score(smi))
            except Exception as e:
                _dbg(f"guacamol scoring failed: {e}")
                return floor
            return _log(s)
        return log_reward

    # ---- Single component of an assembled GuacaMol MPO benchmark ----
    # cfg.reward_benchmark : friendly key ("osimertinib") or raw sb fn name.
    # cfg.reward_component  : int index OR substring matched against the
    #                         diagnostic label (e.g. "TPSA", "logP", "FCFC4").
    # Trains ONE teacher on ONE component; compose K of these downstream.
    if kind == "guacamol_component":
        bench_key = cfg.reward_benchmark
        mean_kind, comps = guacamol_components(bench_key)
        sel = cfg.reward_component

        if isinstance(sel, int):
            matches = [c for c in comps if c[0] == sel]
        else:
            s = str(sel).lower()
            matches = [c for c in comps if s in c[1].lower()]
        if len(matches) == 0:
            avail = "; ".join(f"[{i}] {lbl}" for i, lbl, _ in comps)
            raise ValueError(
                f"reward_component={sel!r} matched no component of "
                f"{bench_key!r}. Available: {avail}")
        if len(matches) > 1:
            avail = "; ".join(f"[{i}] {lbl}" for i, lbl, _ in matches)
            raise ValueError(
                f"reward_component={sel!r} is ambiguous ({len(matches)} "
                f"matches): {avail}. Use an integer index instead.")

        idx, label, scorer = matches[0]
        _dbg(f"guacamol_component {bench_key} -> [{idx}] {label} "
             f"(benchmark mean = {mean_kind})")

        def log_reward(molecule):
            m = mol_to_rdkit(molecule)
            if m is None:
                return floor
            try:
                smi = Chem.MolToSmiles(Chem.RemoveHs(m))
                s = float(scorer.score(smi))
            except Exception as e:
                _dbg(f"guacamol_component scoring failed: {e}")
                return floor
            return _log(s)
        return log_reward

    # ---- RLPF force-field reward ----
    if kind == "force":
        method = cfg.force_method

        def log_reward(molecule):
            try:
                if method == "xtb":
                    rmsf = _force_rmsf_xtb(molecule)
                elif method == "mmff":
                    rmsf = _force_rmsf_mmff(molecule)
                else:
                    raise ValueError(f"Unknown force_method {method!r}")
            except Exception as e:
                _dbg(f"force reward failed: {type(e).__name__}: {e}")
                return floor
            if rmsf is None or not np.isfinite(rmsf):
                _dbg(f"RMSF invalid: {rmsf}")
                return floor
            return -rmsf  # base log reward; trainer applies -beta via reward_beta
        return log_reward

    # ---- atom stability (GEOM benchmark axis; dense, atom-level, prior-aligned)
    if kind == "atom_stability":
        dataset = getattr(cfg, "dataset", "geom")
        def log_reward(molecule):
            try:
                s = _score_atom_stability(molecule, dataset)
            except Exception as e:
                _dbg(f"atom_stability scoring failed: {e}")
                return floor
            if s is None:
                return floor                 # empty / unmapped / failure
            return _log(s)                   # log(1.0)=0 at full stability
        return log_reward

    raise ValueError(f"Unknown cfg.reward = {kind!r}")


# ----------------------- SMILES-native reward -----------------------

def _smiles_to_fake_mol(smi, seed=0xF00D):
    """Embed a SMILES into a 3D conformer and wrap it as a Quetzal-like object
    exposing .atoms (atomic numbers incl. H) and .coords (Angstrom). Only used
    for the 'force' reward, which genuinely needs geometry. Returns None if
    parsing or embedding fails."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        # retry with random coords as a fallback for awkward graphs
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass  # unoptimized geometry is still usable for a rough force reward
    conf = mol.GetConformer()
    Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
    xyz = np.array([list(conf.GetAtomPosition(i))
                    for i in range(mol.GetNumAtoms())], dtype=float)

    class _FakeMol:
        atoms = Z
        coords = xyz
    return _FakeMol()


def build_reward_smiles(cfg):
    """Like build_reward(cfg), but returns log_reward(smiles: str) -> float.

    For every graph-based reward kind (qed, logp, tpsa, isomer, similarity,
    guacamol, guacamol_component) the score depends only on the molecular graph,
    so the SMILES is parsed straight to an RDKit Mol and the existing scorers
    are reused -- NO 3D embedding, NO bond-from-geometry perception. This is
    both faster and more faithful than round-tripping through coordinates.

    For the 'force' kind, a 3D conformer is embedded from the SMILES (ETKDG +
    MMFF) since the reward is defined on geometry; embedding failures return the
    invalid floor.

    Invalid / unparseable SMILES return cfg.invalid_logr, matching build_reward.
    """
    kind = cfg.reward
    floor = cfg.invalid_logr

    def _log(score):
        return math.log(score) if score > 0 else floor

    # ---- EASY reward (Ablation X), SMILES-native: nitrogen fraction ----
    if kind == "nitrogen_count":
        def log_reward(smi):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return floor
            Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)

            class _FakeMol:
                atoms = Z
                coords = np.zeros((len(Z), 3))
            try:
                s = _score_nitrogen_fraction(_FakeMol())
            except Exception as e:
                _dbg(f"nitrogen_count scoring failed: {e}")
                return floor
            return _log(s)
        return log_reward

    # ---- force: needs geometry, so embed then defer to the mol-based path ----
    if kind == "force":
        mol_reward = build_reward(cfg)   # reuse the coords-based implementation

        def log_reward(smi):
            fake = _smiles_to_fake_mol(smi)
            if fake is None:
                _dbg(f"embed failed for {smi!r}")
                return floor
            return mol_reward(fake)
        return log_reward

    # ---- isomer: counts atoms incl. H directly from the graph ----
    if kind == "isomer":
        def log_reward(smi):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return floor
            molH = Chem.AddHs(mol)
            Z = np.array([a.GetAtomicNum() for a in molH.GetAtoms()], dtype=int)

            class _FakeMol:
                atoms = Z
                coords = np.zeros((len(Z), 3))
            try:
                s = _score_isomer(_FakeMol(), cfg.reward_formula)
            except Exception as e:
                _dbg(f"isomer scoring failed: {e}")
                return floor
            return _log(s)
        return log_reward

    # ---- qed / logp / tpsa / similarity: graph descriptors on the RDKit mol ----
    if kind in ("qed", "logp", "tpsa", "similarity"):
        target_fp = None
        if kind == "similarity":
            tmol = Chem.MolFromSmiles(cfg.reward_smiles)
            if tmol is None:
                raise ValueError(f"Bad reward_smiles: {cfg.reward_smiles!r}")
            target_fp = AllChem.GetMorganFingerprintAsBitVect(
                tmol, 2, nBits=2048)

        def log_reward(smi):
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return floor
            try:
                if kind == "qed":
                    s = _score_qed(m)
                elif kind == "logp":
                    s = _score_logp(m, cfg.reward_target, cfg.reward_sigma)
                elif kind == "tpsa":
                    s = _score_tpsa(m, cfg.reward_target, cfg.reward_sigma)
                else:  # similarity
                    s = _score_similarity(m, target_fp, cfg.reward_target)
            except Exception as e:
                _dbg(f"{kind} scoring failed: {e}")
                return floor
            return _log(s)
        return log_reward

    # ---- guacamol passthrough: scorer already takes a SMILES ----
    if kind == "guacamol":
        scorer = _guacamol_scoring_fn(cfg.reward_smiles)

        def log_reward(smi):
            if Chem.MolFromSmiles(smi) is None:
                return floor
            try:
                s = float(scorer.score(smi))
            except Exception as e:
                _dbg(f"guacamol scoring failed: {e}")
                return floor
            return _log(s)
        return log_reward

    # ---- single guacamol component: leaf scorer also takes a SMILES ----
    if kind == "guacamol_component":
        mean_kind, comps = guacamol_components(cfg.reward_benchmark)
        sel = cfg.reward_component
        if isinstance(sel, int):
            matches = [c for c in comps if c[0] == sel]
        else:
            s = str(sel).lower()
            matches = [c for c in comps if s in c[1].lower()]
        if len(matches) == 0:
            avail = "; ".join(f"[{i}] {lbl}" for i, lbl, _ in comps)
            raise ValueError(
                f"reward_component={sel!r} matched no component of "
                f"{cfg.reward_benchmark!r}. Available: {avail}")
        if len(matches) > 1:
            avail = "; ".join(f"[{i}] {lbl}" for i, lbl, _ in matches)
            raise ValueError(
                f"reward_component={sel!r} is ambiguous: {avail}. "
                "Use an integer index instead.")
        idx, label, scorer = matches[0]
        _dbg(f"guacamol_component {cfg.reward_benchmark} -> [{idx}] {label} "
             f"(benchmark mean = {mean_kind})")

        def log_reward(smi):
            if Chem.MolFromSmiles(smi) is None:
                return floor
            try:
                s = float(scorer.score(smi))
            except Exception as e:
                _dbg(f"guacamol_component scoring failed: {e}")
                return floor
            return _log(s)
        return log_reward

    # ---- atom stability from SMILES: needs geometry -> embed then score ----
    if kind == "atom_stability":
        dataset = getattr(cfg, "dataset", "geom")
        def log_reward(smi):
            fake = _smiles_to_fake_mol(smi)
            if fake is None:
                return floor
            try:
                s = _score_atom_stability(fake, dataset)
            except Exception as e:
                _dbg(f"atom_stability scoring failed: {e}")
                return floor
            return floor if s is None else _log(s)
        return log_reward

    raise ValueError(f"Unknown cfg.reward = {kind!r}")


# ----------------------- Standalone diagnostic -----------------------
if __name__ == "__main__":
    """Self-test: exercise EVERY reward kind on a small panel of molecules.

    Run:  python reward_fn.py
          python reward_fn.py --quick      (skip force/xTB, the slow ones)
          python reward_fn.py --bench hard_fexofenadine

    Checks, per reward kind:
      * does build_reward(cfg) construct without raising?
      * does it return a finite float on a real molecule?
      * does it return exactly invalid_logr on garbage?      (floor behaviour)
      * does it VARY across chemically different molecules?  (a reward with zero
        spread is a DEAD AXIS -- nothing can be steered toward it, so this is the
        single most useful thing the self-test can tell you)
      * does build_reward_smiles(cfg) agree with build_reward(cfg)?

    Exit code is non-zero if any kind errors out, so this is usable in CI.
    """
    import sys
    import argparse
    import traceback

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the force rewards (xTB/MMFF are slow)")
    ap.add_argument("--bench", default="hard_osimertinib",
                    help="guacamol benchmark for the guacamol / component kinds")
    ap.add_argument("--bench_key", default="osimertinib",
                    help="friendly key for guacamol_component")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    DEBUG_REWARD = args.verbose
    if not args.verbose:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")     # Morgan deprecation spam drowns the table

    # ---------------- the molecule panel ----------------
    # Deliberately spread out: tiny/large, aliphatic/aromatic, N-rich/N-free,
    # halogenated, and a real drug. If a reward can't tell these apart, it can't
    # tell anything apart.
    PANEL = [
        ("ethanol",      "CCO"),
        ("benzene",      "c1ccccc1"),
        ("caffeine",     "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
        ("aspirin",      "CC(=O)Oc1ccccc1C(=O)O"),
        ("gefitinib-ish", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"),
        ("adenine",      "Nc1ncnc2[nH]cnc12"),
        ("hexane",       "CCCCCC"),
    ]

    def make_fake(smi, seed=0xF00D):
        """SMILES -> a Quetzal-like object with .atoms (atomic numbers, incl H)
        and .coords (Angstrom). Same construction the trainer's molecules have."""
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        m = Chem.AddHs(m)
        if AllChem.EmbedMolecule(m, randomSeed=seed) != 0:
            p = AllChem.ETKDGv3(); p.randomSeed = seed; p.useRandomCoords = True
            if AllChem.EmbedMolecule(m, p) != 0:
                return None
        try:
            AllChem.MMFFOptimizeMolecule(m)
        except Exception:
            pass
        conf = m.GetConformer()

        class _FakeMol:
            atoms = np.array([a.GetAtomicNum() for a in m.GetAtoms()], dtype=int)
            coords = np.array([list(conf.GetAtomPosition(i))
                               for i in range(m.GetNumAtoms())], dtype=float)
        return _FakeMol()

    print("=" * 72)
    print("building 3D conformers for the test panel")
    mols = []
    for name, smi in PANEL:
        fm = make_fake(smi)
        if fm is None:
            print(f"  [skip] {name}: could not embed")
            continue
        mols.append((name, smi, fm))
        print(f"  {name:<14} {len(fm.atoms):>3} atoms (with H)")
    if not mols:
        sys.exit("FATAL: could not embed any test molecule -- is RDKit working?")

    # ---------------- 3D -> RDKit round trip ----------------
    # Everything graph-based flows through mol_to_rdkit, so if bond perception is
    # broken every reward below silently returns the floor.
    print("\n" + "=" * 72)
    print("mol_to_rdkit (3D bond perception) -- the gate every reward passes through")
    n_ok = 0
    for name, smi, fm in mols:
        rd = mol_to_rdkit(fm)
        if rd is None:
            print(f"  {name:<14} FAILED")
            continue
        got = Chem.MolToSmiles(Chem.RemoveHs(rd))
        want = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        match = "match" if got == want else f"DIFFERENT -> {got}"
        print(f"  {name:<14} {match}")
        n_ok += 1
    print(f"  {n_ok}/{len(mols)} recovered")
    if n_ok == 0:
        sys.exit("FATAL: mol_to_rdkit failed on everything; check rdkit version "
                 "(2023.03.3 expected)")

    # ---------------- config bag ----------------
    class Cfg:
        """Plain attribute bag. build_reward only reads the fields its kind needs,
        so setting all of them is harmless."""
        def __init__(self, **kw):
            self.reward = None
            self.invalid_logr = -5.0
            self.reward_target = 0.0
            self.reward_sigma = 1.0
            self.reward_smiles = None
            self.reward_formula = None
            self.reward_benchmark = args.bench_key
            self.reward_component = 0
            self.force_method = "mmff"
            self.dataset = "geom"
            self.__dict__.update(kw)

    # (label, cfg kwargs, needs_guacamol)
    CASES = [
        ("qed",              dict(reward="qed"), False),
        ("logp(t=2,s=1)",    dict(reward="logp", reward_target=2.0, reward_sigma=1.0), False),
        ("tpsa(t=60,s=20)",  dict(reward="tpsa", reward_target=60.0, reward_sigma=20.0), False),
        ("similarity(aspirin)",
                             dict(reward="similarity",
                                  reward_smiles="CC(=O)Oc1ccccc1C(=O)O",
                                  reward_target=0.3), False),
        ("isomer(C6H6)",     dict(reward="isomer", reward_formula="C6H6"), False),
        ("nitrogen_count",   dict(reward="nitrogen_count"), False),
        ("atom_stability",   dict(reward="atom_stability", dataset="geom"), False),
        (f"guacamol({args.bench})",
                             dict(reward="guacamol", reward_smiles=args.bench), True),
    ]
    # one case per component of the MPO, so a dead axis shows up by name
    for ci in range(4):
        CASES.append((f"gcomp({args.bench_key}:{ci})",
                      dict(reward="guacamol_component",
                           reward_benchmark=args.bench_key,
                           reward_component=ci), True))
    if not args.quick:
        CASES.append(("force(mmff)", dict(reward="force", force_method="mmff"), False))
        CASES.append(("force(xtb)",  dict(reward="force", force_method="xtb"), False))

    # An EMPTY molecule is the one input mol_to_rdkit rejects unconditionally
    # ("empty molecule after truncation"), so it is the honest floor probe.
    # Two overlapping carbons is NOT garbage -- bond perception happily returns
    # ethane and the reward scores it, which is correct behaviour.
    GARBAGE = type("Garbage", (), {"atoms": np.array([], dtype=int),
                                   "coords": np.zeros((0, 3))})()

    print("\n" + "=" * 72)
    print("reward kinds  (spread = std over the panel; ~0 means a DEAD AXIS)")
    print("=" * 72)
    hdr = f"{'reward':<26} {'built':>6} {'finite':>7} {'floor':>6} {'mean':>9} {'spread':>8}"
    print(hdr); print("-" * len(hdr))

    failures, dead_axes, all_floor = [], [], []
    per_kind_values = {}

    for label, kw, needs_gc in CASES:
        cfg = Cfg(**kw)
        try:
            fn = build_reward(cfg)
        except Exception as e:
            if needs_gc and isinstance(e, (ImportError, ModuleNotFoundError)):
                print(f"{label:<26} {'skip':>6}   (guacamol not installed)")
                continue
            print(f"{label:<26} {'FAIL':>6}   {type(e).__name__}: {e}")
            failures.append((label, f"build: {type(e).__name__}: {e}"))
            if args.verbose:
                traceback.print_exc()
            continue

        vals, errs = [], []
        for name, smi, fm in mols:
            try:
                v = float(fn(fm))
            except Exception as e:
                errs.append(f"{name}: {type(e).__name__}: {e}")
                v = float("nan")
            vals.append(v)
        arr = np.array(vals, float)
        finite = np.isfinite(arr)
        above_floor = finite & (arr > cfg.invalid_logr + 1e-9)

        # garbage must land exactly on the floor, not raise and not score
        try:
            gv = float(fn(GARBAGE))
            floor_ok = abs(gv - cfg.invalid_logr) < 1e-6 or not np.isfinite(gv)
        except Exception as e:
            floor_ok = False
            errs.append(f"garbage raised: {type(e).__name__}: {e}")

        spread = float(np.std(arr[above_floor])) if above_floor.sum() > 1 else 0.0
        mean = float(np.mean(arr[finite])) if finite.any() else float("nan")
        if above_floor.sum() == 0:
            note = "   <-- ALL FLOOR (scorer unavailable or every mol rejected)"
        elif above_floor.sum() > 1 and spread < 0.05:
            note = "   <-- DEAD AXIS"
        else:
            note = ""
        print(f"{label:<26} {'ok':>6} {int(finite.sum()):>4}/{len(arr):<2} "
              f"{('ok' if floor_ok else 'BAD'):>6} {mean:>9.3f} {spread:>8.4f}{note}")

        per_kind_values[label] = dict(zip([m[0] for m in mols], vals))
        if errs:
            failures.append((label, "; ".join(errs[:3])))
            for e in errs[:3]:
                print(f"    error: {e}")
        if not floor_ok:
            failures.append((label, f"garbage did not return invalid_logr "
                                    f"({cfg.invalid_logr})"))
        if above_floor.sum() == 0:
            all_floor.append(label)
        elif above_floor.sum() > 1 and spread < 0.05:
            dead_axes.append(label)

    # ---------------- build_reward vs build_reward_smiles ----------------
    # The SMILES path is what smiles_hist.py and the harvest scripts use. If it
    # disagrees with the 3D path, every offline analysis is scoring something
    # different from what training optimised.
    print("\n" + "=" * 72)
    print("build_reward (3D) vs build_reward_smiles (SMILES) -- must agree")
    print("=" * 72)
    for label, kw, needs_gc in CASES:
        if kw.get("reward") in ("force", "atom_stability"):
            continue          # genuinely geometry-dependent; no SMILES analogue
        cfg = Cfg(**kw)
        try:
            f3d = build_reward(cfg)
            fsm = build_reward_smiles(cfg)
        except Exception:
            continue
        diffs = []
        for name, smi, fm in mols:
            try:
                a, b = float(f3d(fm)), float(fsm(smi))
            except Exception:
                continue
            if np.isfinite(a) and np.isfinite(b) and abs(a - b) > 1e-4:
                diffs.append(f"{name}: 3D={a:.4f} smi={b:.4f}")
        if diffs:
            print(f"  {label:<26} MISMATCH  {diffs[0]}"
                  + (f"  (+{len(diffs)-1} more)" if len(diffs) > 1 else ""))
            failures.append((label, f"3D/SMILES mismatch: {diffs[0]}"))
        else:
            print(f"  {label:<26} agree")

    # ---------------- per-molecule detail for the headline reward ----------------
    key = next((k for k in per_kind_values if k.startswith("guacamol(")), None)
    if key:
        print("\n" + "=" * 72)
        print(f"per-molecule scores: {key}")
        for name, v in sorted(per_kind_values[key].items(), key=lambda kv: -kv[1]):
            print(f"  {name:<14} {v:>8.4f}")

    # ---------------- verdict ----------------
    print("\n" + "=" * 72)
    if dead_axes:
        print(f"DEAD AXES ({len(dead_axes)}): {', '.join(dead_axes)}")
        print("  These have no spread over the panel. A guide cannot steer them --")
        print("  a flat result on one of these is expected, not a training bug.")
    if all_floor:
        print(f"ALL AT FLOOR ({len(all_floor)}): {', '.join(all_floor)}")
        print("  Nothing scored above invalid_logr. Usually a missing optional")
        print("  dependency (edm_metrics, guacamol, xtb) rather than chemistry --")
        print("  re-run with --verbose to see the swallowed exception.")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for label, msg in failures:
            print(f"  {label}: {msg}")
        sys.exit(1)
    print("all reward kinds built, scored, floored and round-tripped cleanly")
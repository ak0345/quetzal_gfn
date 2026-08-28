"""
figstyle.py -- shared style, artifact paths and helpers for the paper figures.

Every make_fig*.py script reads COMMITTED ARTIFACTS (CSV/JSON under results/)
rather than re-running a model, so figures regenerate on a laptop in seconds and
do not depend on a GPU or on checkpoints being present.

Where a figure's input has not been produced yet, the script exits with the
exact command that produces it rather than emitting a half-empty plot.
"""
import os
import re
import sys
import json
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# repo root, one level above figures/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel(*parts):
    return os.path.join(ROOT, *parts)


# ----------------------------------------------------------------- artifacts
DUMPS_AGG   = rel("results", "dumps", "_aggregate", "master_table.csv")
FLIPS_DIR   = rel("results", "flips-guide")
FLIPS_AGGS  = rel("results", "flips-guide", "_aggs")
HARVEST_DIR = rel("results", "oracle_gfn_mols", "_results")
BON_DIR     = rel("results", "best_of_n")
ABL_DIR     = rel("results", "ablations")

HARVEST = {
    "osim":     os.path.join(HARVEST_DIR, "hard_osimertinib_budget10000.json"),
    "peri":     os.path.join(HARVEST_DIR, "perindopril_rings_budget10000.json"),
    "zaleplon": os.path.join(HARVEST_DIR,
                             "zaleplon_with_other_formula_budget10000.json"),
    "fexo":     os.path.join(HARVEST_DIR, "hard_fexofenadine_budget10000.json"),
}

BENCH_TITLE = {"osim": "Osimertinib MPO", "peri": "Perindopril MPO",
               "zaleplon": "Zaleplon MPO", "fexo": "Fexofenadine MPO"}

# Every benchmark a figure may be asked for, in the order they should be drawn.
# `osim` stays first because the paper is written around it, but no fine-tuning
# run in this repository targets it, so it is normally absent from the harvest
# directory -- see available_benches().
ALL_BENCHES = ("osim", "peri", "zaleplon", "fexo")

# Published GuacaMol baselines, from Brown et al. (2019). Constants from the
# literature, not measurements of anything in this repository, which is why they
# are written here rather than read from an artifact.
PUBLISHED = {
    "osim": {"REINVENT SMILES": 0.837, "ChEMBL best-of-dataset": 0.839},
    "peri": {"REINVENT SMILES": 0.537, "ChEMBL best-of-dataset": 0.575},
    "fexo": {"REINVENT SMILES": 0.784, "ChEMBL best-of-dataset": 0.817},
    # Zaleplon has no row here yet. make_fig01_landscape.py prints a notice for
    # any benchmark missing from this table and draws the panel with the GEOM
    # line only, rather than silently omitting the comparison. To add one, copy
    # the row out of Brown et al. (2019) Table 2:
    #   "zaleplon": {"REINVENT SMILES": <x>, "ChEMBL best-of-dataset": <y>},
}

# PMO (Gao et al., 2022), Table 2: AUC top-10 at a 10,000-call oracle budget,
# mean and standard deviation over five runs. This is the directly comparable
# convention, since our fine-tuned runs report AUC top-10 at the same budget.
#
# CAVEAT: PMO states that its sitagliptin_mpo and zaleplon_mpo differ from the
# GuacaMol implementations. Our Zaleplon is GuacaMol's
# `zaleplon_with_other_formula`, so the zaleplon column below is NOT a
# like-for-like comparison and is marked as such wherever it is drawn.
PMO_AUC_TOP10 = {
    #                  osim            peri            zaleplon
    "REINVENT":       {"osim": (0.837, 0.009), "peri": (0.537, 0.016), "zaleplon": (0.358, 0.062)},
    "Graph GA":       {"osim": (0.831, 0.005), "peri": (0.538, 0.009), "zaleplon": (0.346, 0.032)},
    "REINVENT SELFIES": {"osim": (0.820, 0.003), "peri": (0.517, 0.021), "zaleplon": (0.333, 0.026)},
    "GP BO":          {"osim": (0.787, 0.006), "peri": (0.493, 0.011), "zaleplon": (0.221, 0.072)},
    "STONED":         {"osim": (0.822, 0.012), "peri": (0.488, 0.011), "zaleplon": (0.325, 0.027)},
    "LSTM HC":        {"osim": (0.796, 0.002), "peri": (0.489, 0.007), "zaleplon": (0.206, 0.006)},
    "SMILES GA":      {"osim": (0.817, 0.011), "peri": (0.447, 0.013), "zaleplon": (0.334, 0.041)},
    "SynNet":         {"osim": (0.796, 0.003), "peri": (0.557, 0.011), "zaleplon": (0.341, 0.011)},
    "DoG-Gen":        {"osim": (0.774, 0.002), "peri": (0.474, 0.002), "zaleplon": (0.123, 0.016)},
    "DST":            {"osim": (0.785, 0.004), "peri": (0.462, 0.008), "zaleplon": (0.176, 0.045)},
}
PMO_NOT_COMPARABLE = ("zaleplon",)   # PMO reimplements this benchmark

# ----------------------------------------------------------------- palette
FAMILY_COLOURS = {
    "guide: base":     "#8C8C8C",
    "guide: hidden":   "#4C72B0",
    "guide: tempgain": "#DD8452",
    "composed":        "#937860",
    "FT: proj":        "#55A868",
    "FT: atom":        "#C44E52",
    "FT: full":        "#000000",
    "FT: LoRA":        "#64B5CD",
    "frozen prior":    "#DA8BC3",
    # capacity figure uses a finer split
    "LoRA (proj)":     "#64B5CD",
    "LoRA (atom)":     "#4C72B0",
    "proj":            "#55A868",
    "atom":            "#C44E52",
    "full":            "#000000",
}

GUIDE_COLOURS = {"base": "#8C8C8C", "hidden": "#4C72B0", "tempgain": "#DD8452"}
REF_COLOUR = "crimson"

# One colour per objective family, for figures that put every family in a single
# space. Distinct from GUIDE_COLOURS, which splits by guide architecture.
BENCH_COLOURS = {
    "osim":     "#4C72B0",
    "peri":     "#DD8452",
    "fexo":     "#55A868",
    "zaleplon": "#C44E52",
    "nitrogen": "#937860",
}
BACKDROP_COLOURS = {"GEOM": "#BDBDBD", "prior": "#6E6E6E"}


def use_paper_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
    })


# ----------------------------------------------------------------- loading
def die(msg, how=None):
    print(f"[fig] {msg}", file=sys.stderr)
    if how:
        print(f"[fig] produce it with:\n    {how}", file=sys.stderr)
    raise SystemExit(2)


def need(path, how=None):
    if not os.path.exists(path):
        die(f"missing input: {path}", how)
    return path


def load_json(path, how=None):
    need(path, how)
    with open(path) as f:
        return json.load(f)


BENCH_FN = {"osim": "hard_osimertinib", "peri": "perindopril_rings",
            "zaleplon": "zaleplon_with_other_formula",
            "fexo": "hard_fexofenadine"}

# The `MATCH=` value that selects the fine-tuning runs belonging to a benchmark,
# used only to print a runnable command when a harvest file is missing.
BENCH_MATCH = {"osim": "osim", "peri": "peri", "zaleplon": "zaleplon",
               "fexo": "fexo"}


def harvest_cmd(bench):
    """The command that produces this benchmark's harvest JSON."""
    return (f"BENCH={BENCH_FN.get(bench, bench)} "
            f"MATCH={BENCH_MATCH.get(bench, bench)} "
            f"bash scripts/08_analysis.sh harvest")


def available_benches(candidates=None):
    """Those benchmarks whose harvest JSON has actually been written.

    A figure that spans several benchmarks asks for this rather than assuming
    the paper's pair, so a run of the pipeline that covers a different set of
    objectives still produces the figure over what it does have instead of
    exiting.
    """
    return [b for b in (candidates or ALL_BENCHES)
            if b in HARVEST and os.path.exists(HARVEST[b])]


def default_bench(candidates=None):
    """The populated benchmark carrying the most scored runs.

    Single-panel figures use this instead of a hard-coded name so they land on
    the objective the pipeline actually covered rather than on whichever one the
    paper happened to be written around.
    """
    have = available_benches(candidates)
    if not have:
        die("no harvest JSON found for any benchmark",
            how=harvest_cmd((candidates or ALL_BENCHES)[0]))

    def n_runs(b):
        try:
            with open(HARVEST[b]) as fh:
                return sum(1 for k in json.load(fh) if not k.startswith("_"))
        except (OSError, ValueError):
            return 0

    return max(have, key=n_runs)


def resolve_benches(arg, candidates=None):
    """Map a --bench value to a concrete, populated list of benchmarks.

    "auto"/"both"/"all" mean "whatever has data"; a named benchmark is honoured
    even if it is missing, so an explicit request still fails loudly with the
    command that would produce it.
    """
    if arg in ("auto", "both", "all"):
        have = available_benches(candidates)
        if not have:
            die("no harvest JSON found for any benchmark",
                how=harvest_cmd((candidates or ALL_BENCHES)[0]))
        return have
    return [arg]


def load_harvest(bench):
    """Harvest JSON for one benchmark: {run_name: {...}, '_reference': {...}}."""
    return load_json(HARVEST[bench], how=harvest_cmd(bench))


def load_reference(bench, budget=10000):
    """The GEOM best-of-N baseline for one benchmark, or None.

    Normally this rides along in the harvest as `_reference`, but it depends
    only on the dataset and the objective, so a benchmark with no fine-tuning
    runs can still have one written standalone by make_reference.py. Prefer the
    harvest's copy and fall back to the standalone file.
    """
    if os.path.exists(HARVEST.get(bench, "")):
        ref = (load_json(HARVEST[bench]) or {}).get("_reference")
        if ref:
            return ref
    fn = BENCH_FN.get(bench, bench)
    path = os.path.join(HARVEST_DIR, f"_reference_{fn}_{budget}.json")
    if os.path.exists(path):
        return load_json(path)
    return None


def load_master_table():
    """The guide sweep's aggregated table, one row per run."""
    need(DUMPS_AGG, how="bash scripts/04_dump_guides.sh")
    with open(DUMPS_AGG) as f:
        return list(csv.DictReader(f))


def load_flip_reports(root=None, temp="1.0"):
    """Every per-run flip report, as {label: block_at_that_temperature}."""
    import glob
    root = root or FLIPS_DIR
    paths = sorted(glob.glob(os.path.join(root, "flip_report_*.json")))
    if not paths:
        die(f"no flip reports under {root}",
            how="bash scripts/06_flip_diagnostics.sh")
    out = {}
    for p in paths:
        d = json.load(open(p))
        blk = d.get(f"flip_temp{temp}")
        if blk is None:
            continue
        out[d.get("label") or os.path.basename(p)] = (d, blk)
    if not out:
        die(f"no flip report contains temperature {temp}",
            how="TEMPS='1.0 0.3' bash scripts/06_flip_diagnostics.sh")
    return out


def f(x, default=np.nan):
    """Float or NaN -- CSV cells are strings and may be empty."""
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------- naming
# Run directories carry the training seed as a trailing "-s<k>". It identifies
# the run on disk but says nothing a reader of the paper needs, so it is stripped
# from every label that reaches a figure.
_SEED_SUFFIX = re.compile(r"-s\d+$")


def clean_label(name):
    """A run name as it should appear in a figure: no seed suffix."""
    return _SEED_SUFFIX.sub("", str(name))


def distinguishing_labels(names):
    """Shorten a set of run names to just the parts that differ between them.

    A probe is usually run over guides whose names share everything but one
    field -- `sweep-osim-base-db-replay_off-b10` against `...-hidden-...` -- and
    written out in full those labels are far too wide for a categorical axis and
    overlap into illegibility. Dropping the shared leading and trailing
    hyphen-separated fields leaves exactly the part a reader needs.

    Falls back to the full cleaned names whenever the result would be empty or
    ambiguous, so this can never merge two distinct runs into one label.
    """
    cleaned = [clean_label(n) for n in names]
    if len(cleaned) < 2:
        return cleaned

    parts = [c.split("-") for c in cleaned]
    n_min = min(len(p) for p in parts)

    head = 0
    while head < n_min - 1 and len({p[head] for p in parts}) == 1:
        head += 1
    tail = 0
    while tail < n_min - head - 1 and len({p[-1 - tail] for p in parts}) == 1:
        tail += 1

    short = ["-".join(p[head:len(p) - tail]) for p in parts]
    if any(not s for s in short) or len(set(short)) != len(set(cleaned)):
        return cleaned
    return short


def guide_family(name):
    """Map a sweep run name to the family label used in the legend."""
    for g in ("hidden", "tempgain", "base"):
        if f"-{g}-" in name:
            return f"guide: {g}"
    return None


def ft_family(name):
    """Map a fine-tuning run name to its family label."""
    if "lora" in name:
        return "FT: LoRA"
    if "-full-" in name:
        return "FT: full"
    if "-atom-" in name:
        return "FT: atom"
    if "-proj-" in name or name.startswith("rtb-proj"):
        return "FT: proj"
    return "FT: proj"


def save(fig, out, dpi=300):
    d = os.path.dirname(os.path.abspath(out))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def add_arg_common(ap, default_out):
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--width", type=float, default=7.2)
    return ap

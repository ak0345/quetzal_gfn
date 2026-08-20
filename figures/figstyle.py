"""
figstyle.py -- shared style, artifact paths and helpers for the paper figures.

Every make_fig*.py script reads COMMITTED ARTIFACTS (CSV/JSON under results/)
rather than re-running a model, so figures regenerate on a laptop in seconds and
do not depend on a GPU or on checkpoints being present.

Where a figure's input has not been produced yet, the script exits with the
exact command that produces it rather than emitting a half-empty plot.
"""
import os
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
    "osim": os.path.join(HARVEST_DIR, "hard_osimertinib_budget10000.json"),
    "peri": os.path.join(HARVEST_DIR, "perindopril_rings_budget10000.json"),
}

BENCH_TITLE = {"osim": "Osimertinib MPO", "peri": "Perindopril MPO"}

# Published GuacaMol baselines, from Brown et al. (2019). These are constants
# from the literature, not measurements of anything in this repository, which is
# why they are written here rather than read from an artifact.
PUBLISHED = {
    "osim": {"REINVENT SMILES": 0.837, "ChEMBL best-of-dataset": 0.839},
    "peri": {"REINVENT SMILES": 0.537, "ChEMBL best-of-dataset": 0.575},
}

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


def load_harvest(bench):
    """Harvest JSON for one benchmark: {run_name: {...}, '_reference': {...}}."""
    return load_json(
        HARVEST[bench],
        how=f"bash scripts/08_analysis.sh harvest   "
            f"# BENCH={'hard_osimertinib' if bench == 'osim' else 'perindopril_rings'}")


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

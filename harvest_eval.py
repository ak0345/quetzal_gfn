"""
Score a training run as a goal-directed benchmark, WITHOUT retraining.

`rtb_finetune.py --record_dir DIR` logs every molecule generated during
training, in order, to DIR/molecules.jsonl. Each row's `i` is the oracle call
index. This script slices that stream at a budget, dedupes, rescores with the
benchmark's own scoring function, and reports the standard metrics.

Two budget conventions are emitted, because they are not comparable:

  unbounded : top-k over every molecule ever generated. The original GuacaMol
              leaderboard imposed NO oracle budget (baselines used millions of
              calls), so this is the leaderboard-parity number.
  budgeted  : top-k over the first --budget calls (default 10,000), plus
              AUC-top-10 over the budget. This matches the PMO convention
              (Gao et al., 2022), where AUC-top-10 -- not the final value -- is
              the headline metric, because it rewards sample efficiency. An
              amortised sampler should look best here against a genetic
              algorithm, so compute it even if you lead with the GuacaMol-style
              number.

USAGE
-----
python harvest_eval.py --record_dir records/ft-proj-osim-b10 \
    --bench hard_osimertinib --budget 10000 --out results/ft-proj-osim-b10.json

# score the same stream against a different benchmark (no retraining needed)
python harvest_eval.py --record_dir records/ft-proj-osim-b10 \
    --bench hard_fexofenadine --budget 10000

# compare several runs side by side
python harvest_eval.py --record_dir records/ft-proj-osim-b10 \
       records/ft-lora16-osim-b10 records/ft-full-osim-b10 \
    --bench hard_osimertinib --budget 10000 --csv results/compare.csv
"""

import os
import re
import csv
import json
import glob
import random
import argparse

import numpy as np
import numpy as np, scipy
if not hasattr(scipy, "histogram"): scipy.histogram = np.histogram
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

try:
    import harvest_analysis as HA
except Exception as _e:
    HA = None
    print(f"[warn] harvest_analysis unavailable ({_e}); extended metrics off")


# ---------------------------- scoring ----------------------------

def get_scorer(bench_key):
    """The benchmark's own objective, returning a modified score in [0,1].

    Reuses reward_fn._guacamol_scoring_fn so the scorer is byte-identical to
    the one used during training (same guacamol version, same assembly).
    """
    from reward_fn import _guacamol_scoring_fn
    return _guacamol_scoring_fn(bench_key)


class CachedScorer:
    def __init__(self, scorer):
        self.scorer = scorer
        self.cache = {}
        self.n_calls = 0
        self.n_fail = 0

    def __call__(self, smi):
        if smi in self.cache:
            return self.cache[smi]
        try:
            s = float(self.scorer.score(smi))
        except Exception:
            s = 0.0
            self.n_fail += 1
        self.n_calls += 1
        self.cache[smi] = s
        return s


# ---------------------------- loading ----------------------------

def load_records(record_dir):
    """Read molecules.jsonl in order. Returns a list of dicts sorted by `i`."""
    path = os.path.join(record_dir, "molecules.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no molecules.jsonl in {record_dir}")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from a killed run
    rows.sort(key=lambda r: r["i"])
    # a resumed run can duplicate indices if the jsonl was copied around
    seen, dedup = set(), []
    for r in rows:
        if r["i"] in seen:
            continue
        seen.add(r["i"])
        dedup.append(r)
    return dedup


def canonical(smi):
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return None


# ---------------------------- metrics ----------------------------

def topk_mean(scores, k):
    if not scores:
        return float("nan")
    s = sorted(scores, reverse=True)[:k]
    return float(np.mean(s))


def auc_topk(scored_stream, k, budget, every=100):
    """Area under (top-k mean vs oracle calls), normalised to [0,1].

    scored_stream : list of (call_index, score) in call order, deduped.
    Follows the PMO convention: the curve is evaluated on a grid and
    trapezoid-integrated, then divided by the budget.
    """
    if not scored_stream:
        return float("nan")
    grid = list(range(every, budget + 1, every))
    if not grid or grid[-1] != budget:
        grid.append(budget)

    # start the curve at (0 calls, score 0): nothing has been sampled yet
    xs, ys = [0], [0.0]
    ptr = 0
    running = []
    for g in grid:
        while ptr < len(scored_stream) and scored_stream[ptr][0] < g:
            running.append(scored_stream[ptr][1])
            ptr += 1
        ys.append(topk_mean(running, k) if running else 0.0)
        xs.append(g)
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed it
    return float(trapz(ys, xs) / budget)


def evaluate(rows, scorer, budget, topks=(1, 10, 100), auc_every=100,
             use_stored=False):
    """Score a prefix of the generation stream."""
    window = [r for r in rows if r["i"] < budget] if budget else rows
    n_calls = len(window)

    seen = set()
    stream = []          # (call_index, score) for first occurrences only
    n_smiles = 0
    for r in window:
        smi = canonical(r.get("smiles"))
        if smi is None:
            continue
        n_smiles += 1
        if smi in seen:
            continue
        seen.add(smi)
        if use_stored:
            lr = r.get("log_reward")
            score = float(np.exp(lr)) if lr is not None and lr > -20 else 0.0
        else:
            score = scorer(smi)
        stream.append((r["i"], score))

    scores = [s for _, s in stream]
    out = {
        "n_oracle_calls": n_calls,
        "n_valid_smiles": n_smiles,
        "n_unique_smiles": len(stream),
        "smiles_conversion_rate": (n_smiles / n_calls) if n_calls else float("nan"),
        "uniqueness_among_valid": (len(stream) / n_smiles) if n_smiles else float("nan"),
        "score_mean": float(np.mean(scores)) if scores else float("nan"),
    }
    for k in topks:
        out[f"top{k}"] = topk_mean(scores, k)
    if budget:
        out["auc_top10"] = auc_topk(stream, 10, budget, every=auc_every)
    best = sorted(stream, key=lambda t: -t[1])[:100]
    out["_best"] = [{"i": i, "score": s} for i, s in best]
    # attach the SMILES for the best ones
    idx_to_smi = {}
    for r in window:
        idx_to_smi[r["i"]] = canonical(r.get("smiles"))
    for b in out["_best"]:
        b["smiles"] = idx_to_smi.get(b["i"])
    out["_stream"] = stream
    out["_idx_to_smi"] = idx_to_smi
    return out


# ---------------------------- main ----------------------------

def discover_record_dirs(roots, pattern=None, exclude=None):
    """Recursively find every directory containing a molecules.jsonl.

    Runs live in different places under a root, so this walks rather than
    globbing one level. `pattern` and `exclude` are regexes matched against the
    full path (case-insensitive), e.g. pattern='osim'.
    """
    pat = re.compile(pattern, re.I) if pattern else None
    exc = re.compile(exclude, re.I) if exclude else None
    found = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if "molecules.jsonl" not in filenames:
                continue
            if pat and not pat.search(dirpath):
                continue
            if exc and exc.search(dirpath):
                continue
            found.append(dirpath)
    return sorted(set(found))


def run_names(dirs):
    """Basename where unique, else a path suffix -- two runs in different
    parents can share a basename and would otherwise overwrite each other."""
    base = [os.path.basename(os.path.normpath(d)) for d in dirs]
    dup = {b for b in base if base.count(b) > 1}
    names = []
    for d, b in zip(dirs, base):
        if b in dup:
            parts = os.path.normpath(d).split(os.sep)
            names.append("/".join(parts[-2:]) if len(parts) >= 2 else b)
        else:
            names.append(b)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record_dir", nargs="*", default=[],
                    help="explicit directories written by --record_dir")
    ap.add_argument("--record_root", nargs="*", default=[],
                    help="root(s) to search recursively for molecules.jsonl")
    ap.add_argument("--match", default=None,
                    help="regex the run path must match, e.g. 'osim'")
    ap.add_argument("--exclude", default=None,
                    help="regex of paths to skip, e.g. '_calib|sanity'")
    ap.add_argument("--bench", default="hard_osimertinib",
                    help="guacamol benchmark key, e.g. hard_osimertinib")
    ap.add_argument("--budget", type=int, default=10000,
                    help="oracle-call budget for the budgeted variant")
    ap.add_argument("--topk", default="1,10,100")
    ap.add_argument("--auc_every", type=int, default=100)
    ap.add_argument("--use_stored_reward", action="store_true",
                    help="use exp(log_reward) from training instead of "
                         "rescoring (only valid if the training reward WAS "
                         "this benchmark at beta=1)")
    ap.add_argument("--no_unbounded", action="store_true",
                    help="skip the full-stream (leaderboard-parity) variant")
    ap.add_argument("--out", default=None, help="write JSON here")
    ap.add_argument("--csv", default=None, help="write a comparison CSV here")
    ap.add_argument("--dump_best", default=None,
                    help="directory for per-run top-100 SMILES CSVs")
    # ---- extended analysis ----
    ap.add_argument("--extended", action="store_true",
                    help="quality filters, diversity, descriptors, sample "
                         "efficiency, collapse curves, MPO component breakdown")
    ap.add_argument("--plots_dir", default=None,
                    help="write PNG figures here (implies --extended)")
    ap.add_argument("--ref_smiles", default=None,
                    help="reference SMILES file (one per line, e.g. a GEOM or "
                         "ChEMBL subset) for novelty, NN-similarity, the "
                         "score-distribution overlay and a best-of-dataset row")
    ap.add_argument("--ref_limit", type=int, default=5000,
                    help="max reference molecules to score (scoring the whole "
                         "of GEOM is slow and unnecessary)")
    ap.add_argument("--ref_seed", type=int, default=0,
                    help="seed for the reference subsample; fixed so the "
                         "best-of-dataset baseline is reproducible")
    ap.add_argument("--analysis_topn", type=int, default=100,
                    help="how many top molecules feed the chemistry analyses")
    ap.add_argument("--buckets", type=int, default=20,
                    help="number of windows for the collapse curves")
    args = ap.parse_args()

    topks = tuple(int(t) for t in args.topk.split(",") if t.strip())

    record_dirs = list(args.record_dir)
    if args.record_root:
        found = discover_record_dirs(args.record_root, args.match, args.exclude)
        print(f"[find] {len(found)} run(s) under {', '.join(args.record_root)}"
              + (f" matching /{args.match}/" if args.match else ""))
        for d in found:
            print(f"       {d}")
        record_dirs.extend(d for d in found if d not in record_dirs)
    elif args.match:
        record_dirs = [d for d in record_dirs
                       if re.search(args.match, d, re.I)]
    if not record_dirs:
        ap.error("no record dirs: pass --record_dir and/or --record_root "
                 "(and check --match)")
    names_for = dict(zip(record_dirs, run_names(record_dirs)))

    scorer = CachedScorer(get_scorer(args.bench)) if not args.use_stored_reward else None

    results = {}
    per_run = {}
    do_extended = args.extended or bool(args.plots_dir)
    if do_extended and HA is None:
        print("[warn] --extended requested but harvest_analysis.py not "
              "importable; continuing without it")
        do_extended = False

    ref_set = set()
    ref_info = None
    if args.ref_smiles and do_extended:
        with open(args.ref_smiles) as f:
            for line in f:
                s = line.strip().split()[0] if line.strip() else ""
                m = Chem.MolFromSmiles(s) if s else None
                if m is not None:
                    ref_set.add(Chem.MolToSmiles(m))
        print(f"[ref] {len(ref_set)} reference molecules from {args.ref_smiles}")

        # Score the reference set itself: the matched-budget "best of dataset"
        # virtual-screening baseline, the analogue of the 0.839 row in the
        # GuacaMol table. Beating it is what separates optimisation from
        # retrieval, so it belongs on every plot.
        # `list(ref_set)[:n]` is NOT reproducible: iteration order over a set of
        # strings depends on PYTHONHASHSEED, so the "best of dataset" baseline
        # silently changed between runs (0.794 to 0.799 on Osimertinib across
        # two seeds). Sort for a fixed order, then take a seeded random sample so
        # the subset is reproducible without being alphabetically biased.
        ref_sorted = sorted(ref_set)
        if len(ref_sorted) > args.ref_limit:
            ref_list = random.Random(args.ref_seed).sample(ref_sorted,
                                                           args.ref_limit)
        else:
            ref_list = ref_sorted
        ref_info = {"name": os.path.basename(args.ref_smiles), "n": len(ref_list)}
        if scorer is not None:
            rs = [scorer(s) for s in ref_list]
            ref_info["scores"] = rs
            for k in (1, 10, 100):
                ref_info[f"top{k}"] = HA.topk_mean_list(rs, k) if hasattr(
                    HA, "topk_mean_list") else float(np.mean(sorted(rs, reverse=True)[:k]))
            ref_info["guacamol_score"] = float(np.mean(
                [ref_info["top1"], ref_info["top10"], ref_info["top100"]]))
            best_ref = [s for _, s in sorted(zip(rs, ref_list), reverse=True)][:args.analysis_topn]
            ref_info["quality_pass_rate"] = HA.quality_report(best_ref).get("pass_rate")
            ref_info["descriptors"] = HA.descriptor_table(best_ref)
            print(f"[ref] best-of-dataset: top1={ref_info['top1']:.4f} "
                  f"top10={ref_info['top10']:.4f} top100={ref_info['top100']:.4f} "
                  f"composite={ref_info['guacamol_score']:.4f}")
        else:
            ref_info["descriptors"] = HA.descriptor_table(ref_list[:1000])

    for rd in record_dirs:
        name = names_for[rd]
        rows = load_records(rd)
        if not rows:
            print(f"[{name}] EMPTY -- skipping")
            continue
        total = len(rows)
        print(f"[{name}] {total} recorded molecules")
        if total < args.budget:
            print(f"[{name}] WARNING: fewer records ({total}) than budget "
                  f"({args.budget}). The budgeted numbers below are computed "
                  f"over {total} calls and are NOT comparable to a full "
                  f"{args.budget}-call run.")

        entry = {"record_dir": rd, "benchmark": args.bench,
                 "n_records_total": total}

        entry["budgeted"] = evaluate(
            rows, scorer, min(args.budget, total), topks=topks,
            auc_every=args.auc_every, use_stored=args.use_stored_reward)
        if not args.no_unbounded:
            entry["unbounded"] = evaluate(
                rows, scorer, None, topks=topks,
                use_stored=args.use_stored_reward)

        results[name] = entry

        b = entry["budgeted"]
        # GuacaMol's goal-directed MPO score is the mean of the top-1, top-10
        # and top-100 means (UniformSpecification(1,10,100)) -- NOT top-10
        # alone. This is the number that compares to the published table.
        if all(f"top{k}" in b for k in (1, 10, 100)):
            for variant in ("budgeted", "unbounded"):
                if variant in entry:
                    v = entry[variant]
                    v["guacamol_score"] = float(np.mean(
                        [v["top1"], v["top10"], v["top100"]]))
            print(f"[{name}] GuacaMol composite (budgeted) = "
                  f"{b['guacamol_score']:.4f}")

        print(f"[{name}] budgeted({b['n_oracle_calls']} calls): "
              + " ".join(f"top{k}={b[f'top{k}']:.4f}" for k in topks)
              + f" auc_top10={b.get('auc_top10', float('nan')):.4f}")
        print(f"[{name}]   3D->SMILES rate={b['smiles_conversion_rate']:.3f} "
              f"unique={b['n_unique_smiles']}")
        if "unbounded" in entry:
            u = entry["unbounded"]
            print(f"[{name}] unbounded({u['n_oracle_calls']} calls): "
                  + " ".join(f"top{k}={u[f'top{k}']:.4f}" for k in topks))

        # ---------------- extended analysis ----------------
        if do_extended and HA is not None:
            src = entry.get("unbounded", entry["budgeted"])
            stream = src["_stream"]
            top = [x["smiles"] for x in src["_best"][:args.analysis_topn]
                   if x.get("smiles")]

            ext = {}
            ext["curve_top10"] = HA.best_so_far_curve(
                stream, k=10, every=max(10, args.auc_every),
                budget=src["n_oracle_calls"])
            ext["buckets"] = HA.bucketize(rows, stream, n_buckets=args.buckets)
            ext["all_scores"] = [s for _, s in stream]
            ext.update(HA.first_hit(stream))
            ext.update(HA.yield_above(stream))
            ext["top100"] = src.get("top100")
            # distribution-learning metrics, computed over the whole stream
            # rather than the top-100: validity x uniqueness is the GuacaMol
            # distribution-benchmark convention and is what a collapsing policy
            # loses first.
            v = src.get("smiles_conversion_rate")
            u = src.get("uniqueness_among_valid")
            ext["dist_metrics"] = {
                "validity": v, "uniqueness": u,
                "valid_x_unique": (v * u) if (v and u) else float("nan")}
            if top:
                ext["quality"] = HA.quality_report(top)
                ext["descriptors"] = HA.descriptor_table(top)
                ext["diversity"] = {
                    "internal_diversity": HA.internal_diversity(top),
                    **HA.scaffold_stats(top)}
                comp = HA.component_breakdown(args.bench, top)
                if comp:
                    ext["components"] = comp
                if ref_set:
                    ext["diversity"]["novelty_vs_ref"] = HA.novelty(top, ref_set)
                    ext["diversity"]["nn_similarity_to_ref"] = \
                        HA.nearest_neighbour_similarity(top, ref_set)
                    ext["dist_metrics"]["novelty"] = \
                        HA.novelty([s for s in (idx_smi for idx_smi in
                                                src["_idx_to_smi"].values()) if s],
                                   ref_set)
            if scorer is not None:
                ext["stored_vs_rescored"] = HA.stored_vs_rescored(rows, scorer)

            entry["extended"] = ext
            per_run[name] = ext

            q = ext.get("quality", {}).get("pass_rate")
            d = ext.get("diversity", {}).get("internal_diversity")
            print(f"[{name}]   quality pass={q if q is None else round(q,3)} "
                  f"int_div={d if d is None else round(d,3)} "
                  f"n_scaffolds={ext.get('diversity',{}).get('n_scaffolds')}")
            hit = ext.get("calls_to_0.8")
            print(f"[{name}]   calls to first >=0.8: {hit}  "
                  f"yield>=0.8: {ext.get('yield_0.8')}")
            svr = ext.get("stored_vs_rescored") or {}
            if svr and not svr.get("agrees", True):
                print(f"[{name}]   WARNING: stored reward disagrees with "
                      f"rescoring (mean abs diff {svr['mean_abs_diff']:.3f}) "
                      f"-- training reward config may differ from --bench")

        if args.dump_best:
            os.makedirs(args.dump_best, exist_ok=True)
            src = entry.get("unbounded", entry["budgeted"])
            with open(os.path.join(args.dump_best, f"{name}_top100.csv"), "w",
                      newline="") as f:
                w = csv.writer(f)
                w.writerow(["rank", "oracle_call", "score", "smiles"])
                for rank, rec in enumerate(src["_best"], 1):
                    w.writerow([rank, rec["i"], f"{rec['score']:.6f}", rec["smiles"]])

    # strip the bulky internal lists out of the JSON summary
    slim = json.loads(json.dumps(results, default=str))
    for e in slim.values():
        for v in ("budgeted", "unbounded"):
            if v in e:
                for k in ("_best", "_stream", "_idx_to_smi"):
                    e[v].pop(k, None)
        if "extended" in e:
            e["extended"].pop("all_scores", None)

    if ref_info:
        slim["_reference"] = {k: v for k, v in ref_info.items() if k != "scores"}

    if args.plots_dir and per_run:
        files = HA.make_plots(per_run, args.plots_dir, bench=args.bench,
                              ref=ref_info)
        if files:
            print(f"wrote {len(files)} figure(s) to {args.plots_dir}")
            for p in files:
                print(f"  {os.path.basename(p)}")
        else:
            print("[warn] no figures written (matplotlib missing?)")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(slim, f, indent=2)
        print(f"wrote {args.out}")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        base_cols = ["run", "variant", "n_oracle_calls", "n_valid_smiles",
                     "n_unique_smiles", "smiles_conversion_rate",
                     "uniqueness_among_valid", "score_mean",
                     *[f"top{k}" for k in topks], "guacamol_score", "auc_top10"]
        ext_cols = ["valid_x_unique", "quality_pass_rate", "internal_diversity",
                    "n_scaffolds", "scaffold_diversity", "top_scaffold_frac",
                    "novelty_vs_ref", "nn_similarity_to_ref",
                    "calls_to_0.7", "calls_to_0.8", "calls_to_0.9",
                    "yield_0.5", "yield_0.7", "yield_0.8", "yield_0.9",
                    "mw_mean", "logp_mean", "qed_mean", "sa_mean",
                    "rings_mean", "arom_rings_mean", "rotb_mean",
                    "tpsa_mean", "heavy_mean"]
        # Only emit the extended columns when the extended analysis actually
        # ran -- otherwise every one of them is blank and looks like a bug.
        have_ext = any("extended" in e for e in slim.values())
        cols = base_cols + (ext_cols if have_ext else [])
        if not have_ext and (args.extended or args.plots_dir):
            print("[warn] extended metrics missing from the CSV -- the "
                  "analysis did not run for any record dir")
        elif not have_ext:
            print("[note] CSV has base columns only; pass --extended for "
                  "quality, diversity, novelty and descriptor columns")

        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for name, e in slim.items():
                ext = e.get("extended", {})
                div = ext.get("diversity", {})
                desc = ext.get("descriptors", {})
                flat = {
                    "quality_pass_rate": ext.get("quality", {}).get("pass_rate"),
                    "valid_x_unique": ext.get("dist_metrics", {}).get("valid_x_unique"),
                    "internal_diversity": div.get("internal_diversity"),
                    "n_scaffolds": div.get("n_scaffolds"),
                    "scaffold_diversity": div.get("scaffold_diversity"),
                    "top_scaffold_frac": div.get("top_scaffold_frac"),
                    "novelty_vs_ref": div.get("novelty_vs_ref"),
                    "nn_similarity_to_ref": div.get("nn_similarity_to_ref"),
                }
                for t in ("0.7", "0.8", "0.9"):
                    flat[f"calls_to_{t}"] = ext.get(f"calls_to_{t}")
                for t in ("0.5", "0.7", "0.8", "0.9"):
                    flat[f"yield_{t}"] = ext.get(f"yield_{t}")
                for k in ("mw", "logp", "qed", "sa", "rings", "arom_rings",
                          "rotb", "tpsa", "heavy"):
                    flat[f"{k}_mean"] = (desc.get(k) or {}).get("mean")
                for variant in ("budgeted", "unbounded"):
                    if variant in e:
                        w.writerow({"run": name, "variant": variant,
                                    **flat, **e[variant]})
            if ref_info and "top1" in ref_info:
                w.writerow({"run": f"[reference] {ref_info['name']}",
                            "variant": "best_of_dataset",
                            "n_oracle_calls": ref_info["n"],
                            **{f"top{k}": ref_info.get(f"top{k}") for k in topks},
                            "guacamol_score": ref_info.get("guacamol_score"),
                            "quality_pass_rate": ref_info.get("quality_pass_rate"),
                            **{f"{k}_mean": (ref_info.get("descriptors", {}).get(k) or {}).get("mean")
                               for k in ("mw", "logp", "qed", "sa", "rings",
                                         "arom_rings", "rotb", "tpsa", "heavy")}})
        print(f"wrote {args.csv}")

    if scorer is not None:
        print(f"[scorer] {scorer.n_calls} unique molecules scored, "
              f"{scorer.n_fail} scoring failures")


if __name__ == "__main__":
    main()
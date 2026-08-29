#!/usr/bin/env python3
"""
make_fig18_training_curves.py -- Figure 18: what training looks like over time,
DB against RTB.

Every other figure in this paper reads post-training artifacts (dumped samples,
flip-diagnostic probes). This one reads the wandb run logs directly, which
nothing else does, to show the training dynamics those artifacts are the
endpoint of: does the loss actually descend, does validity survive training,
and does the terminal reward move early or late in the run.

Panels, one guide (hidden, Osimertinib, beta=10, replay off, seed 0), DB vs RTB:
  A  training loss, each objective's own scale
  B  mean training-batch log-reward over training (train/log_reward_mean --
     logged every step; eval/log_reward_top10 is logged once per run in these
     particular runs and is not a curve, so it is not what this figure plots)
  C  fraction of the training batch that converts to a valid molecule
     (train/valid_frac), over training

INPUTS
  logs/wandb/run-*/files/config.yaml   to find the run directory for each config
  logs/wandb/run-*/run-*.wandb         the local wandb datastore (read directly;
                                        no network access, no wandb server needed)

Multiple run directories can share a config name (an aborted or smoke-test
attempt logged under the same --name before the real run). This script keeps
the run with the most history rows for each name, on the assumption that a
partial run is the shorter one.

USAGE
  python figures/make_fig18_training_curves.py --out out/fig18.pdf
  python figures/make_fig18_training_curves.py --names sweep-osim-hidden-db-replay_off-b10-s0,sweep-osim-hidden-rtb-replay_off-b10-s0
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt

import figstyle as fs

WANDB_ROOT = os.path.join(os.path.dirname(fs.__file__), "..", "logs", "wandb")


def _run_name(config_path):
    """The --name value a run was launched with, or None."""
    try:
        with open(config_path) as f:
            lines = f.readlines()
    except OSError:
        return None
    for i, line in enumerate(lines):
        if line.strip() == "- --name":
            for j in range(i + 1, min(i + 3, len(lines))):
                val = lines[j].strip()
                if val.startswith("- "):
                    return val[2:].strip().strip('"')
    return None


def _read_history(wandb_path):
    """Every history record in a local .wandb datastore, as a list of dicts."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    ds = DataStore()
    ds.open_for_scan(wandb_path)
    rows = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in rec.history.item:
            key = item.key if item.key else "/".join(item.nested_key)
            try:
                row[key] = json.loads(item.value_json)
            except (json.JSONDecodeError, TypeError):
                row[key] = item.value_json
        rows.append(row)
    return rows


def find_run(name):
    """The run directory logged under --name `name`, preferring the longest history."""
    best_rows, best_dir = None, None
    for config_path in glob.glob(os.path.join(WANDB_ROOT, "run-*", "files", "config.yaml")):
        if _run_name(config_path) != name:
            continue
        run_dir = os.path.dirname(os.path.dirname(config_path))
        wandb_files = glob.glob(os.path.join(run_dir, "run-*.wandb"))
        if not wandb_files:
            continue
        rows = _read_history(wandb_files[0])
        if best_rows is None or len(rows) > len(best_rows):
            best_rows, best_dir = rows, run_dir
    if best_rows is None:
        print(f"[fig18] no wandb run found for --name {name}")
    return best_rows


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        if key in r and "_step" in r:
            xs.append(r["_step"])
            ys.append(r[key])
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=(
        "sweep-osim-hidden-db-replay_off-b10-s0,"
        "sweep-osim-hidden-rtb-replay_off-b10-s0"))
    fs.add_arg_common(ap, "out/fig18_training_curves.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    runs = {n: find_run(n) for n in names}
    runs = {n: r for n, r in runs.items() if r}
    if not runs:
        print("[fig18] no runs found, nothing to plot")
        return

    colours = {names[0]: "#4C72B0", names[-1]: "#DD8452"}
    labels = {n: ("DB" if "-db-" in n else "RTB" if "-rtb-" in n else n) for n in names}

    fig, axes = plt.subplots(1, 3, figsize=(args.width, 2.8))

    ax = axes[0]
    for n, rows in runs.items():
        loss_key = "db/terminal_loss" if "-db-" in n else "rtb/loss"
        xs, ys = series(rows, loss_key)
        if not xs:
            # fall back to whatever loss key this run actually logged
            candidates = {k for r in rows for k in r if k.endswith("loss")}
            for k in candidates:
                xs, ys = series(rows, k)
                if xs:
                    loss_key = k
                    break
        ax.plot(xs, ys, color=colours.get(n), label=f"{labels[n]} ({loss_key.split('/')[-1]})")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.legend()

    ax = axes[1]
    for n, rows in runs.items():
        xs, ys = series(rows, "train/log_reward_mean")
        ax.plot(xs, ys, color=colours.get(n), label=labels[n])
    ax.set_xlabel("training step")
    ax.set_ylabel("mean training-batch log-reward")
    ax.set_title("Training-batch reward")
    ax.legend()

    ax = axes[2]
    for n, rows in runs.items():
        xs, ys = series(rows, "train/valid_frac")
        ax.plot(xs, ys, color=colours.get(n), label=labels[n])
    ax.set_xlabel("training step")
    ax.set_ylabel("fraction of batch valid")
    ax.set_title("Training-batch validity")
    ax.legend()

    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()

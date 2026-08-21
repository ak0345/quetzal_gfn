# scripts/

Every experiment driver. Each one is a thin wrapper that builds command lines for
the Python entry points at the repo root, so nothing here contains science: if a
number looks wrong, the script tells you which `python ...` produced it.

Two rules hold everywhere:

- **`DRY=1` prints every command and runs nothing.** Use it before anything long.
- **Everything is resumable.** A run whose checkpoint, `dump_summary.json` or
  `_state/<name>.done` marker exists is skipped. Re-invoking a stage costs only
  what is left to do, which is what makes the retry loop cheap.

---

## Quick start

```bash
DRY=1 bash scripts/run_study.sh          # see what would run
nohup bash scripts/run_study.sh > study.log 2>&1 &
```

Before committing days of GPU, measure your throughput on one epoch:

```bash
time MAX_EPOCHS=1 STEPS=100 SEEDS=0 REWARDS=osim GUIDES=hidden \
  OBJECTIVES=db REPLAYS=off BETAS=10 bash scripts/01_train_guides.sh
```

That is exactly 12,800 molecules. Divide the wall-clock by 12,800 to get your
seconds per molecule, then multiply by the molecule count of the grid you intend
to run. Each molecule is one autoregressive rollout plus one CPU reward call.

---

## The stage model

| Stage | Script | Trains | Needs |
|---|---|---|---|
| 1 | `01_train_guides.sh` | guide sweep on the frozen prior | the prior |
| 2 | `02_train_components.sh` | per-component guides, stability control | the prior |
| 3 | `03_finetune.sh` | the prior's own weights (RTB) | the prior |
| 4 | `04_dump_guides.sh` | nothing, samples and scores | stage 1 checkpoints |
| 5 | `05_dump_composed.sh` | nothing, composes and scores | stage 2 checkpoints |
| 6 | `06_flip_diagnostics.sh` | nothing, coupled-flip probe | stage 1 checkpoints |
| 7 | `07_ablations.sh` | nothing, mechanism probes | stage 1 or 2 checkpoints |
| 8 | `08_analysis.sh` | nothing, harvest and baselines | stage 3 molecule streams |

Dependencies that actually bite:

```
1 ──> 4 (dumps)  ──> figures 1, 11-14
  └─> 6 (flips)  ──> figures 2, 6
  └─> 7 (ablations) ──> figures 5, 7
2 ──> 5 (composed) ──> figure 9
3 ──> 8 (harvest) ──> figures 3, 13, 14 and the PMO table
```

Stage 8's harvest reads the molecule streams stage 3 records, so **8 must follow
3**. Everything else can run in any order once its inputs exist.

---

## The study runners

`run_study.sh` and its variants set a grid, then call the stages in order with a
retry loop. The grid is sliced across several files so the pieces can run on
different machines or at different times.

| Runner | Guide rewards | Guides | Fine-tune rewards | Stages |
|---|---|---|---|---|
| `run_study.sh` | osim, peri | hidden, base | osim, peri | 1 4 6 7 3 8 |
| `run_study2.sh` | nitrogen, fexo | hidden, base | fexo | 1 4 6 7 3 8 |
| `run_study-tempgain.sh` | peri, osim, fexo, nitrogen | tempgain | (unused) | 1 4 6 7 8 |
| `03_finetune-zaleplon.sh` | — | — | zaleplon, seed 42 | stage 3 only |

All share: 2 seeds (0, 42), betas {1, 10, 100}, objectives {db, rtb}, replay
{on, off}, batch 128, and a 10-minute hang guard.

### Order and concurrency

```
1 guides    GUIDE_PARALLEL concurrent   (3)
4 dumps     DUMP_PARALLEL concurrent    (1)
6 flips     serial, cheap
7 ablations serial, cheap
3 fine-tune ONE AT A TIME, last of the training stages
8 harvest   serial, must follow 3
```

Guides parallelise because reward evaluation is CPU-serial while generation is on
the GPU, so concurrent processes overlap one run's scoring with another's
sampling. **Fine-tuning is serial by construction**, not by setting: stage 3's
`run()` executes each command synchronously with no `throttle` call, unlike
stages 1 and 4. Raising `MAX_PARALLEL` there does nothing.

### Retries

Each stage is re-invoked up to `MAX_RETRIES` (3) times with `BACKOFF` (30s)
between attempts. Because stages are resumable, a retry re-runs only what did not
finish. This is what recovers a stalled run: the run exits 17, the stage
completes with that one missing, and the next attempt picks it up.

---

## Per-script reference

### `common.sh`
Sourced by every stage. Sets `REPO_ROOT`, puts it on `PYTHONPATH` (needed for
anything under `ablations/` to import `gflow`, `reward_fn`, ...), and `cd`s
there, so stages work from any directory.

| Variable | Default | Meaning |
|---|---|---|
| `QUETZAL_CKPT` | `checkpoints/geom.ckpt` | the frozen prior |
| `CKPT_ROOT` | `logs/quetzal-gfn` | where run directories live |
| `REF_SMILES` | `reference/geom_drugs_smiles.txt` | GEOM corpus for FCD, novelty, the dataset baseline |
| `RESULTS_ROOT` | `results` | where artifacts are written |
| `LOG_ROOT` | `logs/drivers` | driver logs |
| `MAX_PARALLEL` | 1 | concurrent jobs, honoured by stages 1, 2, 4, 5 |
| `NUM_GPUS` | 1 | round-robin `CUDA_VISIBLE_DEVICES` |
| `GUARD_STALL_MINUTES` | 10 | watchdog on batch progress |
| `GUARD_REWARD_TIMEOUT` | 20 | per-molecule reward ceiling, seconds |
| `DRY` | 0 | 1 prints commands only |

If you downloaded weights from Hugging Face, check the layout. The recorded paths
in the flip reports read `logs/quetzal-gfn/legit/<run>/checkpoints/`, so you may
need `CKPT_ROOT=logs/quetzal-gfn/legit`.

### `01_train_guides.sh` — the guide sweep
Trains a small guide on the frozen prior across guide × objective × replay × beta
× reward × seed. Training only: `--eval_n 0 --hist_every_n_epochs 0
--no_fcd_enabled`, because scoring inside the loop is CPU work that stalls the
GPU. Scoring happens in stage 4.

Knobs: `SUBSET` (1 for the study grid, 0 for the wider matrix including tempgain
and the KL objectives), `GUIDES`, `OBJECTIVES`, `REPLAYS`, `BETAS`, `REWARDS`,
`SEEDS`, `MAX_EPOCHS`, `STEPS`, `BSZ`.

Rewards map to flags as: `osim`/`peri`/`zaleplon`/`fexo` go through the GuacaMol
passthrough, `nitrogen` uses the custom `--reward nitrogen_count`
(`_score_nitrogen_fraction` in `reward_fn.py`).

Run names are `sweep-<reward>-<guide>-<objective>-replay_<on|off>-b<beta>` with a
`-s<seed>` suffix when `SEEDS` is set.

### `02_train_components.sh` — component guides and the stability control
One guide per leaf scorer of an MPO objective (the teachers stage 5 composes),
plus guides trained against EDM atom stability. Not in any study runner: nothing
downstream in the current paper needs it except figure 9.

### `03_finetune.sh` — fine-tuning the prior
RTB fine-tuning at three scopes plus LoRA. Every configuration is sized to
collect about 10,000 molecules, which is the budget stage 8 harvests at:

| Scope | Batch | Steps | Molecules |
|---|---|---|---|
| proj, atom, and all LoRA ranks | 64 | 160 | 10,240 |
| full | 12 | 840 | 10,080 |

A nitrogen sanity run goes first and gates the rest. It targets a dense,
atom-decomposable reward, so if it does not move, the loop is broken and nothing
below it can be interpreted. `SKIP_SANITY=1` once it has passed.

Knobs: `REWARDS`, `SEEDS`, `LORA_RANKS`, `BETA`, `BETA_START`, `BSZ`,
`BSZ_FULL`, `ONLY` (one run by name), `DEVICE`.

### `04_dump_guides.sh` — score the guides
Samples `N` molecules per (checkpoint, seed) and computes the metric suite, then
aggregates into `master_table.csv`. Two phases: the frozen prior is dumped once
per reward family and every guide on that reward reuses it via `--base_from`,
which is what makes the sweep affordable.

Note `SEEDS` here means **dump seeds**, which resample molecules from one trained
checkpoint. That is sampling variance, not training variance. Training seeds are
the `-s<N>` suffix from stage 1.

`DIFF_STEPS` stays at 18 to match training. Lowering it changes the geometry and
therefore every bond-perception-derived number.

### `05_dump_composed.sh` — the composition track
Mixes stage 2's component guides under linear, product and harmonic operators.
Use `--route policy`: an earlier flow-route set computed the residual and never
applied it, giving a residual norm of exactly 0.000 at every state.

### `06_flip_diagnostics.sh` — the coupled-flip probe
Rolls trajectories with the frozen prior and compares its next-atom distribution
against the guided one on the identical state under a shared uniform draw. Cheap:
no molecules are scored. Reports delivery, the sampled-flip rate, the prior's
top-1 margin and the flip rate by sequence position.

Delivery is measured separately from the flip rate on purpose. Delivery near 1
with a flip rate near 0 is a confident prior; delivery near 0 is a wiring failure.

### `07_ablations.sh` — mechanism probes
Sections, passed as `$1`: `ceiling` (margin binning, figure 5), `guide`
(residual-scale sweep, figure 7), `flip`, `singles` (figure 9), `tempgain`
(figure 8), `rollout`, `logs`, `single`.

Three ways to point it at checkpoints:

```bash
RUNS="run-a run-b" bash scripts/07_ablations.sh ceiling      # names under CKPT_ROOT
GUIDE_CKPTS=/a.ckpt,/b.ckpt GUIDE_LABELS=a,b bash scripts/07_ablations.sh guide
COMPONENTS="0 2 3" bash scripts/07_ablations.sh singles      # stage 2 guides
```

`singles` genuinely needs component guides, since "each component alone" and
"composition weights" are only defined over a set of teachers. `tempgain` needs
guides that have temperature heads. The study runners pass `RUNS` built from the
grid, because the un-seeded defaults will not match `-s<N>` names.

### `08_analysis.sh` — harvest, baselines, histograms
Sections: `harvest` (score fine-tuning streams at a budget), `baseline`
(best-of-N curves including the matched-budget dataset baseline), `hists`
(per-objective and per-component reward histograms).

Two conventions are emitted because they are not comparable: *unbounded* top-k
over the whole stream (GuacaMol leaderboard parity) and *budgeted* top-k plus
AUC-top-10 over `BUDGET` calls (PMO convention). Since top-k is non-decreasing,
AUC ≤ final.

### `supervise_run.sh` — restart one long job
```bash
bash scripts/supervise_run.sh NAME -- python rtb_finetune.py --name NAME ...
```
Stops on exit 0, retries on 17 (stall) and on crashes, gives up after three
failures inside `MIN_RUNTIME` because a run that dies in seconds is a config
error rather than a transient stall. Mostly superseded by the retry loop in the
study runners, but useful for a single run.

### `run_all.sh` — all eight stages, unsupervised
The original driver. Prefer a study runner.

### `prior/` — upstream Quetzal
SLURM helpers for pretraining the prior itself. Only needed to substitute a
different prior.

---

## The hang guard

Reward evaluation runs on the CPU, one molecule at a time. RDKit's
`rdDetermineBonds` infers bonds from 3D coordinates by searching bond orders and
charges, and on a dense structure that search can effectively never return. The
process then stays alive holding its GPU while producing nothing: no exception,
no OOM, just a run that has stopped.

Two layers, both on by default:

| Flag | Scope | Default | On trigger |
|---|---|---|---|
| `--guard_reward_timeout` | one molecule | 20 s | score at the invalid floor, keep training |
| `--guard_stall_minutes` | the training loop | 10 min | dump stacks, flush the molecule log, exit 17 |

Plus `kill -USR1 <pid>` on any training process to print every thread's stack
without killing it. Dumps land in `logs/quetzal-{gfn,ft}/<name>/`.

**Watch `train/reward_timeouts`.** A timed-out molecule is floored, so a guard
that fires often quietly reshapes the reward distribution. If that count is not
roughly zero, raise the timeout rather than ignore it.

---

## Recipes

```bash
# smoke test the whole pipeline in minutes
SUBSET=1 REWARDS=nitrogen MAX_EPOCHS=1 SEEDS=0 N=200 bash scripts/run_study.sh

# re-score without retraining
STAGES="4 8" bash scripts/run_study.sh

# only the fine-tunes
STAGES=3 bash scripts/run_study.sh

# one run by name
ONLY=rtb-proj-osim-b10-s0 bash scripts/03_finetune.sh

# ease off if VRAM is tight
GUIDE_PARALLEL=2 bash scripts/run_study.sh

# force a stage to redo one run
rm results/oracle_gfn_mols/_state/rtb-proj-osim-b10-s0.done
rm -r logs/quetzal-gfn/sweep-osim-hidden-db-replay_off-b10-s0

# then the figures
bash figures/make_all.sh
```

---

## Two things to check in the current runners

**Step count.** The runners export `MAX_EPOCHS=5` while the inline comment says
600 optimiser steps. At `STEPS=100` that is 500 steps, not 600. Stage 1's own
default is 6, so running it standalone gives 600 and running it through a study
runner gives 500. Set `MAX_EPOCHS=6` in the runner if 600 was intended.

**Zaleplon guides.** No runner has zaleplon in its guide `REWARDS`: they are
`{osim, peri}`, `{nitrogen, fexo}` and `{peri, osim, fexo, nitrogen}`. Zaleplon
appears only in `03_finetune-zaleplon.sh`, which is stage 3 only. If you want
zaleplon guide runs, add it to a runner's `REWARDS`.

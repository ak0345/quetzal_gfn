# Steerability Limits in Frozen 3D Molecular Priors

Code and experiments for **A Characterization of Steerability Limits in Frozen 3D
Molecular Priors** ([paper](paper/steerability-limits-frozen-3d-priors.pdf)).

We steer [Quetzal](https://arxiv.org/abs/2505.13791), an autoregressive 3D
molecular model trained on GEOM-Drugs, toward GuacaMol MPO objectives across many
configurations spanning three guide architectures, five training objectives and
four fine-tuning scopes.

---

## Contents

- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Downloads](#downloads)
- [Running the experiments](#running-the-experiments)
- [Configuration reference](#configuration-reference)
- [Figures](#figures)
- [Pretraining the prior](#pretraining-the-prior)
- [Citation](#citation)

---

## Repository layout

Python modules live flat at the repository root, because they import each other
by bare module name (`from chem import Molecule`). Run them from the root, or
let `scripts/common.sh` set `PYTHONPATH` for you.

```
├── paper/                    the paper this code accompanies
├── scripts/                  experiment drivers, one per stage — start here
│   ├── README.md             what each stage does and how to run them
│   ├── common.sh             shared paths, throttling, checkpoint resolution
│   ├── 01_train_guides.sh    the guide sweep
│   ├── 02_train_components.sh per-component guides + the stability control
│   ├── 03_finetune.sh        RTB fine-tuning of the prior's own weights
│   ├── 04_dump_guides.sh     sample and score every guide checkpoint
│   ├── 05_dump_composed.sh   the composition track
│   ├── 06_flip_diagnostics.sh the coupled-flip diagnostic
│   ├── 07_ablations.sh       mechanism ablations
│   ├── 08_analysis.sh        harvest, best-of-N baselines, reward histograms
│   ├── run_all.sh            all eight stages in order
│   ├── run_study.sh          the supervised study: guides x3, fine-tunes last
│   ├── supervise_run.sh      restart a job that stalls or crashes
│   └── prior/                SLURM helpers for pretraining Quetzal itself
├── ablations/                the mechanism probes stage 7 drives
├── figures/                  one script per paper figure, + make_all.sh
├── notebooks/                play.ipynb, colab.ipynb
├── reference/                GEOM-Drugs SMILES corpus, frozen-prior samples
├── results/                  generated artifacts (see Downloads)
│
├── Steering                  gflow.py, gflow_multi.py, rtb_finetune.py
│                             hidden_guide.py, tempgain_guide.py, replay_buffer.py
├── Robustness                hang_guard.py, scripts/supervise_run.sh
├── Rewards                   reward_fn.py
├── Scoring                   final_dump.py, final_dump_composed.py,
│                             aggregate_dumps.py, harvest_eval.py,
│                             harvest_analysis.py, best_of_n_curve.py,
│                             smiles_hist.py, edm_metrics.py, metrics.py
└── The prior (upstream)      model.py, train.py, attention.py, simple_mlp.py,
                              chem.py, datasets.py, pack.py, qm9.py, geom.py,
                              data_smiles.py, generate.py, density.py, hdeco.py,
                              draw.py
```

### The three entry points

| Script | Trains | Intervenes on | Ratio |
|---|---|---|---|
| `gflow.py` | a small guide network | `p_atom` logits, prior frozen | exact |
| `gflow_multi.py` | nothing (inference only) | composes trained guides | — |
| `rtb_finetune.py` | the prior's own weights | `proj_logits` / trunk / everything | exact under `proj`, approximate otherwise |

---

## Setup

```bash
mamba env create -f environment.yml
mamba activate quetzal
```

**RDKit is pinned at 2023.03.3 and this matters.** Later versions change bond
perception, and therefore every validity, stability and 3D-to-SMILES conversion
figure reported here. On the QM9 100k training set, 2023.03.3 gives 99.99%
validity where later versions give 94.78%.

Optional Weights & Biases setup — training runs log there by default:

```bash
export WANDB_ENTITY=<your_entity>
```

The environment also needs `guacamol==0.5.5` (the MPO objectives) and
`fcd==1.2.2` (the distributional distance); both are in `environment.yml`.

---

## Downloads

### The frozen prior

Every experiment in this repository trains on top of the GEOM checkpoint from
the original Quetzal release. The scripts expect it at `checkpoints/geom.ckpt`.

```bash
mkdir -p checkpoints
wget -O checkpoints/geom.ckpt \
  https://huggingface.co/auhcheng/quetzal/resolve/main/geom.ckpt
```

### The reference corpus

`reference/geom_drugs_smiles.txt` (292k SMILES) ships with the repository and is
used for three things: the FCD and descriptor comparisons, novelty and
nearest-neighbour similarity, and the matched-budget dataset baseline itself.
To regenerate it from the raw GEOM-Drugs release:

```bash
python data_smiles.py
```

---

## Running the experiments

Every stage is independently resumable — a run whose checkpoint or summary
already exists is skipped — so the pipeline can be interrupted and restarted.
Every stage takes `DRY=1` to print its commands without executing them.

```bash
DRY=1 bash scripts/run_all.sh        # see what would run
bash scripts/run_all.sh              # stages 1-8
STAGES="4 8" bash scripts/run_all.sh # just these
```

Smoke-test first. The full sequence is on the order of days on one A100:

```bash
SUBSET=1 REWARDS=nitrogen MAX_EPOCHS=1 N=200 SEEDS=0 bash scripts/run_all.sh
```

### The stages

**1 — Guide sweep.** `scripts/01_train_guides.sh`
Trains a guide on the frozen prior across guide architecture × objective ×
replay × β × reward. Training only; scoring happens in stage 4, so a sweep runs
to completion on one GPU and is measured afterwards. `SUBSET=1` (the default)
runs the reduced grid; `SUBSET=0` runs the full 288-run matrix.

**2 — Component guides and controls.** `scripts/02_train_components.sh`
One guide per leaf scorer of an assembled MPO objective (the teachers stage 5
composes), plus guides trained against EDM atom stability.

**3 — Fine-tuning.** `scripts/03_finetune.sh`
RTB fine-tuning of the prior's own weights at four scopes, reproducing all 21
runs of Table 8. Begins with a sanity run on the dense nitrogen reward that
gates the rest: if that does not move, the loop is broken and nothing below is
interpretable. Molecules stream to `results/oracle_gfn_mols/<name>/molecules.jsonl`
during training.

**4 — Score the guides.** `scripts/04_dump_guides.sh`
Samples N molecules per (checkpoint, seed) and computes the metric suite —
reward histogram against the prior, FCD and descriptor distances to GEOM-Drugs,
EDM atom/mol stability — then aggregates into `master_table.csv` with seed error
bars. The frozen prior's samples are dumped once per reward family and reused by
every guide on that reward, which is what makes the sweep affordable.

**5 — Composition.** `scripts/05_dump_composed.sh`
Mixes the component guides under linear, product and harmonic operators and
scores the result on the assembled objective the components never saw.

**6 — Flip diagnostics.** `scripts/06_flip_diagnostics.sh`
The coupled-flip measurement, over every checkpoint. Trajectories are rolled by
the frozen prior; at each state the prior's next-atom distribution is compared
with the guided one on the identical state, using a shared uniform draw so the
two samplers are coupled. Much cheaper than a dump — no molecules are scored.

**7 — Mechanism ablations.** `scripts/07_ablations.sh`
Margin binning, the residual-scale sweep, per-component effect sizes and weight
skew, what the temperature and gain heads learned, rollout instrumentation, and
training curves from W&B. Sections run individually:
`bash scripts/07_ablations.sh ceiling`.

**8 — Analysis.** `scripts/08_analysis.sh`
Scores fine-tuning runs as goal-directed benchmarks at a fixed oracle budget,
computes the best-of-N curves including the dataset baseline, and plots
per-objective and per-component reward histograms. No GPU training.

### Common overrides

| Variable | Default | Meaning |
|---|---|---|
| `QUETZAL_CKPT` | `checkpoints/geom.ckpt` | the frozen prior |
| `CKPT_ROOT` | `logs/quetzal-gfn` | where run directories live |
| `REF_SMILES` | `reference/geom_drugs_smiles.txt` | reference corpus |
| `RESULTS_ROOT` | `results` | where artifacts are written |
| `MAX_PARALLEL` | `1` | concurrent jobs on the shared GPU |
| `NUM_GPUS` | `1` | for round-robin device pinning |
| `DRY` | `0` | `1` prints commands without running them |

Each concurrent training holds its own copy of the frozen prior in VRAM, so
raising `MAX_PARALLEL` past what the GPU fits will OOM rather than run faster.

### Hang guard

Training can stop progressing while the process stays alive and holds its GPU:
no exception, no OOM, no collapse. It is nearly always a CPU-side reward call
that never returns — RDKit bond perception from 3D coordinates searches over
bond orders and charges and can blow up on a dense structure, and an xtb
subprocess has no timeout of its own.

`hang_guard.py` is wired into `gflow.py` and `rtb_finetune.py` and is **on by
default**, in three layers:

| Layer | What it does | Flag |
|---|---|---|
| `install_faulthandler` | `kill -USR1 <pid>` prints every thread's stack without killing the run | always on |
| `guarded_reward` | hard per-molecule ceiling on reward evaluation; an overrun is scored at the invalid floor | `--guard_reward_timeout` (default 20s, 0 disables) |
| `StallGuard` | no batch in N minutes → dump stacks, flush the molecule log, exit 17 | `--guard_stall_minutes` (default 30, 0 disables) |

Stack dumps go to `logs/quetzal-{gfn,ft}/<name>/`. To diagnose a live hang
without killing it:

```bash
kill -USR1 $(pgrep -f "gflow.py --name <run>")
```

`scripts/supervise_run.sh` restarts a job that stalls or crashes:

```bash
bash scripts/supervise_run.sh rtb-proj-osim-b10 --   python rtb_finetune.py --name rtb-proj-osim-b10 --finetune_scope proj     --reward guacamol --reward_smiles hard_osimertinib --reward_beta 10
```

It stops on exit 0, retries on 17 (stall) and on crashes, and gives up after
three failures inside `MIN_RUNTIME` — a run that dies in seconds is a config
error, not a transient stall, and retrying it twenty times just fills the disk
with identical tracebacks. It also wraps a whole stage:
`bash scripts/supervise_run.sh guides -- bash scripts/01_train_guides.sh`.

Restarting is safe because both trainers resume from the newest checkpoint
automatically, and `rtb_finetune` rolls the molecule log back to the count
stored in the checkpoint (`recorder.rollback_to`). Lightning replays the batches
between the last checkpoint and the crash; without that rollback those molecules
would be logged twice under different indices and a budget slice on `i` would
spend oracle calls on duplicates. `gflow.py` records no molecule stream, so
restart is unconditionally safe there.

**What the guard costs.** A timed-out molecule is floored at `invalid_logr`,
joining the same bucket as one that fails bond perception — which is what it is.
But a guard firing often reshapes the reward distribution, so the count is
logged as `train/reward_timeouts` and printed periodically. **If that is not
approximately zero, the run is affected and the timeout needs raising, not
ignoring.**

`--guard_max_atoms` floors molecules above a size without scoring them. It is
off by default and should stay off: GEOM-sized molecules run to 80–100 heavy
atoms against a `max_len` of 192, so a ceiling near the sampled range trains the
policy away from large molecules, changing the objective rather than guarding
it. Reach for it only if a stack dump has shown large molecules to be the cause.

`final_dump.py` is **not** guarded — a hang during stage 4 will still need
`kill -USR1` and a manual restart.

### Seeds

There are **two independent seed axes**, and they measure different things.

| | Flag | Varies | Set by |
|---|---|---|---|
| **Training seed** | `gflow.py --seed`, `rtb_finetune.py --seed` | guide/adapter initialisation, rollout sampling, the replay buffer | `SEEDS` in stages 1 and 3 |
| **Dump seed** | `final_dump.py --seed` | which molecules are sampled from an *already-trained* checkpoint | `SEEDS` in stage 4 |

Training seeds are opt-in. Left unset, stages 1 and 3 run one job per
configuration with names exactly as before. Set them and each configuration is
trained once per seed, with `--seed` passed and a `-s<N>` suffix appended to the
run name so runs neither collide nor resume into each other:

```bash
SEEDS="0 42 100" bash scripts/01_train_guides.sh
SEEDS="0 42 100" bash scripts/03_finetune.sh
```

The aggregators read that suffix back as a `train_seed` column and also emit
`base_name` (the name with the suffix stripped), so `master_table.csv` gives one
row per (configuration, training seed) — mean and standard deviation over that
run's dump seeds — and you can group by `base_name` to pool across training
seeds. The suffix is optional in the name regexes, so runs recorded before seeds
existed still parse, with `train_seed` empty.

---

## Configuration reference

### Guide architectures (`gflow.py`)

Writing `h` for the trunk's hidden state and `W_proj` for the frozen atom-type
head. All are zero-initialised in their final layer, so the guided logits equal
the prior's exactly at initialisation.

| Name | Form | Flags |
|---|---|---|
| `hidden` | `W_proj(h + δ(h))` | *(default)* |
| `base` | `W_proj·h + g(h)` | `--no_use_hidden_guide` |
| `tempgain` | `W_proj·h / T(h) + γ(h)·g(h)` | `--no_use_hidden_guide --use_prior_temp --use_residual_gain` |

The hidden guide exists because `W_proj` amplifies displacements along the
directions it uses: a δ of norm 0.1 produces a logit change of ≈46, against ≈1.3
for an output residual of the same norm.

### Objectives (`--objective`)

`db` (detailed balance), `rtb` (relative trajectory balance), `revkl`, `fwdkl`.
The KL branches exist to separate the loss from the guidance mechanism: they use
an identical residual, trained instead by a direct KL to the tilted target.

### Rewards (`--reward`)

| Flag | What it scores |
|---|---|
| `--reward guacamol --reward_smiles <fn>` | an assembled GuacaMol MPO objective |
| `--reward guacamol_component --reward_benchmark <b> --reward_component <i>` | one leaf scorer of that objective |
| `--reward nitrogen_count` | fraction of heavy atoms that are nitrogen |
| `--reward atom_stability` | EDM atom stability |

`hard_osimertinib`, `hard_fexofenadine` and `perindopril_rings` are the
benchmark function names used here. Invalid molecules return a fixed floor of
−5 in log space, and the fraction of samples above that floor is reported
alongside every result: a flat mean log-reward is otherwise ambiguous between a
policy that is not steering and one whose samples are mostly invalid.

The nitrogen reward is a positive control, not a design objective. It is dense
(over 90% of samples score above the floor, against 2–6% for the assembled MPO
objectives), monotone in a single atom-level decision, and decomposes over
exactly the decisions a guide controls. It raises the top-10 nitrogen fraction
from 0.452 to 0.983 (+0.791 ± 0.099), two to three orders of magnitude larger
than the largest effect on either MPO benchmark — so the null results are not an
implementation failure.

### Fine-tuning scopes (`--finetune_scope`)

| Scope | Updates | Params | Ratio |
|---|---|---|---|
| `proj` | `W_proj` only | 98,304 | **exact** |
| `proj` + `--lora_rank r` | rank-`r` adapter on `W_proj` | `r(d+\|V\|)` | **exact** |
| `atom` | `W_proj` + the `encode1` trunk | ≈43M | approximate |
| `full` | everything, incl. the coordinate denoiser | ≈85M | approximate |

Only `proj` preserves the exact cancellation of the coordinate term: with
`enc1` and `enc2` frozen, `z_t` is the same function evaluated on the same
argument under policy and prior, so every term of the coordinate log-ratio is
identically zero. Under `atom` and `full` the trunk drifts and the atom-only
ratio incurs a bias; `diag/zprefix_drift` logs its size so the violation is
measured rather than assumed.

---

## Figures

Every figure in the paper has a script in [`figures/`](figures/). Each reads a
**committed artifact** under `results/` rather than re-running a model, so
figures regenerate on a laptop in seconds with no GPU and no checkpoints.

```bash
bash figures/make_all.sh                  # -> figures/out/
OUT_DIR=paper/figs FORMAT=png bash figures/make_all.sh
python figures/make_fig01_landscape.py --bench osim --out /tmp/f1.pdf
```

A figure whose input has not been produced yet prints the command that produces
it and is counted as skipped rather than failed, so a partial run shows exactly
what is outstanding. As committed, 8 of 13 render immediately; Figures 5, 7, 8
and 9 need stage 7, which has not been run in this layout.

[figures/README.md](figures/README.md) has the figure-to-artifact map and
records three known gaps between what the scripts produce and the paper as
written — most importantly that Figure 2's 0.89 first-atom flip rate does not
reproduce from the committed flip reports, where the maximum is 0.452.

## Pretraining the prior

The Quetzal model itself is upstream work
([paper](https://arxiv.org/abs/2505.13791),
[repo](https://github.com/aspuru-guzik-group/quetzal)); this project uses the
released GEOM checkpoint and freezes it. Reproduce the pretraining only if you
want to substitute a different prior — the most direct test of the paper's
account.

```bash
python qm9.py          # < 1 minute
python geom.py         # 30-60 minutes, ~100G

python train.py --name=qm9_run
sbatch scripts/prior/train_geom_slurm.sh

python generate.py --ckpt=<ckpt> --name=geom_samples --device=cuda \
  --num_samples=10000 --num_chunks=10 --diff_steps=120 --max_len=192
python metrics.py --samples_dir=samples/gen/geom_samples --dataset=geom
```

To submit many jobs, list commands in `scripts/prior/jobs` and run
`scripts/prior/submit.sh`. `hdeco.py` (hydrogen decoration),
`scripts/prior/add_hydrogens_obabel.sh` (OpenBabel + Hydride),
`scripts/prior/run_olex2.scpt` and `density.py` (exact log-likelihood) are also
upstream utilities, unchanged.

The architecture: a decoder-only transformer, 12 layers split into two stacks of
6, hidden width 768, 12 heads, block size 512. `enc1` embeds atom types,
coordinates and a Fourier featurisation of the coordinates and produces `h`, from
which `W_proj ∈ R^(128×768)` (no bias) emits atom-type logits. `enc2` re-embeds
the sampled atom type and produces the conditioner for the coordinate model — an
adaptive-layernorm MLP of width 1536 and depth 6, trained as an EDM-style
denoiser and sampled with 18 Heun steps. Atom types are atomic numbers over a
vocabulary of 128, with `STOP = 0`, `PAD = 126`, `GEN = 127`. We use the
exponential-moving-average weights throughout.

---

## Citation

Please cite the Quetzal paper for the prior:

```bibtex
@article{cheng2025scalable,
  title={Scalable Autoregressive {3D} Molecule Generation},
  author={Cheng, Austin H and Sun, Chong and Aspuru-Guzik, Al{\'a}n},
  journal={arXiv preprint arXiv:2505.13791},
  year={2025}
}
```

This work builds on GuacaMol (Brown et al., 2019) for the objectives, PMO
(Gao et al., 2022) for the budgeted metric, relative trajectory balance
(Venkatraman et al., 2024), GFlowNets (Bengio et al., 2021), LoRA (Hu et al.,
2021), residual RL (Johannink et al., 2018), GEOM-Drugs (Axelrod and
Gomez-Bombarelli, 2020) and EDM (Hoogeboom et al., 2022) for the stability
metrics. Full references are in the paper.

## License

See [LICENSE](LICENSE). The upstream Quetzal code retains its original license.

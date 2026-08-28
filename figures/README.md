# Paper figures

One script per figure. Every script reads an artifact under `results/` — a CSV or
JSON the pipeline already wrote — so figures regenerate in seconds with no GPU. A
script whose input is missing prints the command that produces it and exits 2;
`make_all.sh` counts that as *skipped* rather than failed, so a partial run still
shows exactly what is outstanding.

```bash
PYTHON=/root/miniconda3/envs/quetzal/bin/python bash figures/make_all.sh
OUT_DIR=paper/figs FORMAT=png bash figures/make_all.sh
python figures/make_fig01_landscape.py --bench osim --out /tmp/f1.pdf
```

The environment matters: the figure scripts need `numpy`, `matplotlib`, `rdkit`,
`umap-learn` and `scikit-learn`, which live in the `quetzal` conda environment
and not in the system `python3`. Pass `PYTHON=` or activate the environment
first, or every figure "fails" on a missing `numpy`.

## Current scope

**The guide side only.** Fine-tuning and composition results are held back, so
the figures made entirely of them are off by default. Turn them back on with
`WITH_FINETUNE=1` and `WITH_COMPOSED=1`.

**Zaleplon is a negative control.** Only three Zaleplon guides were trained, to
show the objective cannot be guided well. They are excluded from the pooled main
figures and drawn separately as `fig02_zaleplon` / `fig06_zaleplon`. Change the
set with `MAIN_EXCLUDE=`.

**Both flip temperatures are drawn.** Every flip figure shows T = 1.0 and T = 0.3
rather than T = 1.0 alone. Change with `TEMPS=`.

**Seeds never appear in labels.** Run directories carry a trailing `-s0` / `-s42`;
`figstyle.clean_label()` strips it, and `figstyle.distinguishing_labels()`
further reduces a set of run names to the part that actually differs between
them. Route new label sites through those rather than passing raw run names.

## Coverage

| Fig | Script | Reads | Stage | State |
|---|---|---|---|---|
| 1 | `make_fig01_landscape.py` | `dumps/_aggregate/master_table.csv` + harvest/reference JSON | 4, 8 | on |
| 2 | `make_fig02_positional.py` | `flips-guide/flip_report_*.json` | 6 | on |
| 3 | `make_fig03_capacity.py` | harvest JSON | 8 | off — fine-tuning |
| 4 | — | same plot as Figure 8 | | |
| 5 | `make_fig05_ceiling_bins.py` | `ablations/ceiling/ceiling_report.json` | 7 `ceiling` | on |
| 6 | `make_fig06_flip_position_raw.py` | `flips-guide/flip_report_*.json` | 6 | on |
| 7 | `make_fig07_scale_sweep.py` | `ablations/guide-harmonic/ablation_report.json` | 7 `guide` | on |
| 8 | `make_fig08_tempgain.py` | `ablations/tempgain/tempgain_probe.json` | 7 `tempgain` | on |
| 9 | `make_fig09_singles_weights.py` | `ablations/singles-harmonic/singles_weights_report.json` | 7 `singles` | off — composition |
| 10 | `make_fig10_excluded_flow.py` | `ablations/flow/flip-t*/flip_report.json` | 7 `flip`, flow route | on |
| 11 | `make_fig11_collapse_panel.py` | harvest JSON `extended.buckets` | 8 | off — fine-tuning |
| 12 | `make_fig12_mpo_components.py` | harvest JSON `extended.components` | 8 | off — fine-tuning |
| 13 | `make_fig13_sample_efficiency.py` | harvest JSON `extended.curve_top10` | 8 | off — fine-tuning |
| 14 | `make_fig14_quality_vs_score.py` | harvest JSON `extended.quality` | 8 | off — fine-tuning |
| 15 | `make_fig15_chemspace.py` | `dumps/_base/<fam>/` + `dumps/sweep-<fam>-*/` | 4 | on, one per family |
| 16 | `make_fig16_joint_chemspace.py` | the same dumps, all families at once | 4 | on |
| 17 | `make_fig17_decoder_artifact.py` | the same dumps | 4 | on |

## Tables

`make_tables.py` writes every table from the same artifacts, into `OUT_DIR`.
Each skips itself with a note when its input is absent.

| File | Contents |
|---|---|
| `tab_landscape.tex` | score band per guide family per benchmark, with the prior, GEOM and the published baselines |
| `tab_ceiling.tex` | flip rate by prior margin, plus the per-component variance and the dead axis |
| `tab_flip.tex` | pooled flip rate by sequence position at both temperatures, with the margin |
| `tab_scale.tex` | residual scale sweep, and residual magnitude per guide |
| `tab_tempgain.tex` | the range of the learned $T(h)$ and $g(h)$ |
| `tab_chemspace.tex` | joint-space separability and the charged-carbon rate per source |
| `tab_artifact.tex` | headline numbers with decoder artifacts excluded, by objective and by architecture/$\beta$ |
| `tab_pmo.tex` | PMO comparison — fine-tuning, so off by default |

`tab_chemspace.tex` needs the measurements fig17 emits; `make_all.sh` wires that
through automatically, or pass `--sep_json` yourself.

`figstyle.py` holds the shared palette, `rcParams`, artifact paths and loaders.
Published GuacaMol baselines are constants there, not measurements — they come
from Brown et al. (2019), which is why they are written down rather than read
from an artifact.

## What the regenerated figures actually show

Recorded because several numbers differ from the paper as drafted. Every script
prints the values it finds, so a discrepancy is visible rather than silent.

**No Osimertinib fine-tuning runs exist.** `results/oracle_gfn_mols/` contains
runs for Fexofenadine (9), Zaleplon (9) and Perindopril (1), and none for
Osimertinib. The guide sweep does cover Osimertinib — 67 checkpoints, 64 rows in
the master table — so every guide-side figure has it, but the harvest-based
figures never can until those runs are done. `figstyle.available_benches()`
resolves this at run time instead of assuming the paper's pair.

**Figure 1 reproduces.** Osimertinib: 64 configurations, spread 0.041, gap to the
nearest published baseline 0.037, against the paper's 0.042 and 0.035. The GEOM
best-of-10k line for Osimertinib now comes from `make_reference.py`, which scores
the dataset against the objective directly, so it no longer depends on there
being fine-tuning runs.

**Figure 1 has no published baseline for Fexofenadine.** `figstyle.PUBLISHED`
carries GuacaMol numbers for Osimertinib and Perindopril only, so the
Fexofenadine panel draws the GEOM line and no dotted published lines. Add the
Fexofenadine row to `PUBLISHED` from Brown et al. (2019) to get them; the figure
prints a notice rather than silently omitting them.

**Figure 5 reproduces, including the dead axis.** 71.4% of decisions sit at a
margin above 8, against the paper's ~72%, and the flip rate falls to zero above a
margin of 4. Scoring each leaf component separately shows `c1_simECFP6` at
std 0.0000 over reachable molecules — a dead axis, exactly as claimed.

**Figure 7's peak is at 1x, not 4x.** The paper says effect size rises to a
maximum near 4x; in this sweep the mean log-reward shift peaks at 1x and is
already negative at 2x. The residual-to-prior-logit norm ratio is 0.003 to 0.006.

**Figure 8 contradicts the paper's temperature claim.** The paper reports a
learned T between 0.73 and 0.80, i.e. below the `clamp(T, min=1)` floor and
therefore inactive. These checkpoints give T between 1.008 and 1.015 — above 1,
so the mechanism was weakly active rather than inert. The conclusion may still
hold, since a 1.5% softening is negligible, but the stated numbers do not.

**Figure 10 does not show the delivery failure.** The excluded flow-route runs
were described as computing the residual but never applying it, with a flip rate
identically zero. These reports give `delivered_frac = 1.0` and a sampled-flip
rate of 0.011 to 0.020 at both temperatures. The script prints a warning rather
than drawing it as a zero-delivery run; do not caption it as one without
re-checking which runs the claim came from.

**Figure 2 and Figure 6 now have the margin axis.** `mean_gap_by_position` and
`raw.gap_sum_by_position` are recorded, so Figure 2's right-hand axis is real.
Pooled over 237 guide configurations the position-0 flip rate is 0.130 at T=1.0
and 0.055 at T=0.3, and the margin rises from 1.6 at position 0 to above 8 by
position 4 — the decay and its explanation in one plot.

**The nitrogen sanity run dominates Figure 6's spread.** The largest position-0
flip rate in the main pool, 0.921 at T=1.0 and 1.000 at T=0.3, belongs to a
`sweep-nitrogen-*` run. Nitrogen was never optimised for an MPO objective and is
already excluded from the benchmark table. Excluding it here too drops the pool
to 189 guides and the maximum to 0.570 / 0.936. Both versions are written —
`fig06` and `fig06_no_nitrogen`, likewise `fig01` and `fig02` — so the inclusion
rule is a decision rather than an accident. Set
`MAIN_EXCLUDE=zaleplon,nitrogen` to make the exclusion the default.

**Figure 16 finds no separation between objectives.** Embedding every family
together with GEOM and the prior, a 15-NN classifier recovers the objective a
molecule's guide was trained against with 0.203 accuracy, against a majority
class of 0.200 and a shuffled-label null of 0.203. Guiding toward different
objectives does not land anywhere distinguishable.

**The islands in that embedding are a decoder artifact, not chemistry.** The
sharply separated clusters are, to a molecule, structures carrying a formal
charge on a carbon atom — the signature of `rdDetermineBonds` failing to find a
valence-consistent assignment from coordinates and balancing the books with
alternating formal charges. The result sanitises cleanly, so it passes every
downstream filter. GEOM-Drugs, which is real SMILES and never passes through the
decoder, contains **none in 1,200**; the frozen prior and every guided family sit
at **13–18%**. Figure 17 is that result.

**Excluding the artifacts changes nothing that matters.** Re-scoring all 335
guided dumps over the clean molecules alone
(`ablations/exclude_decoder_artifacts.py`) moves the top-10 mean by **−0.0075 on
average**, range −0.069 to 0.000, and leaves the objective-recovery accuracy at
0.201 against 0.203. No reported comparison turns on them. The rate itself is not
uniform, though: base and tempgain sit at 0.145–0.149 at every β, while the
**hidden guide at β=100 reaches 0.251 on average and 1.000 at worst**. Pushing
that architecture hardest damages the geometry rather than the score.

## Still missing

**Figure 9 has no data.** It needs the per-component guides from stage 2
(`compose-osim-c{0,2,3}-*`); there are no `compose-*` checkpoints under
`logs/quetzal-gfn/`, so "each component alone" and "composition weights" are
undefined. `RUNS=` cannot substitute — the figure is about a set of component
teachers, not about any three guides.

```bash
bash scripts/02_train_components.sh      # then: bash scripts/07_ablations.sh singles
```

**The fine-tuning figures need their harvests.** Fexofenadine and Zaleplon are
built; Osimertinib needs the runs themselves.

```bash
BENCH=hard_fexofenadine bash scripts/08_analysis.sh harvest
BENCH=zaleplon_with_other_formula bash scripts/08_analysis.sh harvest
WITH_FINETUNE=1 bash figures/make_all.sh
```

## Reproducing the stage-7 ablations

These need checkpoints rather than dumps, and a GPU. Points at any set of guide
runs under `CKPT_ROOT`:

```bash
OSIM="sweep-osim-base-db-replay_off-b10-s0 \
      sweep-osim-hidden-db-replay_off-b10-s0 \
      sweep-osim-tempgain-db-replay_off-b10-s0"

RUNS="$OSIM" bash scripts/07_ablations.sh guide       # Figure 7
RUNS="$OSIM" bash scripts/07_ablations.sh tempgain    # Figure 8 -- tempgain runs only
ROUTE=flow OUT_ROOT=results/ablations/flow RUNS="$OSIM" \
  bash scripts/07_ablations.sh flip                   # Figure 10

# Figure 5. Pass the leaf scorers explicitly or the right-hand panel collapses to
# a single bar for the assembled objective, which cannot show a dead axis.
EVAL_REWARDS="guacamol:hard_osimertinib=osim_MPO,\
gcomp:osimertinib:0=c0_simFCFP4,gcomp:osimertinib:1=c1_simECFP6,\
gcomp:osimertinib:2=c2_tpsa,gcomp:osimertinib:3=c3_logp" \
  RUNS="$OSIM" bash scripts/07_ablations.sh ceiling
```

Figure 8 needs guides with temperature heads: a base or hidden guide has neither
and comes back as `note: guide has no temp/gain heads`. Use `sweep-*-tempgain-*`.

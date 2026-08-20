# Paper figures

One script per figure in [the paper](../paper/steerability-limits-frozen-3d-priors.pdf).

Every script reads a **committed artifact** under `results/` — a CSV or JSON the
pipeline already wrote — so figures regenerate on a laptop in seconds with no
GPU and no checkpoints. A script whose input is missing prints the command that
produces it and exits 2; `make_all.sh` counts that as *skipped* rather than
failed, so a partial run still shows exactly what is outstanding.

```bash
bash figures/make_all.sh                  # everything -> figures/out/
OUT_DIR=paper/figs FORMAT=png bash figures/make_all.sh
python figures/make_fig01_landscape.py --bench osim --out /tmp/f1.pdf
```

## Coverage

| Fig | Script | Reads | Stage |
|---|---|---|---|
| 1 | `make_fig01_landscape.py` | `dumps/_aggregate/master_table.csv` + harvest JSON | 4, 8 |
| 2 | `make_fig02_positional.py` | `flips-guide/flip_report_*.json` | 6 |
| 3 | `make_fig03_capacity.py` | harvest JSON | 8 |
| 4 | — | same plot as Figure 8 | |
| 5 | `make_fig05_ceiling_bins.py` | `ablations/ceiling/ceiling_report.json` | 7 `ceiling` |
| 6 | `make_fig06_flip_position_raw.py` | `flips-guide/flip_report_*.json` | 6 |
| 7 | `make_fig07_scale_sweep.py` | `ablations/guide-harmonic/ablation_report.json` | 7 `guide` |
| 8 | `make_fig08_tempgain.py` | `ablations/tempgain/tempgain_probe.json` | 7 `tempgain` |
| 9 | `make_fig09_singles_weights.py` | `ablations/singles-harmonic/singles_weights_report.json` | 7 `singles` |
| 10 | `make_fig10_excluded_flow.py` | a `--route flow` flip report (pass `--report`) | 7 `flip`, flow route |
| 11 | `make_fig11_collapse_panel.py` | harvest JSON `extended.buckets` | 8 |
| 12 | `make_fig12_mpo_components.py` | harvest JSON `extended.components` | 8 |
| 13 | `make_fig13_sample_efficiency.py` | harvest JSON `extended.curve_top10` | 8 |
| 14 | `make_fig14_quality_vs_score.py` | harvest JSON `extended.quality` | 8 |

`figstyle.py` holds the shared palette, `rcParams`, artifact paths and loaders.
Published GuacaMol baselines are constants there, not measurements — they come
from Brown et al. (2019), which is why they are written down rather than read
from an artifact.

## Known gaps against the paper as written

Recorded here because they affect what the regenerated figures show, not because
they change the paper's conclusions.

**Figure 2's headline number does not reproduce.** The paper reports a flip rate
of 0.89 at the first atom. Across all 62 committed flip reports the maximum
position-0 rate is **0.452** at T=1.0 and 0.600 at T=0.3, and the pooled mean is
0.072. Figure 6's "factor of six" spread across guides is **181×** in this data.
Whatever produced those numbers is not in `results/flips-guide/`. The scripts
print the values they actually find, so the discrepancy is visible rather than
silent.

**Figure 2's margin axis was never recorded.** `single_flip_ablation.py`
accumulated flip counts per position but not the prior's margin per position, so
the right-hand axis had no saved source. It now records `mean_gap_by_position`
and `raw.gap_sum_by_position`; re-run stage 6 to populate them. Until then
`make_fig02_positional.py` plots the flip curve alone and says so.

**Figure 1's configuration counts differ slightly.** The paper reports 30
Osimertinib and 21 Perindopril configurations with spreads of 0.042 and 0.030.
Reading every matching row gives 35 and 22 with spreads of 0.027 and 0.017. The
gaps to the nearest published baseline match exactly (0.035 and 0.080), so the
axis is right; the difference is in which configurations were included. Decide
the inclusion rule and filter explicitly.

**Figures 5, 7, 8 and 9 have no data yet.** Stage 7 has not been run in the
current layout. Each script names the section that produces it.

# Manuscript 2 pipeline - runbook (HPC)

The data are not in the repo. Run this on the HPC where the raw NCFRP `.sav`,
the codebook, PRAMS, and the CDC live-birth denominators live. Every step writes
into `output_data/` and `output_figures/` under the working directory you pass as
`--outdir`.

**v2 (post lab meeting, gate G10 closed):** the death-year window is **2008-2019**.
`run_all.sh` defaults `YEAR_MIN=2008 YEAR_MAX=2019` and threads `--year-min` /
`--year-max` into every step that loads the case data
(`ncfrp_manuscript_v5.py`, `generate_table{1,2,3}.py`,
`pca_boundary_report_ncfrp.py`, `compare_time_axes.py`). Freeze the v2 outputs
under `../../manuscript_2/AJE/v2/`; never overwrite `AJE/v1/`.

See `AUDIT_FINDINGS.md` for why each step is shaped the way it is. Items still
needing a decision or a codebook lookup are called out inline as **[CHECK]**.

## 0. Get the code onto the HPC (scp, not git)

From the machine that has the audit clone:

```
scp -r ~/Projects/SUID_PCA_manuscripts/manuscript_2_bedsharing_pca \
    <user>@<hpc>:<work_dir>/
```

`<work_dir>` on the HPC must also contain:

```
data/SeattleChildrensNCFRP.sav
data/Codebook_Version5-1_031122.xlsx
from_Ed/Eds_ICD_codes.xlsx
standardized/consolidated/denominator.parquet     # or set $DENOM
standardized/PRAMS/prams.parquet                   # or set $PRAMS
```

Run everything below from `<work_dir>` so the relative paths resolve. `scp` the
`output_data/` + `output_figures/` + `RUN_MANIFEST.md` back afterwards.

## 0b. Environment

```
python -m venv .venv && source .venv/bin/activate      # or a conda env
pip install -r manuscript_2_bedsharing_pca/requirements.txt
python -c "import sys,platform; print(platform.python_version())" > RUN_MANIFEST.md
pip freeze >> RUN_MANIFEST.md
```

## 1. Synthetic smoke test (no real data)

```
python manuscript_2_bedsharing_pca/tests/smoke_synthetic.py
```

Fabricates a tiny NCFRP-shaped CSV + exposure pickles + denominator parquet,
runs `ncfrp_manuscript_v5.main()` end to end, and asserts the audit trail
reconciles and every expected output file is written. Run this first after any
code change.

## 2. State of the "missing" pieces (resolved 2026-09-02)

- **`selection`**: recovered as `src/selection.py` (column-name lists only, no
  case filter). `preprocess_NCFRP.py` no longer imports it.
- **`LR_BW` step**: recovered as `src/lr_bw_impute.py` (birthweight imputation
  `INF3biwt ~ INF3gestage + INFsex`). It only fills the birthweight column; the
  incidence models do not use birthweight, so running the analysis on the
  preprocess output directly gives identical model results. Kept for Table 1.
- **Case definition**: no cause-of-death filter exists anywhere in the pipeline.
  `SeattleChildrensNCFRP.sav` IS the case series (an NCFRP fatality-review
  export). **[CHECK] G6:** confirm how that export was scoped - SIDS-classified
  deaths, SUID, or all NCFRP infant deaths - and set the title/abstract/Methods
  wording to match.
- **`denominator.parquet` / `prams.parquet`**: still HPC-only. Built by
  manuscript 1's `../manuscript_1_sex_pca/src/{format_standardization,concat_files}.py`.

## 3. Codebook checks (T1 resolved; a few HPC confirmations remain)

Resolved from `Codebook_Version5-1_031122.xlsx` (applied to the code):
`INF3gestage` / `INFage` default to blank -> unknown is MISSING, not a sentinel;
`CIRsleepplace 9 = Carseat`, `99 = Unknown`; `INFagecat` 1..5 =
Years/Months/Days/Hours/Minutes; `INF3*tricig` is the cigarette count (so the
smoking trigger is `> 0`, plus the `*tricigunk` flags).

Still confirm on the HPC after the first run:
- `inf3gestunk` survives into `NCFRP_SCH_LR_BW_pop.csv` (the loader drops
  `inf3gestunk == 1` if present);
- `Smoker` code 9 is the only "missing" code for `INF3smbefpreg` /
  `INF3smanytime`;
- the case-selection audit CSV's excluded counts are plausible.

## 4. Preprocess the raw NCFRP file

```
cd <work_dir>          # must contain data/SeattleChildrensNCFRP.sav,
                       # data/Codebook_Version5-1_031122.xlsx, from_Ed/Eds_ICD_codes.xlsx
python <repo>/manuscript_2_bedsharing_pca/src/preprocess_NCFRP.py
# -> data/NCFRP_SCH_sample.csv (2008-2019), data/NCFRP_SCH_pop.csv (all years) - index=False

python <repo>/manuscript_2_bedsharing_pca/src/lr_bw_impute.py \
    --in data/NCFRP_SCH_pop.csv --out standardized/NCFRP_SCH_LR_BW_pop.csv
```

`ncfrp_manuscript_v5.py` re-applies the 2008-2019 death-year window itself
(`--year-min` / `--year-max`) if the file carries `infdodyear`, so it does not
matter whether you feed it the `sample` or `pop` base file.

## 5. Build the population exposure offsets (G2)

```
python <repo>/manuscript_2_bedsharing_pca/src/risk_exposure_joint.py \
    --denominator standardized/consolidated/denominator.parquet \
    --prams       standardized/PRAMS/prams.parquet \
    --outdir      output_data/risk_exposure
```

Writes `joint_smoke_preterm_exposure.pkl` (the 2x2 joint P(smoke, preterm) from
CDC live births), the four marginal `*_exposure.pkl` (for the `--offset-mode
marginal` sensitivity run), `*.csv` copies, and `joint_exposure_summary.txt`
(with the smoke x preterm OR / phi). Reuses manuscript 1's `compute_exposure` /
`compute_preterm` / `compute_bed_share`. Bed sharing stays a marginal from PRAMS
(assumed independent of {smoke, preterm}; state this).

## 6. Choose the PCA window from the data (AUDIT A3.17 / R1.3)

```
python <repo>/manuscript_2_bedsharing_pca/src/pca_boundary_report_ncfrp.py \
    --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \
    --outdir output_data/ncfrp/pca_boundary \
    --min-cell 5 --pca-min 36 --pca-max 90
```

Read `pca_boundary_diagnostics.txt` and `pca_weekly_counts.csv`. Pick
`pca_min` / `pca_max` as the widest contiguous run of weeks with no sparse
subgroup (bed-share / non-bed-share / male / female each >= `min-cell`). Do not
assume 75.

## 7. Main analysis

```
# primary: joint smoke x preterm offsets
python <repo>/manuscript_2_bedsharing_pca/src/ncfrp_manuscript_v5.py \
    --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \
    --exposure-dir output_data/risk_exposure \
    --pca-min <from step 6> --pca-max <from step 6> \
    --offset-mode joint \
    --min-count 10 \
    --gam-alpha-grid 20 \
    --outdir out_joint

# sensitivity: the archived product-of-marginals offsets
python <repo>/manuscript_2_bedsharing_pca/src/ncfrp_manuscript_v5.py \
    ... --offset-mode marginal --outdir out_marginal
```

`run_all.sh` does steps 0-8 in order (analysis + Table 1 + Figure 1 flow +
time-axis comparison).

Writes, per run (ALL / FEMALES / MALES):

- `output_data/ncfrp_flow_audit.csv` - the CONSORT case-selection trail (A9).
- `output_data/ncfrp/two_population_Ns.csv` - full / early / late / peak-excluded N (A3.18).
- `output_data/ncfrp/{run}_early_irr.csv`, `{run}_late_irr.csv`,
  `{run}_gam_exp.csv` - coefficient tables, each checked so every CI brackets
  its estimate (A4.30). Figure 4 reads these. (The `{run}_gam_bedshareXpca_*`
  variant that used to be written here was retired 2026-09-04 - it was
  numerically degenerate; see AUDIT_FINDINGS.md A3.27's close-out and 7b
  below for its corrected replacement.)
- `output_data/ncfrp/{run}_{early,late}_summary.txt` - full model summaries.
- `output_data/ncfrp/smoke_bedshare_case_cooccurrence.{csv,txt}` - the case-only
  chi-square, labelled as co-occurrence not risk (A3.24).
- `output_figures/ncfrp/Figure2_{run}.{png,tiff}`, `Figure4_{run}.{png,tiff}`,
  `output_figures/ncfrp/gam/Figure3_{run}.{png,tiff}`. Figure 2 now shows the
  raw weekly values as faint dots under a centred moving-average line
  (`--smooth-window-pna` / `--smooth-window-pca`, display only; A3.21/G4).

### v2 lab-meeting add-ons (JOINT run only; each fails soft, `--no-extra-strata` skips all)

- **LBW / NONLBW runs** - the full two-stage pipeline over `inf3biwt < 2500` vs
  `>= 2500`, exactly like FEMALES / MALES. `{lbw,nonlbw}_{early,late}_irr.csv`,
  `{lbw,nonlbw}_gam_exp.csv`, `Figure{2,3,4}_{LBW,NONLBW}.*`. Bed-sharing IRR is
  read across the two strata (no population LBW denominator - state this). The
  **LBW late window (~1k cases) is too thin for the 7-term NB** (alpha collapses,
  the CI-bracket check fails, no `lbw_late_irr.csv`); use the interaction model
  below for the late-window birthweight question.
- **Smoking-stratified windows** (`run_windowed_stratified`) ->
  `output_data/ncfrp/stratified/smoke{0,1}_{early,late}_irr.csv` +
  `output_figures/ncfrp/Figure5_smoke_stratified.{png,tiff}`. Reduced design
  (no `smoke` term). Lab action items 2 & 3.
- **`under1mo_by_smoke_bedshare`** -> `output_data/ncfrp/stratified/under1mo_by_smoke_bedshare.{csv,md}`.
  Descriptive deaths before 1 month of age, smoker x bed sharing.
- **`run_windowed_lbw_interaction`** ->
  `output_data/ncfrp/lbw_interaction_{early,late}_irr.csv` - windowed NB with a
  `<2500 g` indicator and a `bed_share x lbw` term (offset-neutral in lbw). The
  early cell may be underpowered; a failure is logged, not raised.

### 7b. Bed-share x PCA varying-coefficient robustness check (A3.27 replacement)

`run_all.sh` step 5c. Corrected replacement for the retired `gam_res_int`
variant (see AUDIT_FINDINGS.md A3.27's close-out for the root cause and why
this is a standalone script rather than a `GLMGam` patch). Additive to
`out_joint`/`out_marginal` - new file names, new figure subdirectory, nothing
it writes overwrites anything from step 5/5b.

```bash
python <repo>/manuscript_2_bedsharing_pca/src/gam_bedshare_pca_varycoef.py \
    --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \
    --exposure-dir output_data/risk_exposure \
    --pca-min "$PCA_MIN" --pca-max "$PCA_MAX" \
    --year-min "$YEAR_MIN" --year-max "$YEAR_MAX" \
    --offset-mode joint --outdir out_joint
# and again with --offset-mode marginal --outdir out_marginal
```

Writes, per run (ALL / FEMALES / MALES, each fails soft):

- `output_data/ncfrp/{run}_bedshare_pca_varycoef_irr.csv` /
  `{run}_bedshare_pca_varycoef_summary.csv` - the same `save_nb_model_outputs`
  single-source coefficient table used by the piecewise NB models (A4.30),
  now also SE-sanity-checked (`--se-sanity-max`, default 20).
- `output_figures/ncfrp/varycoef/Figure3b_{run}.{png,tiff}` - fitted
  incidence trajectories (panel A) and the bed-sharing rate-ratio curve across
  PCA (panel B), mirroring `Figure3_{run}` but from the corrected
  varying-coefficient model.

**Before trusting the output for the manuscript**, check (see AUDIT_FINDINGS.md
A3.27/A3.28 for why these are the right thresholds):

- Every `std_err` well under ~5 on the log scale (the broken variant's was
  ~329) - `check_design_rank`/the SE-sanity guard should not have needed to
  raise; if a stratum's run logs a warning and is skipped, that's real
  evidence worth reporting, not a bug to silence.
- `bed_share`/interaction-term IRRs the same broad order of magnitude as the
  working `{run}_gam_exp.csv`'s `bed_share` IRR (~2.7, CI ~[2.2, 3.3]) - not
  astronomical.
- `Figure3b_{run}.png`'s CI band should be a plausible width throughout: if it
  widens sharply specifically at the high-PCA tail, that is genuine data
  sparsity (the bed_share=1 tail is thin - `two_population_Ns.csv`) reasserting
  itself, not a construction bug - truncate the reported domain rather than
  trust a falsely-confident curve out there.
- Compare the shape against the existing `Figure3_{run}` (deviation-smooth)
  curve - they should agree on direction.

If this checks out, it's a strong candidate to also satisfy Ed's Figure 3B
redesign ask (a fitted rate-ratio curve with a proper CI) - a manuscript
decision, not a pipeline one.

## 8. Manuscript-support outputs (run after step 7; run_all.sh does this)

- `src/generate_table1.py` -> `output_data/ncfrp/table1_by_bedshare.{csv,md}`.
  "Characteristics of the analytic SUID cohort by bed-sharing status", counts +
  column %, on the pre-peak-split cohort (AUDIT A4.29 / R1.8 / Ed E14). Sex is a
  row; birthweight uses the LR_BW-imputed column. The row-building logic
  (`build_cohort_table`) is shared with the two tables below.
- `src/generate_table2.py` -> `output_data/ncfrp/table2_by_birthweight.{csv,md}`.
  Same layout, stratified by birthweight category (<2500 / 2500-3499 / >=3500 g
  by default; `--lbw-cut` / `--hbw-cut` to change). Bed sharing becomes a row;
  the birthweight block is dropped. Needs the LR_BW-imputed `inf3biwt` column.
- `src/generate_table3.py` -> `output_data/ncfrp/table3_by_sex.{csv,md}`. Same
  layout, stratified by infant sex (the direct analogue of manuscript 1's
  `table1_by_sex`). Bed sharing becomes a row; the "Infant sex" block is dropped.
- `src/generate_figure1_flow.py` -> `output_figures/ncfrp/Figure1_flow.{png,tiff}`
  and `output_data/ncfrp/figure1_flow.txt`. CONSORT flow straight from
  `ncfrp_flow_audit.csv` + `two_population_Ns.csv`, so every count reconciles and
  the two analytic populations are shown explicitly (AUDIT A4.28 / R1.4 / Ed E4).
- `src/compare_time_axes.py` -> `output_data/ncfrp/time_axes/`: `gam_fit_comparison.csv`
  (PCA vs chronological-age GAM deviance/AIC/llf), `observed_by_axis.{png,tiff}`
  (now with a `--smooth-window`-wk moving-average overlay),
  `pca_vs_chronological_{scatter.png,correlation.txt}`,
  `proportion_preterm_{by_pca_week,by_chronological_week}.csv` +
  `proportion_preterm_by_age.png` (AUDIT A6.40-42, A3.26, Ed E8/E9/E13/E15), and
  `time_axes/reporting_artifact/`: `periodogram_{pca,chrono}.{csv,png}`,
  `acf_{pca,chrono}.csv`, `spikes_by_age_unit.{csv,md,png}`,
  `reporting_artifact_summary.txt` - the affirmative check that the repeating
  local peaks fall exactly on the completed-week images of whole-month ages
  (weeks {4,8,12,17,21,25,30,34,38,42,47} under preprocess's 30-day month),
  with unequal 4-5 wk gaps (calendar arithmetic, not a fixed biological period),
  essentially all in `infagecat == 2` records and absent from days-reported
  ones. ~91% of NCFRP ages are months-reported so there is no "drop and re-fit"
  sensitivity; the summary notes the windowed >=4-wk aggregation + NB dispersion
  + GAM smoothing as why it does not touch the inference.

- `src/generate_supp_firstweek_table.py` (NOT in run_all.sh - it compares two
  runs) -> `<outdir>/supp_table_firstweek_sensitivity.{csv,md,xlsx}`. Manuscript
  Supplementary Table 3: early/late/GAM IRRs under the primary rule (deaths at
  7-13 d excluded, joint offsets) vs the `FIRST_WEEK_CUT=0` run (retained) vs
  the primary run's marginal offsets. Pass the three `output_data/ncfrp`
  directories explicitly, e.g. on the HPC after both runs exist:

  ```
  python src/generate_supp_firstweek_table.py \
      --primary-dir     ../manuscript2_pre_v7/out_joint/output_data/ncfrp \
      --sensitivity-dir out_joint/output_data/ncfrp \
      --marginal-dir    ../manuscript2_pre_v7/out_marginal/output_data/ncfrp \
      --outdir output_data/ncfrp
  ```

  The script refuses to write if the two joint runs differ in the late window
  (they must be identical: the export has no deaths before day 7, so the
  sensitivity run only adds days 7-13, all below PCA 47). Reads saved CSVs only.

## 8b. Breastfeeding sensitivity analysis (A/B design; not in run_all.sh)

`src/bfed_sensitivity.py` reruns the ENTIRE pipeline with the case file filtered
on NCFRP `INF3breast` before any cleaning (Y = ever breastfed, N = never,
U = unknown, K = known, ALL = reproduction check), compares Y vs N (and K vs U)
on every windowed term and the GAM main effects with a Wald test on the
log-IRR difference, and reports the share of the analytic cohort with unknown
breastfeeding status (overall and by bed sharing, smoking, preterm, sex,
window, death year). Y/N models get breastfeeding-specific offsets (joint
smoke x preterm from denominator rows with the same `bfed`; bed-sharing
prevalence from PRAMS respondents with the same `brstfed`), plus a national-
offset check. Run after `run_all.sh` (needs `output_data/risk_exposure/`):

```
python src/bfed_sensitivity.py \
    --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \
    --denominator standardized/consolidated/denominator.parquet \
    --prams standardized/PRAMS/prams.parquet \
    --national-exposure-dir output_data/risk_exposure \
    --outdir out_bfed --pca-min 36 --pca-max 90 --year-min 2008 --year-max 2019 --first-week-cut 1
```

Outputs in `out_bfed/`: `bfed_sensitivity_summary.md`, `bfed_comparison_{bfed,national}.{csv,md}`,
`bfed_missingness.{csv,md}`, `bfed_pca_distribution.{csv,png}`, `offset_proportions.csv`,
`run_<group>_<offsets>/` (standard pipeline outputs per subset), `bfed_sensitivity.log`.
The log states whether the ALL run reproduces the primary bed-sharing IRRs (PASS/FAIL).
Copy `out_bfed/` back as its own `AJE/v<N>/` bundle.

## 9. Hand back

Copy `output_data/`, `output_figures/`, and `RUN_MANIFEST.md` back for review
against `AUDIT_FINDINGS.md` and for the reviewer-proofing memo.

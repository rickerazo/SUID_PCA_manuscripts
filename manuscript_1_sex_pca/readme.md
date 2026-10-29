# Manuscript 1: Developmental Timing of Male Excess in Sudden Unexpected Infant Death

## Overview

This folder contains the analysis code for the manuscript:

**Developmental Timing of Male Excess in Sudden Unexpected Infant Death**
Ricardo Erazo and Jan-Marino Ramirez (Seattle Children's Research Institute; University of Washington).
Submitted to the *American Journal of Epidemiology*.

The study asks whether the male excess in sudden unexpected infant death (SUID) is constant across infancy or emerges during a specific developmental interval. Infants are aligned on postconceptional age (PCA), defined as gestational age at birth plus postnatal age. Chronological age alone can obscure developmental structure because infants are born at different gestational ages; PCA aligns them on a maturational axis so that sex-specific risk trajectories can be compared across development.

Main finding: PCA-aligned SUID incidence rises, peaks at PCA 43 to 46 weeks, and declines. Male and female incidence is similar during the rise (early window IRR 1.14, 95% CI 0.85 to 1.53) and diverges after the peak (late window IRR 1.36, 95% CI 1.29 to 1.42). The male excess is therefore not present from birth but emerges after the developmental peak.

## Data source

U.S. CDC linked birth/infant death records, birth cohorts 2014 to 2021.

- Cases: SUID defined by underlying cause-of-death ICD-10 codes R95 (SIDS), W75 (accidental suffocation and strangulation in bed) and R99 (other ill-defined and unspecified causes). Deaths in the first postnatal week are excluded.
- Denominator: U.S. live births 2014 to 2021 (natality files), used to build the population offsets.

Raw data are **not distributed in this repository**. Reproducing the analysis requires independent access to the CDC linked birth/infant death and natality files and reconstruction of the intermediate files described below.

### Case selection

| Step | n |
|---|---|
| SUID deaths, 2014 to 2021 | 25,257 |
| Excluded: first-week deaths | 1,177 |
| Excluded: PCA < 36 weeks | 110 |
| Excluded: PCA > 90 weeks | 28 |
| Excluded: invalid maternal smoking data | 256 |
| **Analytic cohort** (Table 1, Figures 2 and 3) | **23,686** |
| Excluded from windowed models only: peak interval, PCA 43 to 46 weeks | 4,438 |
| **Windowed-model series** (Figure 4): early < 43 weeks / late >= 47 weeks | **19,248** (2,946 / 16,302) |

The analytic PCA range (36 to 90 weeks) is the range in which every week has at least five male and at least five female deaths. Because first-week deaths are excluded, term infants cannot enter the cohort before PCA week 39, so PCA weeks 36 to 38 contain preterm infants only.

## Variables

- Postconceptional age (PCA): gestational age at birth (`combgest`) plus postnatal age at death (`infage`), in weeks. PCA is the time scale, not a covariate.
- Infant sex: primary exposure.
- Maternal smoking during pregnancy (`cig_rec`, any reported smoking): covariate.
- Preterm birth (gestational age < 37 completed weeks): covariate.
- Breastfeeding at discharge (`BFED`): not a model covariate (recorded only at discharge, and a postnatal time-varying exposure); used in a stratified sensitivity analysis.

## Analytical framework

1. **Full-curve model.** A negative binomial generalized additive model with spline smooths over PCA estimates sex-specific SUID trajectories across PCA weeks 36 to 90 (Figure 3).
2. **Confirmatory windowed models.** Negative binomial regressions fit separately in a prespecified early rising window (PCA 36 to 42) and late declining window (PCA >= 47), each with a log-linear PCA term, sex, maternal smoking, preterm birth and the prespecified Smoking x Preterm and Sex x Preterm interactions (Figure 4). The peak interval (PCA 43 to 46) is summarized descriptively instead, because incidence turns from rising to declining within it and a log-linear PCA term cannot represent that change.
3. **Offset.** Each stratum's offset is the log of the product of the national live-birth proportions for its sex, maternal smoking and preterm categories (`risk_exposure_fixed.py`).

## Workflow

Scripts are listed in run order. All scripts sit in one folder (`src/`) and import each other from that folder. Run `python <script>.py --help` or read the module docstring for exact usage, inputs and outputs.

**1. Raw data preparation**
- `format_standardization.py`: standardizes the yearly CDC source files into a common structure.
- `concat_files.py`: assembles the standardized yearly files into consolidated datasets.
- `consolidate_icd_datasets.py`: consolidates records by ICD-10 cause-of-death code.

**2. Case series and offsets**
- `audit_subject_breakdown.py`: shared case-selection library (`clean_case_series`, `exclude_invalid_smoking`, `split_analytic_windows`, `table1_by_sex`) used by every downstream script; run directly, it writes an audited case-selection trail and the PCA-boundary report.
- `risk_exposure_fixed.py`: computes the population offset pickles (sex, maternal smoking, preterm) from the live-birth denominator.

**3. Descriptive outputs**
- `generate_table1.py`: Table 1 (cohort characteristics by infant sex).
- `generate_figure2_rate_curves.py`: Figure 2 (weekly SUID rates by postnatal age and by PCA).

**4. Models**
- `generate_figure3_gam_curves.py`: Figure 3 (full-curve negative binomial GAM; observed and fitted counts, male minus female difference, log ratio).
- `run_windowed_nb_models.py`: confirmatory windowed negative binomial regressions; also imported as a library by the scripts below.
- `generate_figure4_forest_plots.py`: Figure 4 (forest plots of windowed-model IRRs).
- `generate_supplemental_table23.py`: Supplemental Tables 2 and 3 (IRRs and derived absolute risks by risk-factor profile, early and late windows).
- `generate_supplemental_table4.py`: Supplemental Table 4 (late-window model estimates).

**5. Supporting and sensitivity analyses (peer review)**
- `analyze_peak_postnatal_age.py`: postnatal (chronological) age at death for the 4,438 deaths in the PCA 43 to 46 peak, overall and by preterm status, maternal smoking and sex; Supplemental Tables 5 to 7 and Supplemental Figure 1.
- `r2_checks.py`: Supplemental Table 1 (early-window estimates with the lower bound at PCA 36 vs 38) with a reproduction check against the main estimate; counts by PCA week, term status and sex for PCA 36 to 46; BFED availability in the denominator; denominator totals.
- `bfed_sensitivity.py`: breastfeeding sensitivity analysis. Re-runs the whole pipeline separately for infants breastfed (BFED=Y) and not breastfed (BFED=N) at discharge, with offsets from live births with the same breastfeeding status, and compares windowed-model estimates between groups with a Wald test on the difference in log IRRs. Also reruns the full analysis (ALL) and checks that it reproduces the published estimates.

### Outputs by manuscript item

| Manuscript item | Script |
|---|---|
| Figure 1 (case selection) | `audit_subject_breakdown.py` (counts) |
| Table 1 | `generate_table1.py` |
| Figure 2 | `generate_figure2_rate_curves.py` |
| Figure 3 | `generate_figure3_gam_curves.py` |
| Figure 4 | `run_windowed_nb_models.py`, `generate_figure4_forest_plots.py` |
| Breastfeeding sensitivity analysis (Results) | `bfed_sensitivity.py` |
| Supplemental Table 1 | `r2_checks.py` |
| Supplemental Tables 2 and 3 | `generate_supplemental_table23.py` |
| Supplemental Table 4 | `generate_supplemental_table4.py` |
| Supplemental Tables 5 to 7, Supplemental Figure 1 | `analyze_peak_postnatal_age.py` |

## Figures

`figures/` holds the figures as submitted: `Figure1.pdf` (case-selection flow diagram), `Figure2.png`, `Figure3.png`, `Figure4.png` and `Supplemental_Figure1.png`. The submission used 300 dpi TIFF versions of Figures 2 to 4, which are not stored here because of their size.

## Requirements

See `requirements.txt` in this folder for the packages these scripts need. The repository-root `environment-*.yml` and `environment-explicit.txt` files capture the broader HPC environment shared with manuscript 2 and are a superset of what is needed here.

## Reproducibility notes

- The scripts were run on an HPC cluster where the source data are stored.
- The analysis scripts (steps 3 to 5, and `audit_subject_breakdown.py`) take file paths as command-line arguments. The preprocessing scripts do not:
  - `format_standardization.py`, `concat_files.py` and `consolidate_icd_datasets.py` read and write relative paths (`MS_data/`, `standardized/`, `standardized/consolidated/`) and are run from the data working directory.
  - `risk_exposure_fixed.py` reads the natality denominator and a PRAMS file from fixed cluster paths at the top of the file (edit these two lines before running) and writes to `output_data/risk_exposure/`. The PRAMS file feeds only a bed-sharing offset used by manuscript 2; manuscript 1 uses the sex, smoking and preterm offsets.
- Each script's docstring states the input format and columns it expects.
- `r2_checks.py` and `bfed_sensitivity.py` log a PASS/FAIL check that the main estimates (early Male IRR 1.14, late Male IRR 1.36) are reproduced.

## A note on this code release

This folder was revised during peer review after a data-pipeline audit found and corrected several bugs in an earlier internal version of the analysis (for example, a case-selection step that had silently become a no-op, a stale PCA-window boundary, an exposure-computation bug, and a boundary-inclusivity inconsistency between the case series and the regression models). Every script here reflects the corrected pipeline, and all reported results come from it. The earlier exploratory notebook is not included, so that this release reflects a single, internally consistent analysis.

## License

MIT License; see `LICENSE` at the repository root. The license covers the code only, not the CDC source data.

## Maintenance status

This folder is an archival code release accompanying the manuscript. Ongoing support and feature development are not planned.

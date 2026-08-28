# Manuscript 1 — Developmental Timing of Male Excess in Sudden Unexpected Infant Death

## Overview

This folder contains the analysis code associated with the manuscript:

**Developmental Timing of Male Excess in Sudden Unexpected Infant Death**

This project examines whether the well-known male excess in sudden unexpected infant death (SUID) is constant across infancy or instead emerges during a specific developmental interval. The analysis uses postconceptional age (PCA), defined as gestational age at birth plus postnatal age, as the primary developmental time scale.

The core epidemiologic idea is that chronological age alone may obscure developmental structure because infants are born at different gestational ages. PCA aligns infants on a maturational axis and allows sex-specific risk trajectories to be compared across development. The manuscript uses this framework to test whether male excess is present from birth or instead emerges after the developmental incidence peak.

## Data source

This analysis was conducted using **U.S. CDC linked birth–infant death records**. The analytic series uses national data spanning **2014 to 2021**, with SUID defined using ICD-10 codes **R95, R99, and W75**, excluding deaths in the first postnatal week.

Raw data are **not distributed in this repository**. Reproducing the analysis requires independent access to CDC linked birth–infant death data and reconstruction of the intermediate analytic files described below.

## A note on this code release

This folder was revised during peer review after a data-pipeline audit found and corrected several real bugs in an earlier internal version of the analysis (e.g., a case-selection step that had silently become a no-op, a stale PCA-window boundary, an exposure-computation bug, and a boundary-inclusivity inconsistency between the case series and the regression models). Every script here reflects the corrected pipeline. The original exploratory notebook that predated this correction is not included, so that this release reflects a single, internally consistent analysis rather than a mix of superseded and current code.

## Primary variables

- postconceptional age (PCA), defined as gestational age at birth plus postnatal age at death
- infant sex
- maternal smoking during pregnancy
- preterm birth (gestational age at birth < 37 completed weeks)
- live-birth denominators for offset/exposure construction

## Analytical framework

The analysis is organized around two complementary model types:

1. **Full-curve developmental model.** A negative binomial generalized additive model (GAM) estimates the smooth developmental trajectory of SUID incidence across PCA weeks, used to visualize the rise/peak/decline structure and to compare fitted male and female trajectories.
2. **Confirmatory windowed models.** Piecewise negative binomial regression is fit separately in prespecified early (rising) and late (declining) developmental windows, producing clinically interpretable incidence rate ratios. The peak interval itself is summarized descriptively rather than modeled directly, given high week-to-week variability within it.

## Workflow

Scripts are listed in the order they are run. Each script takes its inputs/outputs via command-line arguments; run `python <script>.py --help` or read its module docstring for exact usage.

**1. Raw data preparation** (`src/`)
- `format_standardization.py` — standardizes yearly source files into a common structure
- `concat_files.py` — assembles standardized yearly files into consolidated datasets
- `consolidate_icd_datasets.py` — consolidates records by ICD-10 cause-of-death code

**2. Case series and exposure construction**
- `audit_subject_breakdown.py` — the shared case-selection library (`clean_case_series`, `exclude_invalid_smoking`, `split_analytic_windows`, `table1_by_sex`) used by every downstream script; also runnable directly for an audited case-selection trail
- `risk_exposure_fixed.py` — computes population-level exposure/offset pickles (sex, smoking, preterm, PCA) from the live-birth denominator, consumed by the modeling scripts below

**3. Descriptive outputs**
- `generate_table1.py` — Table 1 (cohort characteristics by infant sex)
- `generate_figure2_rate_curves.py` — Figure 2 (descriptive rate curves by postnatal age and PCA)

**4. Modeling**
- `generate_figure3_gam_curves.py` — Figure 3 (full-curve negative binomial GAM)
- `run_windowed_nb_models.py` — the confirmatory windowed negative binomial regressions (early/late developmental windows); also importable as a library by the two scripts below
- `generate_figure4_forest_plots.py` — Figure 4 (forest plots of windowed-model incidence rate ratios)
- `generate_supplemental_table23.py` — Supplemental Tables 2–3 (early/late-window model coefficients by risk-factor profile)
- `generate_supplemental_table4.py` — Supplemental Table 4 (late-window coefficient table)

**5. Supporting analysis (peer-review response)**
- `analyze_peak_postnatal_age.py` — checks the manuscript's PCA-defined peak interval against the postnatal (chronologic) age scale used elsewhere in the SIDS literature, stratified by preterm status, maternal smoking, and infant sex; produces Supplemental Tables 5–7 and the accompanying supplemental figure

## Requirements

See `requirements.txt` in this folder for the minimal set of packages these scripts need. The repository-root `environment-*.yml`/`environment-explicit.txt` files capture a broader HPC environment shared across both manuscript folders in this repository and are a superset of what's needed here.

## Reproducibility notes

Because the source data are not bundled here, execution requires:
- access to CDC linked birth–infant death data
- reconstruction of the intermediate analytic files each script expects (see each script's docstring for exact expected input format/columns)
- local configuration of file paths (all scripts take paths as command-line arguments; none hard-code a local path)

## Maintenance status

This folder is a code release accompanying the manuscript. Ongoing support, feature development, and active maintenance are not planned.

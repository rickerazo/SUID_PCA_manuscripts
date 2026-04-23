# Manuscript 1 — Developmental Timing of Male Excess in Sudden Unexpected Infant Death

## Overview

This folder contains archival code associated with the manuscript:

**Developmental Timing of Male Excess in Sudden Unexpected Infant Death**

This project examines whether the well-known male excess in sudden unexpected infant death (SUID) is constant across infancy or instead emerges during a specific developmental interval. The analysis uses postconceptional age (PCA), defined as gestational age at birth plus postnatal age, as the primary developmental time scale.

The core epidemiologic idea is that chronological age alone may obscure developmental structure because infants are born at different gestational ages. PCA aligns infants on a maturational axis and allows sex-specific risk trajectories to be compared across development. The final manuscript uses this framework to test whether male excess is present from birth or instead emerges after the developmental incidence peak. :contentReference[oaicite:0]{index=0}

## Data source

This analysis was conducted using **U.S. CDC linked birth–infant death records**.

The manuscript-specific analytic series uses national data spanning **2014 to 2021**, with SUID defined using ICD-10 codes **R95, R99, and W75**, excluding deaths in the first postnatal week. :contentReference[oaicite:1]{index=1}

Raw data are not distributed in this repository.

## Main research question

The main question is whether male excess in SUID is present from birth or whether it emerges only after the developmental incidence peak.

More broadly, this project asks whether developmental alignment by PCA reveals temporal structure in sex differences that is obscured when infant age is represented only as chronological postnatal age.

## Primary variables

Key variables used in this project include:

- postconceptional age (PCA)
- infant sex
- maternal smoking during pregnancy
- preterm birth
- live-birth denominators for offset construction

Depending on the script, additional intermediate variables or exploratory subgroup definitions may also appear. Not all variables present in this folder are necessarily used in the final manuscript models.

## Analytical framework

The analysis is organized around two complementary model types:

### 1. Full-curve developmental model

A negative binomial generalized additive model (GAM) is used to estimate the smooth developmental trajectory of SUID incidence across PCA weeks.

This model is used to:

- visualize the rise, peak, and decline structure of incidence
- compare fitted male and female trajectories across development
- summarize sex contrasts across PCA

### 2. Confirmatory windowed models

Piecewise negative binomial regression models are fit separately in prespecified developmental windows.

These models are used to estimate clinically interpretable incidence rate ratios during:

- an early rising developmental interval
- a later declining developmental interval

The peak interval is summarized descriptively rather than treated as a stable inferential window. :contentReference[oaicite:2]{index=2}

## Folder purpose

This folder is provided so readers and reviewers can inspect:

- preprocessing logic used to derive developmental timing variables
- manuscript-specific analytic scripts
- exploratory and confirmatory modeling code
- figure-generation code
- the overall structure linking raw mortality records to the reported developmental analyses

This is not intended to be a general-purpose software package. It is an archival companion to the manuscript.

## General workflow

The code in this folder follows a manuscript-oriented workflow rather than a single packaged pipeline. In broad terms, the process is:

1. **Standardize source data formats** so that files from different years or sources can be handled consistently.
2. **Construct harmonized analytic cohorts** from the standardized files.
3. **Generate descriptive summaries** of infant death counts and sex ratios across age.
4. **Evaluate distributional properties** of the data, including mean–variance structure and overdispersion.
5. **Compute incidence and subgroup-specific rates** using numerator and denominator components.
6. **Fit statistical models** ranging from exploratory regressions to the final manuscript models.
7. **Generate manuscript-linked figures and summary outputs**.

Because this repository is archival, some scripts reflect intermediate or exploratory stages of the project and may precede the final analytic specification used in the manuscript.

## Notes on script roles

Several scripts in this folder reflect earlier or supporting stages of the project. These were retained for transparency because they document how the analytic logic evolved.

Examples of script roles include:

- **data standardization**
  - scripts such as `format_standardization.py` were used to bring source data files into a common structure before downstream processing

- **cohort construction and file consolidation**
  - scripts such as `concat_files.py` or similar utilities were used to assemble standardized records into manuscript-specific analytic datasets

- **descriptive sex-ratio analyses**
  - scripts such as `infage_ratios.py`, `months_snapshots.py`, and `weeks_snapshots.py` were used to summarize male/female death patterns across infant age scales

- **incidence and denominator calculations**
  - scripts such as `incidence.py`, `smoker_to_nonsmoker_proportions.py`, and denominator-processing utilities were used to construct rate-based summaries and supporting epidemiologic quantities

- **distributional checks**
  - scripts such as `check_dataset.py` were used to inspect mean–variance relationships and assess overdispersion, which informed the eventual choice of count-modeling strategy

- **earlier modeling steps**
  - some scripts implement exploratory linear, Poisson, or binomial-type regressions used during method development
  - these scripts should be interpreted as part of the project history, not necessarily as the exact final models reported in the published manuscript

- **final manuscript modeling**
  - the final inferential framework centers on full-curve GAMs and confirmatory windowed negative binomial models aligned on PCA

## Exploratory versus final analysis

A practical note for readers:

This folder may contain scripts from both **exploratory analysis** and **final manuscript analysis**. That is intentional. The goal of this archive is not only to expose the final figure-generating code, but also to preserve the sequence of analytic decisions that led to the final model specification.

Accordingly:

- some scripts are descriptive
- some are diagnostic
- some reflect earlier model ideas that were later replaced
- some correspond directly to the final manuscript results

Readers interested in reproducing the final manuscript should prioritize scripts and outputs associated with PCA-based GAM and piecewise negative binomial analyses.

## Expected contents

Typical contents may include:

- `src/` for analysis utilities and modeling functions
- `scripts/` for manuscript-specific execution steps
- `figures/` for generated or regenerable outputs
- intermediate processing or cohort-construction scripts
- archived exploratory scripts retained for transparency

Environment files are provided at the **repository root**, because both manuscript folders in this archival repository share the same conda/mamba software environment.

## Reproducibility notes

Because the source data are not bundled here, execution may require:

- access to CDC linked birth–infant death data
- recreation of manuscript-specific analytic datasets
- local adjustment of paths, filenames, and environment settings
- selective interpretation of which scripts correspond to exploratory versus final analyses

Environment details for this analysis are provided in the repository-level files, such as:

- `environment-minimal.yml`
- `environment-full.yml`
- `environment-explicit.txt`
- `mamba-info.txt`

Accordingly, this folder should be treated as an archival analysis companion rather than a one-command reproduction package.

## Summary of findings addressed by this code

At a high level, this analysis supports the interpretation that SUID incidence follows a developmental rise, peak, and decline pattern when aligned by PCA, and that male excess emerges primarily after the developmental peak rather than being present uniformly from birth. :contentReference[oaicite:3]{index=3}

## Maintenance status

This folder is archived for manuscript transparency. Ongoing support, feature development, and active maintenance are not planned.
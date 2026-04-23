# SUID Developmental Timing Analyses

This repository is an archival code release for two related manuscript projects on sudden unexpected infant death (SUID). Both projects use a developmental epidemiology framework based on postconceptional age (PCA), full-curve generalized additive modeling, and piecewise negative binomial regression, but they were implemented as separate analysis pipelines on different data sources.

## Purpose

The purpose of this repository is transparency and reproducibility. It is intended to allow reviewers and readers to inspect the analytical logic, preprocessing decisions, modeling structure, and figure-generation workflow associated with the manuscripts.

This repository is not intended to function as a maintained software package or a unified production pipeline.

## Repository contents

This repository contains two manuscript-specific analysis folders:

- `manuscript_1_sex_pca/`  
  Code associated with the manuscript on developmental timing of male excess in SUID using CDC linked birth–infant death data.

- `manuscript_2_bedsharing_pca/`  
  Code associated with the manuscript on developmental timing of SUID and the infant sleep environment using NCFRP case data and PRAMS-derived denominator information.

Each manuscript folder contains its own documentation, scripts, and workflow notes.

## Scientific rationale shared across both manuscripts

Although the implementations differ, both projects are motivated by the same core conceptual framework:

- infant age is better represented on a developmental scale than by chronological postnatal age alone
- postconceptional age (PCA) provides an epidemiologic alignment variable that accounts for maturity at birth
- SUID incidence across infancy can be represented as a developmental rise, peak, and decline trajectory
- this trajectory can be examined using:
  - full-curve generalized additive models
  - confirmatory piecewise negative binomial regression

The manuscripts differ in their exposures, data sources, and preprocessing logic, but they share this developmental modeling perspective.

## Important note about data availability

The raw analytic data are **not included** in this repository.

This repository does not distribute restricted, case-level, or protected source data. Reproducing the analyses requires independent access to the relevant datasets and, where applicable, the appropriate permissions or data-use approvals.

In general:

- Manuscript 1 relies on CDC linked birth–infant death data
- Manuscript 2 relies on NCFRP case data and external denominator construction using PRAMS and live-birth data

Please refer to the manuscript-specific README files for details.

## Reproducibility scope

This repository is intended to make the analytical workflow inspectable and auditable. Because the underlying datasets are not distributed here, full end-to-end execution may require:

- access to the original source datasets
- reconstruction of intermediate analytic files
- adaptation of local file paths and environment settings

Accordingly, this repository should be understood as an archival research companion rather than a turnkey executable software package.

## Suggested structure

```text
.
├── README.md
├── manuscript_1_sex_pca/
│   ├── README.md
│   ├── environment.yml
│   ├── src/
│   ├── scripts/
│   └── figures/
└── manuscript_2_bedsharing_pca/
    ├── README.md
    ├── environment.yml
    ├── src/
    ├── scripts/
    └── figures/
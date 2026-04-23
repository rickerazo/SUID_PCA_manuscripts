# Manuscript 2 — Developmental Timing of SUID and the Infant Sleep Environment

## Overview

This folder contains archival code associated with the manuscript:

**Developmental Timing of SUID and the Infant Sleep Environment**

This project examines how SUID incidence changes across development when infants are aligned by postconceptional age (PCA), with particular attention to the infant sleep environment and related exposures.

The analysis uses PCA as the primary developmental time scale in order to better represent physiologic maturity than chronological postnatal age alone.

## Data sources

This project uses a multi-source design.

### Case data
Case-level information comes from the **National Center for Fatality Review and Prevention (NCFRP) Case Reporting System**.

### Denominator inputs
External sources are used to construct exposure-adjusted denominator information:

- **CDC live-birth data** for smoking and prematurity distributions
- **PRAMS** for population prevalence of bed sharing

Raw data are not distributed in this repository.

## Important exposure-definition note

In this project, **case-level bed sharing is classified from NCFRP death-scene sleep-environment variables**, not from PRAMS.

PRAMS is used to estimate **population bed-sharing prevalence for denominator construction**, not to classify the case exposure itself.

This distinction is central to the analytic design.

## Main research question

The main question is how SUID incidence evolves across developmental time in relation to:

- bed sharing
- maternal smoking during pregnancy
- preterm birth

and whether these exposures are associated with distinct or shared developmental patterns of risk.

## Primary variables

Key variables used in this project include:

- postconceptional age (PCA)
- bed sharing
- maternal smoking during pregnancy
- preterm birth
- sex (including sex-stratified sensitivity analyses)

## Analytical framework

The analysis is organized around two complementary model types:

### 1. Full-curve developmental model
A negative binomial generalized additive model (GAM) is used to estimate the nonlinear developmental trajectory of SUID incidence across PCA.

This model is used to:
- characterize the overall rise, peak, and decline structure
- evaluate whether exposures shift the trajectory upward across development
- provide fitted developmental curves

### 2. Confirmatory windowed models
Piecewise negative binomial regression models are fit separately in an early rising interval and a later declining interval.

These models are used to estimate incidence rate ratios for:

- PCA
- bed sharing
- maternal smoking
- prematurity

The peak interval is summarized descriptively rather than treated as a stable inferential segment.

## Folder purpose

This folder is provided so readers and reviewers can inspect:

- case preprocessing logic
- exposure construction logic
- denominator and offset logic
- developmental modeling code
- manuscript-linked figure generation

This is an archival manuscript companion, not a maintained software tool.

## Reproducibility notes

Because the source data are not bundled here, execution may require:

- access to NCFRP case data
- access to external denominator data sources
- recreation of manuscript-specific intermediate files
- local adjustment of file paths and environment settings

In particular, reproducing the bed-sharing denominator component may require reconstruction of the exact PRAMS-derived prevalence logic used in the manuscript workflow.

## Expected contents

Typical contents may include:

- `src/` for modeling and utility functions
- `scripts/` for preprocessing and model execution
- `figures/` for generated or regenerable outputs
- `environment.yml` or similar environment specification

You may adapt the exact file descriptions to match the final release structure.

## Summary of findings addressed by this code

At a high level, this analysis supports the interpretation that when SUID incidence is aligned by PCA, developmental structure becomes clearer, with a rise, peak, and decline pattern across infancy. Within that framework, bed sharing and maternal smoking are associated with elevated incidence across much of the trajectory, while prematurity appears to contribute more strongly in the earlier developmental interval.

## Maintenance status

This folder is archived for manuscript transparency. Ongoing support is not planned.
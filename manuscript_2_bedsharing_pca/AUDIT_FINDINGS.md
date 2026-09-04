# Manuscript 2 pipeline audit — findings

Branch `audit/manuscript-2`. Living document. Each finding: evidence, the Manuscript 1
reference behaviour it is measured against, severity, status, and the fix.

Status legend: **OPEN** (not started) · **STATIC-FIX** (fixable now without data) ·
**[HPC]** (needs a run or the NCFRP codebook to confirm/resolve) · **DECISION** (needs a
user call, tracked as a gate G1–G9 in the plan) · **DONE**.

### Static-fix pass 1 (2026-09-02, branch `audit/manuscript-2`)

Applied on the branch, verified by `py_compile` + a synthetic smoke test spec
(`tests/smoke_synthetic.py`); not yet run against real data.

- `src/ncfrp_manuscript_v5.py` rewritten: single audited loader with a CONSORT
  `AuditTrail` (A9); sentinel exclusion + real post-term cap per G3 (A2.10);
  models fit on raw integer counts, pre-fit smoothing removed per G4 (A3.21);
  prespecified `bed_share x smoke` / `bed_share x preterm` interaction terms
  (A3.23); inclusive `peak3` defaulting to `PCA_MAX` (A3.25); one saved IRR CSV
  per model that Figure 4 reads, plus a CI-brackets-estimate assertion (A4.30);
  FEMALES/MALES runs restored (A4.31); Figure 3 TIFF format + Figure 4 tight
  bbox + Figure 2 shared y-axis + Figure 3B redrawn as a fitted rate-ratio
  curve (A4.32, Ed E7/E10/E11); seed + optional CV of the GAM penalty (A1.5);
  docstring/import cleanup (A1.6); `two_population_Ns.csv` emitted (A3.18).
- `src/preprocess_NCFRP.py`: unused `import selection` removed with a note
  (A1.1); label-safe vectorized replacements for the `.iloc[label]` loops
  (A2.12); `to_csv(..., index=False)` (A2.15); dead `date_min` removed (A2.16);
  `INFagecat` all-units decode (A2.13b); trimester cigarette trigger `> 1 -> > 0`
  plus the `*unk` flags (A2.14c). New finding **A2.14b**: the old smoking truth
  table defaulted every UNhandled `{befpreg, anytime}` code combination to
  "smoker"; the vectorized version defaults those to "unknown" (excluded).

### Codebook pass (T1, 2026-09-02)

The user supplied `Codebook_Version5-1_031122.xlsx` rows. Resolved:
- `INF3gestage` / `INFage` "default to blank" - unknown is MISSING, not a
  numeric sentinel. Loader now: drop NaN + `inf3gestunk == 1`; **exclude**
  implausible values (`inf3gestage > 47` or `< 20`, `infagedays >= 365`) instead
  of recoding; cap only genuine post-term `> 42 -> 42`. Param names changed
  `gestage_sentinel -> gestage_max`, `infage_sentinel_days -> max_infage_days`.
- `CIRsleepplace 9 = Carseat`, `99 = Unknown`. Pass 1 had wrongly removed `9`
  from the "solitary" set; **reverted** - the set `{1,2,5,9,10,13,14,15,16,17}`
  is correct (A2.11).
- `INFagecat` 1..5 = Years/Months/Days/Hours/Minutes; `age_decoder` now maps all
  five (A2.13b).
- `INF3*tricig` = number of cigarettes; `> 1` trigger was wrong (missed
  one-cigarette smokers) -> `> 0` + `INF3*tricigunk == 1` (A2.14c).
- New `src/pca_boundary_report_ncfrp.py` (ported from M1's
  `pca_boundary_report`) so the PCA window can be set from a count-stability
  rule (A3.17). New `requirements.txt` (A1.3), `run_all.sh` + `RUNBOOK.md`
  (A1.4), `tests/smoke_synthetic.py`.
- A3.19/A3.20 reframed: the marginal-product offset is **inherited from
  manuscript 1** (`run_windowed_nb_models.py` does the same), so it is not a
  M2 coding regression. The extreme smoking IRR is a symptom of the case-only
  NCFRP numerator over a national marginal denominator (A3.20).

### Manuscript-support scripts pass (milestone 5, 2026-09-02)

New, `py_compile`-clean, ready for the HPC run:
- `src/generate_table1.py` (A4.29 / R1.8 / Ed E14) - Table 1 by bed-sharing
  status on the pre-peak cohort; ported from M1's `table1_by_sex`.
- `src/generate_figure1_flow.py` (A4.28 / R1.4 / Ed E4) - CONSORT Figure 1 from
  `ncfrp_flow_audit.csv` + `two_population_Ns.csv`; text + PNG/TIFF.
- `src/compare_time_axes.py` (A6.40-42, A3.26, Ed E8/E9/E13/E15) - PCA vs
  chronological-age GAM fit comparison; PCA~CA scatter + correlation; proportion
  preterm by age on both axes.
- `risk_exposure_joint.py` now warns if `--denominator` looks like a matched
  set (< 1M rows) rather than the full live-birth file; docstring spells out
  that M1's birthweight matching does not touch the M2 offsets.
- `run_all.sh` / `RUNBOOK.md` extended to call all three.

### Recovery + joint-offset pass (T2-T5, 2026-09-02)

- **T2**: the two `preprocess_NCFRP.py` copies (`scripts/` vs committed `src/`)
  are byte-identical. No reconciliation needed.
- **T3 / A1.1**: no cause-of-death case filter exists in the pipeline; the
  `SeattleChildrensNCFRP.sav` export is the case series. `selection.py` is
  column lists only. G6 (SIDS vs SUID wording) still needs a user answer on how
  the export was scoped.
- **T4**: `selection.py` recovered into `src/`; the LR_BW step recovered as
  `src/lr_bw_impute.py` (birthweight imputation - not used by the incidence
  models; for Table 1). Both wired into `run_all.sh` / `RUNBOOK.md`.
- **T5 / A3.19 / G2**: `src/risk_exposure_joint.py` - joint `P(smoke, preterm)`
  from CDC live births + marginal bed-share from PRAMS, reusing M1's
  `compute_exposure` / `compute_preterm` / `compute_bed_share`.
  `ncfrp_manuscript_v5.py` gains `--offset-mode {joint,marginal}` (default
  joint); `run_all.sh` runs both for a sensitivity comparison.
- `ncfrp_manuscript_v5.py` also gains an explicit 2008-2019 death-year exclusion
  step (`--year-min` / `--year-max`, on `infdodyear` if present) so the flow N
  matches Figure 1 regardless of the base file (T3).

Reference implementation for every "M1 does X" claim below is the corrected Manuscript 1
code in `../manuscript_1_sex_pca/src/` (`audit_subject_breakdown.py`,
`run_windowed_nb_models.py`, `generate_figure{2,3}_*.py`, `generate_table1.py`,
`risk_exposure_fixed.py`), i.e. the versions that went back to AJE in the revision.

Files audited: `src/preprocess_NCFRP.py` (829 lines), `src/ncfrp_manuscript_v5.py`
(1595 lines). No raw data available locally; the analysis input
`standardized/NCFRP_SCH_LR_BW_pop.csv`, the `output_data/risk_exposure/*.pkl` offsets, and
the raw NCFRP `.sav` were not present to run against.

---

## Summary of the most consequential findings

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| A19/A20 | IRRs are built by dividing NCFRP case counts by a denominator faked from the **product of national marginal exposure prevalences**; there is no actual population/non-case sample. Smoking IRR lands at 13–18 (M1: ~4.5–7; literature ~2–4). | Critical | DECISION (G2) + [HPC] |
| A1 | The **case definition itself is not in the committed code** (SIDS/SUID restriction lives in a missing `selection` module or a missing standardization step). Cohort cannot be reproduced or described as-is. | Critical | [HPC] |
| A21 | `ncfrp_manuscript_v5.py` fits the windowed NB **and** the GAM on **rolling-mean-smoothed non-integer counts**. M1 fits on raw integer counts; smoothing is display-only. | High | STATIC-FIX (pending G4) |
| A10 | Unknown/implausible gestational age is **recoded to 40 ("term")**, not excluded, before PCA and preterm are derived. M1 excludes the `>=99` sentinel before computing PCA. | High | STATIC-FIX + [HPC] |
| A17 | PCA window hardcoded 36–75. This is the pre-revision M1 setup that reviewer 1 rejected (R1.3). M1's fix was a prespecified count-stability rule and a move to 90. | High | [HPC] |
| A23 | Models are main-effects only. M1 carries two prespecified interaction terms and tabulates them (R1.7). The paper's headline "bed sharing concentrates risk earlier" is an untested interaction. | High | [HPC] |
| A28 | `Figure_1.docx` flow does not reconcile (steps out of order, a number repeated with two different percentages, non-monotonic "Remaining"). Not generated from the pipeline. | High | STATIC-FIX + [HPC] |
| A9 | The loader `main()` actually uses (`load_ncfrp_data`) relies on NaN-poisoning + `dropna` with **no per-step audit trail**; the audit-capable loader (`load_ncfrp_data_with_audit`) is **commented out**. | Medium | STATIC-FIX |
| A31 | Sex-stratified runs are **commented out** in `main()`, yet v6 reports sex-stratified IRRs and the repo ships `Figure*_{MALES,FEMALES}.png`. | Medium | STATIC-FIX |
| A11 | Unknown bed sharing is reclassified **only toward "non-bed-sharing"**, and the trigger set includes an unknown sleep-place code. Directional misclassification that biases the bed-sharing IRR. | Medium | STATIC-FIX + [HPC] |

---

## A1 — Reproducibility and provenance

### A1.1 Case definition — **RESOLVED (2026-09-02), G6 confirmation pending**
Investigated `/Users/rerazo/Projects/manuscript_2/scripts/` (the older 4-aim project the user
added). Findings:
- `selection.py` is **only column-name lists** (`all_data`, `no_cig`, `multnum`, `Eds_list`);
  it has no case filter. Recovered as `src/selection.py`.
- **No script anywhere subsets rows to a cause of death.** `preprocess_NCFRP.py` and `sids.py`
  both just `read_sav('SeattleChildrensNCFRP.sav')` and recode. The `Eds_Code == 'W75'` /
  `CAUicdcode` blocks are all commented out (an abandoned attempt to *classify bed sharing*
  from ICD codes, not to select cases).
So the **`.sav` export itself is the case series** (NCFRP fatality-review system). The audited
loader's data-quality steps + first-year-of-life + PCA window define the analytic cohort.
**G6 RESOLVED (2026-09-02):** the user confirms every NCFRP case in this export is a SUID -
each was investigated and the cause was undetermined or accidental suffocation. So the case
definition is "all infant deaths in the NCFRP Case Reporting System export" and the outcome
is **SUID**, not SIDS. Manuscript action: v6 Methods says "classified as sudden infant death
syndrome" - change to SUID with a one-line definition ("sudden unexpected infant deaths
reviewed in the NCFRP CRS: SIDS/undetermined plus accidental suffocation and strangulation in
bed"), and make the title/abstract consistent (A5.34).
Verified separately: `NCFRP_SCH_pop.csv` is (24,314 rows x 167 cols), `INFdodyear` from 2004
on - it is NOT year-restricted, and the archived analysis read it, so the loader's year filter
is off by default (reproduces the 24,314 start). The loader's expected columns
(`INFageDays`, `INF3gestage`, `INF3gestunk`, `Smoker`, `INFsex`, `Bed_sharing`, `INFdodyear`,
`INF3*tricigunk`) all appear in that header.

### A1.2 Missing pipeline pieces — **Mostly RESOLVED (2026-09-02)**
- `selection` module -> recovered as `src/selection.py` (column lists only).
- The `NCFRP_SCH_pop.csv -> NCFRP_SCH_LR_BW_pop.csv` step -> recovered as
  `src/lr_bw_impute.py` (it is a **birthweight imputation**, `INF3biwt ~ INF3gestage + INFsex`;
  not a denominator step). The incidence models don't use birthweight, so it only matters for
  a complete Table 1.
- Exposure-pickle builder -> written as `src/risk_exposure_joint.py`, reusing M1's
  `compute_exposure` / `compute_preterm` / `compute_bed_share`. Produces the joint
  `joint_smoke_preterm_exposure.pkl` (G2) + the four marginal pkls + `.csv` copies +
  `joint_exposure_summary.txt`. Set a seed; writes prevalences to CSV for inspection.
- `denominator.parquet` / `prams.parquet` -> still HPC-only; built by M1's
  `format_standardization.py` + `concat_files.py`. RUNBOOK step 5 points `risk_exposure_joint.py`
  at them.

### A1.3 Environment not pinned — **Medium, STATIC-FIX**
`environment-minimal.yml` is named `py310` but pins `python==3.8.12`; `mamba-info.txt` reports
`python 3.10.14`; `forestplot`, `pyreadstat`, `openpyxl`, `seaborn` are imported
(`ncfrp_manuscript_v5.py:29`, `preprocess_NCFRP.py:33,35,40`) but not listed in
`environment-minimal.yml`. M1 added a `manuscript_1_sex_pca/requirements.txt`.
Fix: add `manuscript_2_bedsharing_pca/requirements.txt` matching the real runtime; record the
Python version used for the final run in `RUN_MANIFEST.md`.

### A1.4 No master runner / documented order — **Medium, STATIC-FIX**
Both READMEs describe a `scripts/` dir that does not exist. No Makefile / run_all. Execution
order is only implicit.
Fix: add `run_all.sh` (or Makefile) + `RUNBOOK.md` with the real order: raw `.sav` ->
`preprocess_NCFRP.py` -> standardization/`LR_BW` -> exposure-pkl builder ->
`ncfrp_manuscript_v5.py`.

### A1.5 Non-determinism — **Medium, STATIC-FIX (pending decision)**
`fit_nb_gam` hardcodes `family = NB(alpha=0.10)` (`ncfrp_manuscript_v5.py:1201`) — a fixed
dispersion with no cross-validation. M1's `generate_figure3_gam_curves.py` starts from
`--nb-alpha 0.10` but then runs `GLMGam.select_penweight_kfold` over a logspace alpha grid ×
`--cv-folds` and refits with the CV-selected penalty (M1 lines 239–245). No global seed is
set anywhere in either M2 script.
Fix: adopt M1's CV penalty selection with a fixed seed for the folds, or state and justify a
fixed alpha in Methods. Set `np.random.seed(...)` at the top of `main()`.

### A1.6 Script / version drift — **Low, STATIC-FIX**
`ncfrp_manuscript_v5.py:1` docstring says `ncfrp_manuscript_v4.py`. `import numpy as np` /
`import pandas as pd` appear at module top (26–27) and again at line 89. Two loaders with
divergent logic (`load_ncfrp_data_with_audit` line 93, `load_ncfrp_data` line 319).
`preprocess_NCFRP.py` header dates itself "Fall 2023".
Fix: correct the docstring, dedupe imports, collapse to one loader (A9).

### A1.7 Absolute / CWD-bound paths — **Low, STATIC-FIX**
`figures/figure_captions.md` hardcodes `/Volumes/ramirezlab/ricardo/ncfrp/Figure2_ALL.png`
etc. `ncfrp_manuscript_v5.py` module constants (47–54) and `preprocess_NCFRP.py` (42–43,
48, 801–802) assume a fixed CWD. M1's corrected scripts take every path as a CLI arg.
Fix: parameterize via `argparse`; strip the `/Volumes/...` paths from the captions.

---

## A2 — Case selection and preprocessing

### A2.8 First-postnatal-week exclusion wording vs code — **Low/Medium, DECISION (G5)**
`ncfrp_manuscript_v5.py:209` and `:331` exclude `infageweeks <= 1` where
`infageweeks = floor(infagedays / 7)`, i.e. days 0–13. M1 does the same (`infage <= 1` week,
`audit_subject_breakdown.py:206`), and M1 passed review, so this is **consistent with the
standard**. The issue is purely descriptive: v6 and Ed (E5) read it as "first postnatal week"
/ "<7 days". Text and code must state the same thing.
Fix: after G5, describe exactly what the code excludes (e.g. "deaths before completed
postnatal week 2") and keep it identical to M1's convention.

### A2.9 No audit trail on the loader `main()` uses — **Medium, STATIC-FIX**
`main()` calls `load_ncfrp_data()` (`ncfrp_manuscript_v5.py:1571`), which poisons values to
`np.nan` (`:331`, `:338`, `:343-344`) and relies on a terminal `dropna(subset=keep_cols)`
(`:392`). It works only incidentally (because `pca` is recomputed at `:341` after the NaN
writes and is in `keep_cols`). It prints step shapes but records nothing. The audit-capable
`load_ncfrp_data_with_audit()` — which does explicit sequential drops and builds an audit
DataFrame — is **commented out** in `main()` (`:1585-1592`). M1's standard is an explicit
`AuditTrail.record()` after every exclusion step (`audit_subject_breakdown.py:88-128`).
Fix: make `load_ncfrp_data_with_audit` the single loader, wire it into `main()`, and have it
emit `output_data/ncfrp_flow_audit.csv` on every run.

### A2.10 Unknown/implausible gestational age recoded to "term", not excluded — **High, DONE (pass 1) + [HPC]**
The archived `df.loc[df["inf3gestage"] > 42, "inf3gestage"] = 40` ran before PCA and preterm
were derived. **Codebook (T1): `INF3gestage` "defaults to blank"** — unknown gestational age
is MISSING, not a numeric sentinel — and the codebook lists an `INF3gestunk` flag. So the
real risk here was a *decode error* (or a stray large value) being silently rewritten to a
valid term `40` and entering the cohort, corrupting PCA and the preterm flag — the same class
as M1's `combgest >= 99` guard.
Fixed in `ncfrp_manuscript_v5.py::load_ncfrp_data`: drop `inf3gestunk == 1` (if present) and
NaN gestational age; **exclude** `inf3gestage > GESTAGE_MAX (47)` or `< GESTAGE_MIN (20)` as
implausible (not recode); cap only genuine post-term `> 42 -> 42` (G3). Same treatment for
infant age: NaN dropped, `infagedays >= MAX_INFAGE_DAYS (365)` excluded as implausible.
Remaining [HPC]: confirm `INF3gestunk` survives into the standardized CSV; sanity-check the
excluded counts.

### A2.11 Directional reclassification of unknown bed sharing — **Medium, DONE (pass 1) + [HPC]**
`preprocess_NCFRP.py`: rows with `Bed_sharing == 9` (unknown same-surface) are set to `2`
(non-bed-sharing) when the incident sleep place is a single-occupant device.
**Correction to an earlier version of this finding:** the "solitary" set
`{1,2,5,9,10,13,14,15,16,17}` is correct — **codebook (T1): `CIRsleepplace 9 = Carseat`, and
`99 = Unknown`** (99 is not in the set). Pass 1 had wrongly removed `9`; it is restored.
The set legitimately excludes the surfaces where bed-sharing is plausible (3 adult bed,
4 waterbed, 6 couch, 7 chair, 8 floor, 11 other, 12 futon) and the unknowns (0, 99).
Residual concern: the reclassification is still **directional** — unknown `Bed_sharing` only
ever moves toward the non-exposed reference group, never toward exposed — which biases the
bed-sharing IRR upward. M1's `risk_exposure_fixed.py:80-84` flags the mirror-image on the
PRAMS denominator side (non-response -> bedshare=1), biasing the IRR down; the two do not
cancel.
Fix (pending [HPC]): `ncfrp_manuscript_v5.py` should run a sensitivity analysis that leaves
`Bed_sharing == 9` as missing (drop those cases) and report the reclassified count.

### A2.12 `.iloc[i]` indexed by label — **Medium, STATIC-FIX**
`preprocess_NCFRP.py` uses `for i in df.index: subject = df.iloc[i]` in the smoking loop
(`:231-244`), `GIVs_drugs` (`:372-388`), `GIVs_edu` (`:397-400`), and the `Bed_sharing` loop
(`:690-697`); also `df.loc[np.nonzero(df['INFageDays']<30)[0], ...]` (`:107`). `.iloc` is
positional but `df.index` yields labels and `np.nonzero` yields positions — these agree only
while the index is a clean `RangeIndex` and no prior filter has dropped rows. `read_sav`
returns a RangeIndex so it currently works, but it is fragile to any reordering.
Fix: convert to label-safe vectorized operations; add a guard/assert that the index is a
RangeIndex at entry.

### A2.13 `INFageMonths` cascade redundancy — **Low, DONE (pass 1)**
The first `<30 -> 0` line was dead (immediately overwritten). Tidied; `np.nonzero`-as-labels
fixed to a boolean mask (A2.12).

### A2.13b `INFagecat` "Years" not decoded — **Low, DONE (pass 1)**
**Codebook (T1): `INFagecat` = unit of `INFage`** — 1 Years, 2 Months, 3 Days, 4 Hours,
5 Minutes, 0 blank. The archived `age_decoder` only handled 2 and 3, so any infant coded in
whole years (`INFagecat == 1`) got `INFageDays = NaN` and was dropped as "missing age" rather
than counted; hours/minutes were also dropped (harmless, they are first-hour deaths).
Fixed in `preprocess_NCFRP.py`: `age_decoder` maps all five units (1 -> 365, 4 -> 1/24,
5 -> 1/1440); blank stays undecoded -> NaN -> dropped honestly.

### A2.14 Smoking construction — **Partially DONE (pass 1) + [HPC]**
`Smoker` truth table vectorized; unhandled code combinations now default to unknown, not
"smoker" (A2.14b). `INF3smanytime = 1` trigger: **codebook (T1) says `INF3*tricig` is the
NUMBER of cigarettes** (blank when N/A), so the archived `> 1` missed one-cigarette smokers —
changed to `> 0`, and the paired `INF3*tricigunk == 1` ("smoked, quantity unknown") flags are
now folded in (A2.14c).
Remaining [HPC]: confirm `Smoker` code 9 is the only missing code for `INF3smbefpreg` /
`INF3smanytime`; check the excluded-smoking count is plausible.

### A2.15 `to_csv` writes the index — **Low, STATIC-FIX**
`preprocess_NCFRP.py:801-802` writes without `index=False`, so the analysis CSV gains an
unnamed index column that becomes `unnamed: 0` after `df.columns.str.lower()`.
Fix: `index=False` on both writes.

### A2.16 Year window undocumented — **Low, STATIC-FIX + [HPC]**
`select_sample` (`preprocess_NCFRP.py:67-75`) keeps `INFdodyear` 2008–2019; a `date_min=2000`
local is dead. `Figure_1.docx` says "2008 to 2019"; v6 Methods does not state the years.
Fix: remove the dead local; state the years in Methods; confirm 2019 is the last complete
NCFRP year available.

---

## A3 — Modeling and methodology

### A3.17 Fixed PCA window 36–75 — **High, [HPC]**
`ncfrp_manuscript_v5.py:53-54` (`PCA_MIN = 36`, `PCA_MAX = 75`). `Figure_1.docx` reports
"PCA > 75 weeks n = 606 (2.5%)" — the same asymmetric upper-tail exclusion that reviewer 1
challenged in M1 (R1.3: "either include more weeks PCA than 75, or give some other
justification"). M1's resolution: a prespecified count-stability rule (>=5 sex-specific deaths
per PCA week), which extended the upper bound to 90, plus a symmetric-trim comparison for
context; implemented as `pca_boundary_report()` in `audit_subject_breakdown.py:520`.
Fix: port `pca_boundary_report`'s logic to the NCFRP cohort (overall and sex-specific weekly
counts across the full observed PCA range), set the bounds where the threshold is met, and
produce the boundary-diagnostics table. Feeds the Figure 1 rebuild (A28).

### A3.18 Two-population accounting — **Medium, STATIC-FIX + [HPC]**
`run_piecewise_and_gam` (`ncfrp_manuscript_v5.py:1485`) feeds the same `input_df` to both the
GAM (full range, includes the 43–46 peak — correct) and the piecewise split (`peak1=43`,
`peak2=47` -> peak excluded — correct). This matches M1's design. What is missing is the
explicit accounting: the manuscript and Figure 1 must state total analytic N, early N, late N,
and peak-excluded N, as M1 was forced to add for R1.4.
Fix: emit all four Ns from the audited loader + `split_analytic_windows` equivalent; use them
in the Figure 1 rebuild and the memo.

### A3.19 Offset = product of marginal exposure prevalences — **DONE (pass 2), G2 = joint**
The archived `make_ncfrp_input` formed `exposure = P(smoke) * P(preterm) * P(bed_share)`, all
marginals, assuming population independence that the case data contradict (smoking x bed
sharing co-occur; smoking x preterm co-occur). M1 uses the same marginal-product form, so it
was inherited, not a new bug.
Implemented (G2): `src/risk_exposure_joint.py` builds the **joint** `P(smoke, preterm)` from
the CDC live-birth denominator; bed sharing stays a marginal from PRAMS (it does not link to
the birth file). `ncfrp_manuscript_v5.py --offset-mode joint` (the default) uses
`joint[(smoke, preterm)] * P(bed_share)`; `--offset-mode marginal` reproduces the old form for
a side-by-side sensitivity run (`run_all.sh` does both). The manuscript must state the
smoke-preterm joint source and the bed-sharing independence assumption. Whether this brings
the smoking IRR into a plausible range is for the HPC run to show (see A3.20 — the case-only
design is the deeper issue).

### A3.20 Numerator / denominator population mismatch — **Critical, DECISION (G2) + [HPC]**
NCFRP is a fatality-review case system covering only participating jurisdictions and years;
there is **no non-case sample and no matching population registry**. The offsets in A19 are
national marginal prevalences. So every "IRR" is NCFRP case odds reweighted by national
prevalence, not an incidence rate ratio in a defined population. Symptoms: full-curve smoking
`exp(β) = 13.35`, early-window smoking `IRR = 17.75` (v6 Results) versus M1's maternal-smoking
IRR ~4.5–7 (same modelling machinery, but a real linked population) and the case-control
literature's ~2–4. Figure 2 additionally prints absolute "per 100,000 live births" rates
(`build_observed_rate_table`, `ncfrp_manuscript_v5.py:761-813`) from NCFRP cases over national
births — not interpretable as a rate.
Fix: if an NCFRP-jurisdiction-restricted denominator is obtainable, use it; otherwise
reframe every "IRR" as a prevalence-reweighted case-composition ratio, relabel Figure 2 as a
relative/arbitrary scale, and state the design limitation prominently (v6 only alludes to it
in Limitations point 3).

### A3.21 NB and GAM fit on pre-smoothed non-integer counts — **High, STATIC-FIX (pending G4)**
`run_piecewise_and_gam` (`ncfrp_manuscript_v5.py:1501-1510`) calls
`smooth_stratified_counts(window=3, center=True)` then `apply_min_count_threshold`, and the
smoothed `count` column then flows into both `prepare_piecewise_data` ->
`NegativeBinomial(y1, ...)` (`:1521-1522`) and `prepare_curve_data_ncfrp` -> `fit_nb_gam`
(`:1532-1533`). Fitting a count likelihood to rolling-mean-averaged, non-integer data
understates variance and is hard to defend. M1 does **not** do this: `run_windowed_nb_models.py`
fits `NegativeBinomial` directly on raw `value_counts()` output; the only `.rolling(window=3)`
in M1 is on the **observed overlay curves in the Figure 3 display**
(`generate_figure3_gam_curves.py:329,344`).
Fix (G4): fit on raw integer counts; restrict smoothing to plotted curves. Also consolidate
the double `MIN_COUNT` filter (`count_ncfrp_data:454` and `apply_min_count_threshold:1510`).

### A3.22 `count >= 10` per-stratum threshold + no zero cells — **Medium, [HPC]**
`count_ncfrp_data` (`ncfrp_manuscript_v5.py:448-455`) builds strata from `value_counts()` (no
zero rows) and drops any stratum-week with `count < MIN_COUNT (10)`; `apply_min_count_threshold`
applies it again post-smoothing. M1's `count_data` also produces no zero rows but applies **no
minimum-count threshold** in `run_windowed_nb_models.py`. So M2 discards substantially more
low-information strata than the M1 standard, and v6 Limitations concedes the models "describe
observed high-information strata rather than a fully enumerated population lattice."
Fix: quantify how many strata-weeks and cases the threshold removes; lower or remove it to
match M1, or justify the value explicitly in Methods.

### A3.23 Main-effects-only models; the headline claim is an untested interaction — **High, [HPC]**
`prepare_piecewise_data` (`ncfrp_manuscript_v5.py:1069`) and `prepare_curve_data_ncfrp`
(`:1140-1145`) use `[pca, bed_share, smoke, preterm]` with no interaction terms. M1's
`prepare_data` builds `sex*preterm` and `smoke*preterm` and includes them in the design
(`run_windowed_nb_models.py:200-201,213`), and M1 had to tabulate every interaction
estimate/CI/p for R1.7. The v6 claim that bed sharing "concentrates risk more strongly in the
earlier portion of the vulnerable window" is a `bed_share × PCA` interaction claim currently
supported **only** by the GAM `pca_beddev` deviation smooth (A27), never by a tested
coefficient.
Fix: add prespecified interactions — `bed_share × PCA` (formally test the early-vs-late
bed-sharing IRR difference), `bed_share × smoke` (the co-occurrence the paper foregrounds),
and consider `bed_share × preterm` — and report all estimates/CIs/p in a coefficient table.

### A3.24 Case-only χ² described as a risk association — **Medium, STATIC-FIX (memo)**
`smoke_bedshare_case_association` (`ncfrp_manuscript_v5.py:503-509`) runs
`chi2_contingency` on a **cases-only** smoke × bed_share table. v6 Conclusions: "confirm a
significant positive association of SUID **risk** between bed-sharing and smoking during
pregnancy." A cases-only test shows co-occurrence among deaths, not risk — the exact
exposure-correlation-vs-outcome-interaction conflation R1.7 called out in M1.
Fix: reframe as co-occurrence among cases in the text; if a risk statement is wanted it needs
the population joint distribution from A19/G2.

### A3.25 Dead `peak3 = 100` truncation knob — **Low, STATIC-FIX**
`fit_piecewise_nb` default `peak3 = 100` (`ncfrp_manuscript_v5.py:1084`) and the call at
`:1519` pass `peak3 = 100` while data are capped at 75. Harmless now, but this is exactly the
"stale peak3 default silently re-truncates the late window" foot-gun M1's
`run_windowed_nb_models.py` docstring (lines 21–28) was written to remove. M1 also switched
the late-window upper bound to **inclusive** (`<= peak3`) to match cohort construction; M2
uses exclusive `< peak3` (`:1064`).
Fix: default `peak3` to `PCA_MAX`, make the late-window upper bound inclusive, matching M1.

### A3.26 "Prematurity contributes earlier" claim — **High, [HPC]**
v6 Discussion: prematurity "contributed most strongly during the early rising phase and then
weakened substantially later" (late IRR 0.90, p = .063; full-curve `exp(β) = 1.04`, ns). This
is the same shape as the M1 Discussion sentence ("prematurity attenuated and reversed") that
**did not survive M1's own corrected pipeline** and was removed. Ed's independent CDC Wonder
check (E13) points the other way. Do not assert this until it survives the corrected pipeline
(fixed sentinels A10, offsets G2, raw counts A21, window rule A17).
Fix: add a descriptive "proportion of deaths preterm by PCA week and by chronological-age
week" plot; be prepared to drop or reverse the claim and cite what others report.

### A3.27 Non-standard GAM deviation smooth carries the main visual claim — **DONE (2026-09-04)**
`prepare_curve_data_ncfrp` (`ncfrp_manuscript_v5.py:865-890`) builds
`pca_beddev = where(bed_share == 1, pca, median(pca))` and fits a second B-spline on it as a
"bed-share deviation" smooth. This is the mechanism behind the "bed-sharing trajectory peaks
earlier and more sharply" statement. It is not a standard interaction spline and its knot
placement (`make_knots_from_data`, uniform `step=6`) differs from M1's peak-anchored knots
(`generate_figure3_gam_curves.py:167-178`).
Fix: add a standard `bed_share`-by-PCA tensor / interaction-spline formulation as a robustness
check; report knot placement, `df`, and the fixed vs CV alpha choice in Methods.

**Status update:** the "standard interaction-spline" fix described above was implemented
(`gam_res_int`, `run_piecewise_and_gam` lines ~1076-1095 of an earlier version of this file)
but was never verified before being wired into the pipeline. It was **numerically degenerate**:
under the primary joint-offset design, every stratum's `*_gam_bedshareXpca_exp.csv` had
`bed_share` and several high-order spline terms with matching std errors ≈329 and IRR/CI
spanning ~560 orders of magnitude (e.g. `out_joint/output_data/ncfrp/all_gam_bedshareXpca_exp.csv`:
`bed_share` coef=-6.29 SE=329.2, `x1_s6..x1_s14` all SE≈329). Root cause, confirmed by tracing
the code: that variant built its interaction smooth from the raw product `xs_int = pca *
bed_share` (`:1082-1083` of that version) — exactly 0 for every non-bed-sharing row — and fed
it into `make_knots_from_data` (knots from `min`/`max` of that column), which placed knots in a
PCA range (~6-30) with **zero real support** (no bed_share=1 row has PCA < 36; every bed_share=0
row contributed only the repeated value 0). A second, deeper issue: `BSplines()` expands
whatever column it is given as an independent scalar covariate — it has no "by" mechanism, so
multiplying by `bed_share` **before** basis expansion (that code) is not the same operation as
multiplying **after** expansion (the standard varying-coefficient / mgcv `s(x, by=g)`
construction), and only the latter is identifiable against the parametric `bed_share` term.
`save_gam_model_outputs` never asserted CI-brackets-estimate (see A3.28), so this reached disk
with zero warning.

**Fix applied:** `gam_res_int` was retired from `run_piecewise_and_gam` (it no longer runs or
writes `*_gam_bedshareXpca_*` files). Its corrected replacement is a **standalone script**,
`src/gam_bedshare_pca_varycoef.py`, fit as a plain NB MLE
(`statsmodels.discrete.discrete_model.NegativeBinomial`, the same optimizer already used and
trusted for the early/late piecewise models) rather than through `GLMGam` — this sidesteps the
penalized-IRLS machinery that was the proximate numerical failure mode. It builds the
interaction smooth correctly: knots from the bed_share==1 **subset's** real PCA quantiles
(fixes the knot-range bug), the basis evaluated at every row's real PCA value (no zero
placeholder), centered on the bed_share==1 subset's column means (the standard sum-to-zero
identifiability constraint), and only **then** multiplied by `bed_share` — "expand, then
multiply," not "multiply, then expand." (The main `s(pca)` smooth needed the same centering
treatment for an unrelated reason: patsy's `cr()` basis is a partition of unity, so it is
perfectly collinear with a separate parametric Intercept column unless centered — caught by
`check_design_rank`, see A3.28, while developing the fix.) See
`gam_bedshare_pca_varycoef.py`'s module docstring for the full construction and the
alternatives considered (a corrected `GLMGam`, mgcv via rpy2, pyGAM) and why a fixed-df NB MLE
was chosen instead. Verified on a synthetic fixture with a known injected varying-coefficient
effect (`tests/smoke_synthetic.py::test_varycoef_replaces_degenerate_construction`) that (a) the
OLD construction still degenerates on that fixture (interaction-basis terms collapse to ~0 —
the same rank-deficiency, manifesting as collapse-to-zero rather than explode-to-infinity at
this smaller scale) and (b) the NEW construction recovers sane, bounded estimates (max std err
well under 1 on this fixture vs the real run's ~329) and the correct early>late shape of the
injected effect. **Not yet run against real data** (HPC-only) — see `RUNBOOK.md` "9b" for how,
and check the numeric thresholds there before trusting the output for the manuscript. Residual
caveat once run for real: even a corrected fit's high-PCA tail for bed_share=1 will have thin
support (`two_population_Ns.csv`); treat the far right end of the curve/CI as illustrative, not
precise, and prefer truncating the reported domain over forcing a falsely-confident curve if the
CI band still widens sharply there.

### A3.28 `save_gam_model_outputs` never checked CI-brackets-estimate — **DONE (2026-09-04)**
`save_nb_model_outputs` (`:572-583`) has asserted `IRR_ci_low <= IRR <= IRR_ci_high`
(`_assert_ci_brackets_estimate`, `:562-569`) before writing since A4.30. `save_gam_model_outputs`
(`:586-596`), used for every GAM fit including the degenerate `gam_res_int` variant described in
A3.27, had no such check — it wrote `{out_prefix}_exp.csv` unconditionally. This is exactly how
an IRR of 0.0018 with a CI spanning 560 orders of magnitude reached disk with no warning.
Fix: `save_gam_model_outputs` now runs the same `_assert_ci_brackets_estimate` check, plus a new
max-SE sanity guard (`SE_SANITY_MAX = 20.0` on the log scale) that raises with the offending
terms named if any coefficient's std error exceeds it — a CI can technically bracket its point
estimate and still be numerically meaningless, which the bracket check alone would not catch.
Also added: `check_design_rank(X, colnames, label, cond_threshold=1e8)`, a general-purpose
pre-fit collinearity/rank check (numpy `matrix_rank`/`cond`, naming suspect columns via a pivoted
QR on failure) — call it before any new NB/GAM fit in this pipeline. Both live near
`_assert_ci_brackets_estimate` in `ncfrp_manuscript_v5.py` and are used by
`gam_bedshare_pca_varycoef.py`.

---

## A4 — Figures, tables, internal consistency

### A4.28 Figure 1 does not reconcile — **DONE (scaffolded), needs the run**
`src/generate_figure1_flow.py` builds it from `ncfrp_flow_audit.csv` +
`two_population_Ns.csv`. Original problem below for the record.

`../manuscript_2/Figure_1.docx` flow (from the extracted text): steps appear out of order
("Start: All SUID cases 2008 to 2019 N = 24,314" appears *after* the sex/bed-sharing
exclusions), `n = 7,816` is listed once as "6.3%" (invalid bed sharing) and once as "32.1%"
(invalid smoker), "Remaining" values are non-monotonic (15,139 then 23,746 then 15,162), and
the final N (13,597) disagrees with the figure captions (13,915). It was not generated from
the pipeline.
Fix: generate Figure 1 programmatically from the audited loader's step table (A9), following
M1's `figure1_reconciliation` pattern (`audit_subject_breakdown.py:360`), so every "excluded n"
and "remaining N" adds up and ends at the analytic N; show the descriptive/GAM cohort and the
early/late/peak windowed branches with their Ns.

### A4.29 No Table 1 — **DONE (scaffolded), needs the run**
`src/generate_table1.py` builds "Characteristics of the analytic SUID cohort by bed-sharing
status" on the pre-peak cohort. Context below.

No Table 1 exists anywhere in `../manuscript_2/`. This is the single most concrete thing
reviewer 1 asked M1 for (R1.8). M1's `generate_table1.py` builds it on the cohort **after**
`clean_case_series` + `exclude_invalid_smoking` but **before** the peak-interval split
(the N=23,686 population), stratified by the primary exposure.
Fix: port `table1_by_sex` to `table1_by_bedshare` — counts + column % by cause-of-death
category, postnatal age at death, PCA window, gestational-age category, maternal smoking, sex,
year band, stratified by bed-sharing status, on the pre-peak-split cohort. Also serves Ed E14.

### A4.30 No single source of numbers — **Medium, STATIC-FIX**
`plot_nb_forest` re-derives IRRs from the fitted model objects (`extract_nb_irr_table`,
`ncfrp_manuscript_v5.py:912`) and `save_nb_model_outputs` writes separate CSVs; nothing
guarantees the forest plot, any coefficient table, and the in-text numbers agree. M1's fix
after R1.16/R1.17 was to have Figure 4 and the supplemental tables read the **same saved IRR
CSVs** and to add an automated "every CI brackets its estimate" check.
Fix: write IRR/coefficient CSVs once from the fit; have every figure and table read those;
add the CI-bracketing assertion.

### A4.31 Sex-stratified runs disabled — **Medium, STATIC-FIX**
`main()` comments out the FEMALES and MALES runs (`ncfrp_manuscript_v5.py:1580-1583`) and the
audit-loader call (`:1585-1592`), yet v6 reports sex-stratified IRRs in detail and the repo
ships `Figure2/3/4_{FEMALES,MALES}.png`. The shipped stratified figures came from an earlier,
unversioned run.
Fix: re-enable all three runs; note that for the sex-stratified runs the sex prevalence must
be handled correctly in the offset (`make_ncfrp_input` comment at `:468-471` says sex is
deliberately omitted for the ALL run — confirm the stratified runs need it added or that the
case_df restriction already accounts for it).

### A4.32 Figure export issues — **Medium, STATIC-FIX**
- 32a: `plot_gam_curve` writes `Figure3_*.tiff` with `fig.savefig(..., dpi=300)` and no
  `format="tiff"` (`ncfrp_manuscript_v5.py:1477`), unlike Figure 2/4 which pass
  `format="tiff"`. Confirm the extension is honoured; standardize 300-dpi PNG+TIFF across all.
- 32b (Ed E11): Figure 4 exports clipped ("missing the margins"). `plot_nb_forest` uses
  `fig.subplots_adjust(left=0.42, ...)` plus `forestplot` drawn into a shared axis and
  figure-fraction panel labels (`:1016-1047`) — prone to cropping the left variable labels.
  Switch to `bbox_inches="tight"` / `constrained_layout`; verify all labels render.
- 32c (Ed E10): Figure 3B ("bed-sharing minus non-sharing count", `:1408-1465`) is
  uninterpretable to Ed. Its CI band is a crude `lo_bs - hi_ns` / `hi_bs - lo_ns`
  (`:1421-1422`). M1's analogous panel plots a fitted difference with a proper delta-method /
  prediction CI and a rolling-smoothed observed overlay
  (`generate_figure3_gam_curves.py:294-348`). Redesign to match, or replace with a
  rate-ratio-vs-PCA curve.

### A4.33 Caption problems — **Low/Medium, STATIC-FIX**
`figures/figure_captions.md`: `/Volumes/...` absolute image paths (`:12,24,35`); the Figure 4
caption body repeats the Figure 3 sentence "Observed and predicted SUID case counts by sex and
PCA" (`:37`) and the panel is labelled "Figure 3A" then "Figure 4"; Figure 1 caption is
HTML-commented-out and says N = 13,915. Figure 1 is never cited in the v6 Results text (Ed E4)
and its flow is not described in Methods.
Fix: rewrite captions from the reconciled run; add the Figure 1 in-text citation and a Methods
sentence describing the flow (memo item for the author).

---

## A5 — Terminology and reporting hygiene (for the reviewer-proofing memo)

- A5.34 (G6 RESOLVED, Ed E3): outcome is **SUID**, not SIDS (the user confirms every case in
  the NCFRP export is a SUID: investigated, cause undetermined or accidental suffocation).
  v6 Methods "classified as sudden infant death syndrome" -> "sudden unexpected infant death
  (SUID)"; add the one-line definition; make the title/abstract consistent (they already say
  SUID). Memo item for the author.
- A5.35: "post-neonatal mortality" in the v6 Introduction — the term reviewer 1 rejected
  (R1.9) for a cohort starting after the first postnatal week. Use a direct description.
- A5.36: `[insert paper 1 citation]` placeholder in v6 Results — wire in the Manuscript 1
  reference.
- A5.37 (R2.1): v6 says incidence peaks "at approximately 2 to 4 months" (Intro) and
  "mid-40-week PCA range" (later). State which axis each refers to; consider the peak-band
  postnatal-age check M1 did (`analyze_peak_postnatal_age.py`).
- A5.38: no funding / acknowledgements / data & code availability / AI-use / IRB statements in
  v6. Reviewer 1 explicitly complained about a missing funding source in M1 (R1.13).
- A5.39 (Ed E6): the vaping sentence cites refs [12–15]; only ref 15 (Mitchell 2026) is about
  vaping. Fix the citation set or narrow the claim. Also check single-citation density for
  established associations against R1.15.

---

## A6 — New analyses requested by Ed Mitchell — **DONE (scaffolded in `src/compare_time_axes.py` + `src/generate_table1.py`), need the run**

- A6.40 (Ed E9): `compare_time_axes.py` fits the NB spline GAM with `pca` and again with
  `infageweeks` as the time axis and writes `gam_fit_comparison.csv` (deviance / AIC / llf /
  pseudo-R2) + `observed_by_axis.{png,tiff}`.
- A6.41 (Ed E15): `compare_time_axes.py` writes `pca_vs_chronological_correlation.txt`
  (Pearson + Spearman) and `pca_vs_chronological_scatter.png`.
- A6.42 (Ed E14): covered by `generate_table1.py` (preterm / smoking / bed-sharing counts and
  the gestational-age category rows). Original note:
  category counts (preterm / smoke-exposed / bed-sharing, n and %) and the
  gestational-age distribution as a cohort sanity check (expect overall mode ~39 wk, preterm
  mode ~36 wk, very-preterm a small fraction). Largely Table 1 (A29) but call it out
  explicitly.
- A6.43 (Ed E8, G9): the "environmental vs developmental" framing challenge is a scientific
  call for the author; support it with A40, A41, and the `bed_share × PCA` interaction (A23).

---

## Decision gates (mirrors the plan; the user's call)

- **G1** OPEN — retire `/Users/rerazo/Projects/repo/` in favour of the clone, or keep it synced?
- **G2** RESOLVED (2026-09-02): **build a joint population exposure distribution where the data
  allow** (cross-tab smoke × preterm from CDC live births; PRAMS for bed sharing; explicit
  independence assumption only where unavoidable). Expect headline IRRs to move toward the
  literature range.
- **G3** RESOLVED (2026-09-02) and implemented: codebook (T1) shows unknown gestational/infant
  age is BLANK, not a sentinel. Loader drops NaN + `inf3gestunk == 1`, **excludes** implausible
  values, caps only genuine post-term `> 42 -> 42`. Done.
- **G4** RESOLVED (2026-09-02): **fit the NB and GAM models on raw integer counts**; rolling
  mean is display-only, matching M1.
- **G5** OPEN — first-week exclusion described as days 0–6 or as (M1-consistent) days 0–13?
- **G6** RESOLVED (2026-09-02): **hold** until the actual case-selection code is recovered from
  the HPC (A1.1); then align title/abstract/Methods to whatever it selects.
- **G7** OPEN — authorship / funding / IRB details for the front matter.
- **G8** OPEN — target journal (sets the word/abstract limits Ed E1 flags).
- **G9** OPEN — how to position "developmental" against Ed's "environmental hazard" reading.

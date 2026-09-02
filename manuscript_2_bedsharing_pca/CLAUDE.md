# manuscript_2_bedsharing_pca — working notes for Claude

Follow-up to Manuscript 1, targeting the American Journal of Epidemiology:
*"Developmental Timing of SUID and the Infant Sleep Environment."* Bed sharing is
the primary exposure; PCA (postconceptional age) is the developmental axis.

## State (branch `audit/manuscript-2`)

The code here has been audited against Manuscript 1's post-revision standard and
its referee reports, plus co-author Ed Mitchell's review of draft v6. Full
finding list: **`AUDIT_FINDINGS.md`**. How to run it: **`RUNBOOK.md`** and
`run_all.sh`. Nothing has been run against real data yet (no scientific-Python
env on the audit machine; verification so far is `py_compile` + `tests/smoke_synthetic.py`).

## Pipeline (see RUNBOOK.md for the full command lines)

```
preprocess_NCFRP.py        raw NCFRP .sav -> data/NCFRP_SCH_{sample,pop}.csv
lr_bw_impute.py            birthweight imputation -> standardized/NCFRP_SCH_LR_BW_pop.csv
                           (Table 1 completeness only; not a model covariate)
risk_exposure_joint.py     denominator.parquet + prams.parquet -> exposure offsets
                           (JOINT P(smoke,preterm) from CDC births x marginal P(bed_share))
pca_boundary_report_ncfrp.py   count-stability diagnostic -> choose --pca-min/--pca-max
ncfrp_manuscript_v5.py     the analysis (--offset-mode joint|marginal; ALL/FEMALES/MALES)
generate_table1.py         Table 1 by bed-sharing status
generate_figure1_flow.py   CONSORT Figure 1 from the audit trail
compare_time_axes.py       PCA vs chronological-age GAM fit comparison + checks
```

## Conventions

- The audited case loader (`ncfrp_manuscript_v5.load_ncfrp_data`) is the single
  source of the analytic cohort; it writes `output_data/ncfrp_flow_audit.csv`.
- Every figure/table reads one saved model-output CSV; a check asserts each CI
  brackets its estimate.
- Fit NB/GAM on raw integer counts; smoothing is display-only.
- `--denominator` for `risk_exposure_joint.py` is the FULL US live-birth file,
  never a matched case/control set.
- Do not write manuscript / response-letter prose into `.docx` files; that is
  the author's. Code, figures, tables, captions, runbooks, and audit docs are
  fair game.
- No em dashes in drafted prose.

## Open decisions (as of 2026-09-02)

G1 retire `../repo/`? · G5 first-week wording · G7 funding/IRB/authorship ·
G8 confirm AJE target · G9 "developmental" vs Ed's "environmental" framing ·
G10 apply the 2008-2019 year window (start N drops from 24,314) or keep all years?

## Next step

1. (Optional backup) `git push -u origin audit/manuscript-2` - committed locally
   as `cf4fc62`; the audit session had no push credentials.
2. **Copy this folder to the HPC by scp** (NOT git):
   `scp -r ~/Projects/SUID_PCA_manuscripts/manuscript_2_bedsharing_pca <user>@<hpc>:<work_dir>/`
   `<work_dir>` also needs `data/SeattleChildrensNCFRP.sav`,
   `data/Codebook_Version5-1_031122.xlsx`, `from_Ed/Eds_ICD_codes.xlsx`, and the
   CDC `denominator.parquet` + PRAMS `prams.parquet`.
3. On the HPC: venv + `pip install -r manuscript_2_bedsharing_pca/requirements.txt`,
   then `bash manuscript_2_bedsharing_pca/run_all.sh 36 90` (see RUNBOOK.md).
4. `scp` `output_data/` + `output_figures/` + `RUN_MANIFEST.md` back; review
   against `AUDIT_FINDINGS.md`; then the reviewer-proofing memo.

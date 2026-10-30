#!/usr/bin/env bash
# Manuscript 2 pipeline driver. See RUNBOOK.md for context and for the steps that
# still need a codebook check / recovered data (marked [CHECK] there).
#
# Usage:
#   cd <work_dir>              # holds data/ , from_Ed/ , and the denominator/PRAMS parquets
#   bash <repo>/manuscript_2_bedsharing_pca/run_all.sh [PCA_MIN] [PCA_MAX] [YEAR_MIN] [YEAR_MAX] [FIRST_WEEK_CUT]
#
# v2 (post lab meeting): death-year window is 2008-2019 by default (gate G10).
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/src" && pwd)"
PCA_MIN="${1:-36}"
PCA_MAX="${2:-90}"          # set from step 4's boundary report; do NOT assume 75
YEAR_MIN="${3:-2008}"       # G10: v2 restricts to 2008-2019
YEAR_MAX="${4:-2019}"
FIRST_WEEK_CUT="${5:-1}"    # exclude infageweeks <= this (G5); 1 = days 0-13 (primary,
                            # matches Manuscript 1); 0 = days 0-6 only, first-week sensitivity
DATA="standardized/NCFRP_SCH_LR_BW_pop.csv"

DENOM="${DENOM:-standardized/consolidated/denominator.parquet}"
PRAMS="${PRAMS:-standardized/PRAMS/prams.parquet}"

echo "== 0. environment =="
python -c "import platform; print('python', platform.python_version())" | tee RUN_MANIFEST.md
pip freeze >> RUN_MANIFEST.md
echo "PCA_MIN=$PCA_MIN PCA_MAX=$PCA_MAX YEAR_MIN=$YEAR_MIN YEAR_MAX=$YEAR_MAX FIRST_WEEK_CUT=$FIRST_WEEK_CUT" | tee -a RUN_MANIFEST.md

echo "== 0b. smoke test =="
python "$SRC/../tests/smoke_synthetic.py"

echo "== 1. preprocess raw NCFRP .sav =="
python "$SRC/preprocess_NCFRP.py"
# -> data/NCFRP_SCH_sample.csv (2008-2019), data/NCFRP_SCH_pop.csv (all years)

echo "== 2. LR_BW birthweight imputation (Tables 1-3 completeness; not used by the primary models) =="
python "$SRC/lr_bw_impute.py" \
    --in  data/NCFRP_SCH_pop.csv \
    --out "$DATA"

echo "== 3. population exposure offsets (JOINT smoke x preterm + marginal bed-share) =="
python "$SRC/risk_exposure_joint.py" \
    --denominator "$DENOM" \
    --prams       "$PRAMS" \
    --outdir      output_data/risk_exposure
# -> output_data/risk_exposure/joint_smoke_preterm_exposure.pkl (+ marginals + .csv + summary)

echo "== 4. PCA-window boundary report (for the 2008-2019 cohort) =="
python "$SRC/pca_boundary_report_ncfrp.py" \
    --data-path "$DATA" \
    --outdir output_data/ncfrp/pca_boundary \
    --min-cell 5 --pca-min "$PCA_MIN" --pca-max "$PCA_MAX" \
    --first-week-cut "$FIRST_WEEK_CUT" \
    --year-min "$YEAR_MIN" --year-max "$YEAR_MAX"
echo "   -> read output_data/ncfrp/pca_boundary/pca_boundary_diagnostics.txt;"
echo "      re-run below with the chosen PCA_MIN / PCA_MAX if they differ"

echo "== 5. main analysis, JOINT offsets (primary) - ALL / FEMALES / MALES / LBW / NONLBW"
echo "        + smoking-stratified windows + <1mo table + bed_share x LBW + months-age sensitivity =="
python "$SRC/ncfrp_manuscript_v5.py" \
    --data-path "$DATA" \
    --exposure-dir output_data/risk_exposure \
    --pca-min "$PCA_MIN" --pca-max "$PCA_MAX" \
    --first-week-cut "$FIRST_WEEK_CUT" \
    --year-min "$YEAR_MIN" --year-max "$YEAR_MAX" \
    --offset-mode joint \
    --min-count 10 \
    --gam-alpha-grid 20 \
    --outdir out_joint

echo "== 5b. sensitivity: marginal-product offsets (the archived approach) =="
python "$SRC/ncfrp_manuscript_v5.py" \
    --data-path "$DATA" \
    --exposure-dir output_data/risk_exposure \
    --pca-min "$PCA_MIN" --pca-max "$PCA_MAX" \
    --first-week-cut "$FIRST_WEEK_CUT" \
    --year-min "$YEAR_MIN" --year-max "$YEAR_MAX" \
    --offset-mode marginal \
    --min-count 10 \
    --no-extra-strata \
    --outdir out_marginal

echo "== 5c. bed_share x PCA varying-coefficient robustness check (A3.27 replacement for the retired gam_res_int) =="
python "$SRC/gam_bedshare_pca_varycoef.py" \
    --data-path "$DATA" \
    --exposure-dir output_data/risk_exposure \
    --pca-min "$PCA_MIN" --pca-max "$PCA_MAX" \
    --first-week-cut "$FIRST_WEEK_CUT" \
    --year-min "$YEAR_MIN" --year-max "$YEAR_MAX" \
    --offset-mode joint \
    --min-count 10 \
    --outdir out_joint
python "$SRC/gam_bedshare_pca_varycoef.py" \
    --data-path "$DATA" \
    --exposure-dir output_data/risk_exposure \
    --pca-min "$PCA_MIN" --pca-max "$PCA_MAX" \
    --first-week-cut "$FIRST_WEEK_CUT" \
    --year-min "$YEAR_MIN" --year-max "$YEAR_MAX" \
    --offset-mode marginal \
    --min-count 10 \
    --outdir out_marginal

echo "== 6. Tables 1-3, CONSORT Figure 1, and the PCA-vs-chronological-age + reporting-artifact analysis =="
for T in generate_table1 generate_table2 generate_table3; do
    python "$SRC/$T.py" \
        --data-path "$DATA" \
        --pca-min "$PCA_MIN" --pca-max "$PCA_MAX" \
        --first-week-cut "$FIRST_WEEK_CUT" \
        --year-min "$YEAR_MIN" --year-max "$YEAR_MAX" \
        --outdir out_joint/output_data/ncfrp
done
python "$SRC/generate_figure1_flow.py" --workdir out_joint
python "$SRC/compare_time_axes.py" \
    --data-path "$DATA" \
    --exposure-dir output_data/risk_exposure \
    --pca-min "$PCA_MIN" --pca-max "$PCA_MAX" \
    --first-week-cut "$FIRST_WEEK_CUT" \
    --year-min "$YEAR_MIN" --year-max "$YEAR_MAX" --offset-mode joint \
    --outdir out_joint/output_data/ncfrp/time_axes

echo "== 7. breastfeeding sensitivity analysis (A/B on NCFRP INF3breast; RUNBOOK 8b) =="
# Reruns the whole pipeline on the breastfed / never / unknown / known subsets,
# with breastfeeding-specific offsets for Y and N, and writes the missingness
# table. Same PCA / year / first-week settings as the primary run.
python "$SRC/bfed_sensitivity.py" \
    --data-path "$DATA" \
    --denominator "$DENOM" \
    --prams "$PRAMS" \
    --national-exposure-dir output_data/risk_exposure \
    --outdir out_bfed \
    --pca-min "$PCA_MIN" --pca-max "$PCA_MAX" \
    --first-week-cut "$FIRST_WEEK_CUT" \
    --year-min "$YEAR_MIN" --year-max "$YEAR_MAX"
# -> out_bfed/bfed_sensitivity_summary.md, bfed_comparison_*.csv, bfed_missingness.csv

echo "== done. compare out_joint/ vs out_marginal/ ; hand back out_joint/, out_marginal/, out_bfed/ and RUN_MANIFEST.md =="

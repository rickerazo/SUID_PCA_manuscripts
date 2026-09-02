#!/usr/bin/env bash
# Manuscript 2 pipeline driver. See RUNBOOK.md for context and for the steps that
# still need a codebook check / recovered data (marked [CHECK] there).
#
# Usage:
#   cd <work_dir>              # holds data/ , from_Ed/ , and the denominator/PRAMS parquets
#   bash <repo>/manuscript_2_bedsharing_pca/run_all.sh [PCA_MIN] [PCA_MAX]
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/src" && pwd)"
PCA_MIN="${1:-36}"
PCA_MAX="${2:-90}"          # set from step 3's boundary report; do NOT assume 75

DENOM="${DENOM:-standardized/consolidated/denominator.parquet}"
PRAMS="${PRAMS:-standardized/PRAMS/prams.parquet}"

echo "== 0. environment =="
python -c "import platform; print('python', platform.python_version())" | tee RUN_MANIFEST.md
pip freeze >> RUN_MANIFEST.md

echo "== 0b. smoke test =="
python "$SRC/../tests/smoke_synthetic.py"

echo "== 1. preprocess raw NCFRP .sav =="
python "$SRC/preprocess_NCFRP.py"
# -> data/NCFRP_SCH_sample.csv (2008-2019), data/NCFRP_SCH_pop.csv (all years)

echo "== 2. LR_BW birthweight imputation (Table 1 completeness; not used by the models) =="
python "$SRC/lr_bw_impute.py" \
    --in  data/NCFRP_SCH_pop.csv \
    --out standardized/NCFRP_SCH_LR_BW_pop.csv

echo "== 3. population exposure offsets (JOINT smoke x preterm + marginal bed-share) =="
python "$SRC/risk_exposure_joint.py" \
    --denominator "$DENOM" \
    --prams       "$PRAMS" \
    --outdir      output_data/risk_exposure
# -> output_data/risk_exposure/joint_smoke_preterm_exposure.pkl (+ marginals + .csv + summary)

echo "== 4. PCA-window boundary report =="
python "$SRC/pca_boundary_report_ncfrp.py" \
    --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \
    --outdir output_data/ncfrp/pca_boundary \
    --min-cell 5 --pca-min "$PCA_MIN" --pca-max "$PCA_MAX"
echo "   -> read output_data/ncfrp/pca_boundary/pca_boundary_diagnostics.txt;"
echo "      re-run below with the chosen PCA_MIN / PCA_MAX if they differ"

echo "== 5. main analysis, JOINT offsets (primary) - ALL / FEMALES / MALES =="
python "$SRC/ncfrp_manuscript_v5.py" \
    --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \
    --exposure-dir output_data/risk_exposure \
    --pca-min "$PCA_MIN" --pca-max "$PCA_MAX" \
    --offset-mode joint \
    --min-count 10 \
    --gam-alpha-grid 20 \
    --outdir out_joint

echo "== 5b. sensitivity: marginal-product offsets (the archived approach) =="
python "$SRC/ncfrp_manuscript_v5.py" \
    --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \
    --exposure-dir output_data/risk_exposure \
    --pca-min "$PCA_MIN" --pca-max "$PCA_MAX" \
    --offset-mode marginal \
    --min-count 10 \
    --outdir out_marginal

echo "== 6. Table 1, CONSORT Figure 1, and the PCA-vs-chronological-age comparison =="
python "$SRC/generate_table1.py" \
    --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \
    --pca-min "$PCA_MIN" --pca-max "$PCA_MAX" \
    --outdir out_joint/output_data/ncfrp
python "$SRC/generate_figure1_flow.py" --workdir out_joint
python "$SRC/compare_time_axes.py" \
    --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \
    --exposure-dir output_data/risk_exposure \
    --pca-min "$PCA_MIN" --pca-max "$PCA_MAX" --offset-mode joint \
    --outdir out_joint/output_data/ncfrp/time_axes

echo "== done. compare out_joint/ vs out_marginal/ ; hand back with RUN_MANIFEST.md =="

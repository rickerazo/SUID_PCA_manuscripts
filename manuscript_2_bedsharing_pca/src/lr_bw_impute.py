#!/usr/bin/env python3
"""
lr_bw_impute.py  ("LR_BW" step)

Recovered and cleaned from
/Users/rerazo/Projects/manuscript_2/scripts/linear_regression.py (T4). That
script mixed an exploratory model comparison (simple vs 2- vs 3-predictor linear
regression, R2 ~= 0.68 / 0.694 / 0.698) with the one step the pipeline actually
needs: impute missing infant birthweight from gestational age and sex, then write
`*_LR_BW_*.csv`. Only that step is kept here.

Method (unchanged from the original's chosen model):
    fit  INF3biwt ~ INF3gestage + INFsex   (ordinary least squares) on the rows
    that have a recorded birthweight, then predict + floor() the birthweight for
    the rows where INF3biwt is missing but gestational age and sex are present.

WHERE THIS SITS
    preprocess_NCFRP.py  ->  lr_bw_impute.py  ->  ncfrp_manuscript_v5.py
The manuscript's incidence models do NOT use birthweight; this step exists so a
Table 1 birthweight row is complete. Running the analysis directly on the
preprocess output (without this step) gives identical model results.

USAGE
    python lr_bw_impute.py \
        --in data/NCFRP_SCH_pop.csv \
        --out standardized/NCFRP_SCH_LR_BW_pop.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

LOG = logging.getLogger("lr_bw_impute")

BW_COL = "INF3biwt"          # cleaned birthweight (grams); NaN where unknown/outlier
GEST_COL = "INF3gestage"
SEX_COL = "INFsex"           # 1 = M, 2 = F, 9 = unknown


def impute_birthweight(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # numeric-clean the predictors: 9 = unknown sex -> NaN so it can't be a level
    sex = df[SEX_COL].where(df[SEX_COL].isin([1, 2]))
    gest = pd.to_numeric(df[GEST_COL], errors="coerce")
    gest = gest.where((gest >= 20) & (gest <= 47))     # implausible -> NaN (A2.10)
    bw = pd.to_numeric(df[BW_COL], errors="coerce")

    train = pd.DataFrame({"bw": bw, "gest": gest, "sex": sex}).dropna()
    if len(train) < 50:
        raise ValueError(f"Only {len(train)} rows have birthweight + gestage + sex; "
                         f"not enough to fit the imputation model.")

    model = LinearRegression().fit(train[["gest", "sex"]].to_numpy(),
                                   train["bw"].to_numpy())
    LOG.info("Fit INF3biwt ~ gest + sex on n=%d  (intercept=%.1f, coef=%s, R2=%.3f)",
             len(train), model.intercept_, np.round(model.coef_, 2),
             model.score(train[["gest", "sex"]].to_numpy(), train["bw"].to_numpy()))

    need = bw.isna() & gest.notna() & sex.notna()
    n_imputed = int(need.sum())
    if n_imputed:
        pred = np.floor(model.predict(
            pd.DataFrame({"gest": gest[need], "sex": sex[need]}).to_numpy()))
        df.loc[need, BW_COL] = pred
    LOG.info("Imputed birthweight for %d rows; %d still missing (no gestage/sex)",
             n_imputed, int(df[BW_COL].isna().sum()))

    df["INF3biwt_imputed_flag"] = 0
    df.loc[need, "INF3biwt_imputed_flag"] = 1
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", type=Path, required=True,
                   help="preprocess output, e.g. data/NCFRP_SCH_pop.csv")
    p.add_argument("--out", dest="out_path", type=Path, required=True,
                   help="e.g. standardized/NCFRP_SCH_LR_BW_pop.csv")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    df = pd.read_csv(args.in_path, low_memory=False)
    df = df.loc[:, ~df.columns.str.match(r"[Uu]nnamed: \d+(\.\d+)?$")]
    for col in (BW_COL, GEST_COL, SEX_COL):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not in {args.in_path}. "
                             f"Present: {list(df.columns)[:40]} ...")

    out = impute_birthweight(df)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_path, index=False)
    LOG.info("Wrote %s  (shape %s)", args.out_path, out.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

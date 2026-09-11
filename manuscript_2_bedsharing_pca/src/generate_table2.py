#!/usr/bin/env python3
"""
generate_table2.py

"Table 2. Characteristics of the analytic SUID cohort by birthweight" - the same
counts + column-% layout as Table 1, restratified from bed sharing to birthweight
category. Bed sharing becomes a row; the birthweight block is dropped (it is the
stratifier). Companion to generate_table1.py / generate_table3.py; the row logic
is build_cohort_table() in generate_table1.py.

Population: the audited pre-peak-split analytic cohort from
ncfrp_manuscript_v5.load_ncfrp_data - identical to Table 1.

Birthweight uses the LR_BW-imputed column (inf3biwt); run lr_bw_impute.py first
and point --data-path at standardized/NCFRP_SCH_LR_BW_pop.csv. Default strata
match the birthweight rows already reported in Table 1 (<2500 / 2500-3499 /
>=3500 g); override with --lbw-cut / --hbw-cut.

USAGE
    python generate_table2.py \
        --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \
        --pca-min 36 --pca-max 90 \
        --outdir output_data/ncfrp
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from ncfrp_manuscript_v5 import load_ncfrp_data
from generate_table1 import (BW_COL, BW_SECTION, add_common_args,
                             build_cohort_table, write_table)

LOG = logging.getLogger("generate_table2")


def table2_by_birthweight(df: pd.DataFrame, *, peak1: int, peak2: int,
                          lbw_cut: int = 2500, hbw_cut: int = 3500) -> pd.DataFrame:
    if BW_COL not in df.columns:
        raise SystemExit(
            f"column {BW_COL!r} not found. Run lr_bw_impute.py and pass "
            f"--data-path standardized/NCFRP_SCH_LR_BW_pop.csv")

    bw = pd.to_numeric(df[BW_COL], errors="coerce")
    strata = [
        (f"<{lbw_cut} g", bw < lbw_cut),
        (f"{lbw_cut}-{hbw_cut - 1} g", (bw >= lbw_cut) & (bw < hbw_cut)),
        (f">={hbw_cut} g", bw >= hbw_cut),
    ]
    if bw.isna().any():
        strata.append(("Birthweight missing", bw.isna()))

    return build_cohort_table(
        df, strata,
        include_bedshare_section=True,
        omit_sections=frozenset({BW_SECTION}),
        peak1=peak1, peak2=peak2,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--lbw-cut", type=int, default=2500,
                   help="low-birthweight cut in grams (default 2500)")
    p.add_argument("--hbw-cut", type=int, default=3500,
                   help="high-birthweight cut in grams (default 3500)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    df, _ = load_ncfrp_data(args.data_path, pca_min=args.pca_min, pca_max=args.pca_max,
                            first_week_cut=args.first_week_cut,
                            year_min=args.year_min, year_max=args.year_max, audit_csv=None)
    LOG.info("Table 2 population (pre peak-split analytic cohort): n=%d", len(df))
    tab = table2_by_birthweight(df, peak1=args.peak1, peak2=args.peak2,
                                lbw_cut=args.lbw_cut, hbw_cut=args.hbw_cut)
    write_table(tab, args.outdir, "table2_by_birthweight", log=LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
generate_table3.py

"Table 3. Characteristics of the analytic SUID cohort by infant sex" - the same
counts + column-% layout as Table 1, restratified from bed sharing to sex. Bed
sharing becomes a row; the "Infant sex" block is dropped (it is the stratifier).
This is the direct analogue of manuscript 1's table1_by_sex. Companion to
generate_table1.py / generate_table2.py; the row logic is build_cohort_table()
in generate_table1.py.

Population: the audited pre-peak-split analytic cohort from
ncfrp_manuscript_v5.load_ncfrp_data - identical to Table 1. In that cohort sex is
already restricted to M / F (loader step 12).

USAGE
    python generate_table3.py \
        --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \
        --pca-min 36 --pca-max 90 \
        --outdir output_data/ncfrp
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from ncfrp_manuscript_v5 import load_ncfrp_data
from generate_table1 import (SEX_SECTION, add_common_args, build_cohort_table,
                             write_table)

LOG = logging.getLogger("generate_table3")


def table3_by_sex(df: pd.DataFrame, *, peak1: int, peak2: int) -> pd.DataFrame:
    sex = df["sex"].astype(str).str.upper().str[0]
    strata = [("Male", sex.eq("M")), ("Female", sex.eq("F"))]
    return build_cohort_table(
        df, strata,
        include_bedshare_section=True,
        omit_sections=frozenset({SEX_SECTION}),
        peak1=peak1, peak2=peak2,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    df, _ = load_ncfrp_data(args.data_path, pca_min=args.pca_min, pca_max=args.pca_max,
                            first_week_cut=args.first_week_cut,
                            year_min=args.year_min, year_max=args.year_max, audit_csv=None)
    LOG.info("Table 3 population (pre peak-split analytic cohort): n=%d", len(df))
    tab = table3_by_sex(df, peak1=args.peak1, peak2=args.peak2)
    write_table(tab, args.outdir, "table3_by_sex", log=LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

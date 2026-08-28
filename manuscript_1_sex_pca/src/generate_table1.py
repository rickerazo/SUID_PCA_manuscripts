#!/usr/bin/env python3
"""
generate_table1.py

Builds "Table 1. Characteristics of the analytic SUID cohort by infant sex" —
counts and percentages of the analytic cohort by covariate, stratified by
the primary exposure (sex), the standard descriptive table this kind of
publication is expected to have and that the current draft is missing (see
the reviewer comment this script responds to: no Table 1 exists; the closest
thing, Supplemental Table 2, mixes a population-prevalence column into what
is otherwise a modeling-results table).

Does not duplicate any cleaning logic — reuses the already-validated
audit_subject_breakdown.py pipeline (clean_case_series, exclude_invalid_
smoking, table1_by_sex) exactly as-is.

POPULATION USED (important, and not the same as either existing table1_by_sex
CSV on disk — see WHY THIS SCRIPT EXISTS below)
------------------------------------------------------------------------------
clean_case_series() [steps 0-6: first-week/unknown-age exclusion, PCA
compute, PCA window] + exclude_invalid_smoking() [step 7], STOPPING BEFORE
split_analytic_windows() [step 8, the 43-46wk peak-interval exclusion].

This is exactly the N=23,686 population HANDOFF_CONTEXT.md's "Final
validated numbers" section labels "Table 1 / Figure 2 / Figure 3
population" — the full analytic cohort used everywhere in this pipeline
EXCEPT the confirmatory windowed regression (which additionally drops the
peak interval). A standard Table 1 describes the whole analytic cohort
under study, not the regression-only subset, so this is the population a
Table 1 should use.

WHY THIS SCRIPT EXISTS (neither existing table1_by_sex() output matches this)
------------------------------------------------------------------------------
table1_by_sex() already existed in audit_subject_breakdown.py before this
script, but nothing on disk actually calls it on the N=23,686 population:
  - `audit/table1_by_sex.csv` (main()'s own CLI output) is built on case_df
    from clean_case_series() ALONE — i.e. BEFORE exclude_invalid_smoking(),
    so it's actually N=23,942 (confirmed by main()'s own log line: "Final
    analytic sample (post PCA-window only)"). Missing the cig_rec filter.
  - `audit_figure1/table1_by_sex_final_analytic.csv` (figure1_reconciliation()'s
    output, despite the "final_analytic" name) is built on corr_final =
    split_analytic_windows()'s early+late combined — i.e. N=19,248, which
    EXCLUDES the peak interval. That's the confirmatory-regression
    population, not the overall analytic cohort.
Neither is wrong for what it was built for, but neither is N=23,686, so
neither is the population a Table 1 for this manuscript should describe.
This script is the first thing to actually produce that.

USAGE
-----
    python generate_table1.py \
        --input output_data/controls/matched_cases_cdc.xlsx \
        --outdir .
"""

from __future__ import annotations  # required: tuple[...]/list[...]/X|None below need this on py3.8

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_subject_breakdown import (  # noqa: E402
    clean_case_series,
    exclude_invalid_smoking,
    read_any,
    table1_by_sex,
)

LOG = logging.getLogger("generate_table1")


def build_table1_population(
    raw_df: pd.DataFrame,
    *,
    infage_col: str,
    combgest_col: str,
    smoking_col: str,
    pca_min: int,
    pca_max: int,
) -> tuple[pd.DataFrame, "AuditTrail"]:  # noqa: F821 - AuditTrail imported transitively
    """Steps 0-7 of the pipeline: PCA window applied, invalid cig_rec dropped,
    peak interval NOT yet excluded. This is the N=23,686 Table 1 population."""
    case_df, _, trail = clean_case_series(
        raw_df,
        infage_col=infage_col,
        combgest_col=combgest_col,
        pca_min=pca_min,
        pca_max=pca_max,
        pca_max_inclusive=False,
        reproduce_original_bug=False,
    )
    case_df = exclude_invalid_smoking(case_df, trail, smoking_col=smoking_col)
    return case_df, trail


def to_markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False))
    return "\n".join([header, sep, body])


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                    help="Path to the raw case file, e.g. matched_cases_cdc.xlsx")
    p.add_argument("--outdir", type=Path, default=Path("."))

    p.add_argument("--infage-col", default="infage")
    p.add_argument("--combgest-col", default="combgest")
    p.add_argument("--sex-col", default="sex")
    p.add_argument("--smoking-col", default="cig_rec")
    p.add_argument("--year-col", default="dob_yy")
    p.add_argument("--cause-col", default=None,
                    help="Explicit cause-of-death column name; auto-detected if omitted.")
    p.add_argument("--pca-min", type=int, default=36)
    p.add_argument("--pca-max", type=int, default=90)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.outdir / "table1_run.log")],
    )

    if not args.input.exists():
        LOG.error("Input file not found: %s", args.input)
        return 1

    LOG.info("Loading %s", args.input)
    raw_df = read_any(args.input)
    LOG.info("Raw shape: %s", raw_df.shape)

    case_df, trail = build_table1_population(
        raw_df,
        infage_col=args.infage_col, combgest_col=args.combgest_col,
        smoking_col=args.smoking_col, pca_min=args.pca_min, pca_max=args.pca_max,
    )
    trail.to_frame().to_csv(args.outdir / "audit_trail_table1.csv", index=False)
    LOG.info("Table 1 population: n=%d (PCA window [%d,%d], valid cig_rec, peak interval "
              "INCLUDED — this is the overall analytic cohort, not the windowed-regression "
              "subset)", len(case_df), args.pca_min, args.pca_max)

    table1 = table1_by_sex(
        case_df,
        sex_col=args.sex_col,
        postnatal_age_col=args.infage_col,
        gestational_age_col=args.combgest_col,
        smoking_col=args.smoking_col,
        year_col=args.year_col,
        cause_col=args.cause_col,
        pca_col="pca",
    )

    csv_path = args.outdir / "table1_by_sex.csv"
    xlsx_path = args.outdir / "table1_by_sex.xlsx"
    md_path = args.outdir / "table1_by_sex.md"
    table1.to_csv(csv_path, index=False)
    table1.to_excel(xlsx_path, index=False)

    title = (
        f"Table 1. Characteristics of the analytic SUID cohort by infant sex "
        f"(N={len(case_df):,})."
    )
    with open(md_path, "w") as f:
        f.write(title + "\n\n")
        f.write(to_markdown_table(table1))
        f.write("\n")

    LOG.info(title)
    print(title)
    print()
    print(table1.to_string(index=False))
    print(f"\nWrote {csv_path}, {xlsx_path}, {md_path}")
    LOG.info("DONE. See %s", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
generate_table1.py  (AUDIT A4.29 / R1.8 / Ed E14)

"Table 1. Characteristics of the analytic SUID cohort by bed-sharing status" -
counts and column percentages, the conventional descriptive table reviewer 1
asked manuscript 1 for and manuscript 2 is missing.

Population: the audited analytic cohort from ncfrp_manuscript_v5.load_ncfrp_data
(all data-quality exclusions + the PCA window), BEFORE the early/late/peak split -
a Table 1 describes the whole cohort under study, not the regression-only subset
(same choice manuscript 1's generate_table1.py made).

Structure ported from ../manuscript_1_sex_pca/src/audit_subject_breakdown.py
::table1_by_sex, restratified from sex to bed sharing (the primary exposure).
Sex becomes a row. Birthweight uses the LR_BW-imputed column when present.

The row-generation logic lives in build_cohort_table(); generate_table2.py
(by birthweight) and generate_table3.py (by sex) reuse it with a different
stratifying variable.

USAGE
    python generate_table1.py \
        --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \
        --pca-min 36 --pca-max 90 \
        --outdir output_data/ncfrp
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ncfrp_manuscript_v5 import load_ncfrp_data, PEAK1, PEAK2

LOG = logging.getLogger("generate_table1")

# CAUcausedth category labels (NCFRP). Adjust to the codebook if these differ.
CAUSE_LABELS = {1: "Undetermined", 2: "Accidental suffocation / strangulation",
                3: "Natural / medical", 4: "Unknown"}

# Section headings other table scripts omit when that variable is the stratifier.
SEX_SECTION = "Infant sex"
BW_SECTION = "Birthweight (grams; LR_BW-imputed where missing)"
BW_COL = "inf3biwt"


def _fmt(n: int, denom: int) -> str:
    n, denom = int(n), int(denom)
    return f"{n:,} (NA)" if denom == 0 else f"{n:,} ({100 * n / denom:.1f})"


def build_cohort_table(
    df: pd.DataFrame,
    strata: list[tuple[str, pd.Series]],
    *,
    include_bedshare_section: bool = False,
    omit_sections: frozenset[str] = frozenset(),
    peak1: int = PEAK1,
    peak2: int = PEAK2,
) -> pd.DataFrame:
    """
    Counts + column % for the analytic cohort, one column per stratum.

    strata                   ordered [(column label, boolean mask), ...]; an
                             "Overall" column is prepended automatically.
    include_bedshare_section add a "Bed sharing" Yes/No block (used by the
                             by-birthweight and by-sex tables, where bed sharing
                             is a row rather than the stratifier).
    omit_sections            section headings to drop entirely - the stratifying
                             variable's own block (e.g. BW_SECTION for Table 2).
    """
    data = df.copy()
    d_all = len(data)
    col_all = f"Overall N = {d_all:,}"

    strat_cols: list[str] = []
    strat_masks: list[pd.Series] = []
    strat_denoms: list[int] = []
    for label, mask in strata:
        mask = mask.fillna(False)
        cnt = int(mask.sum())
        strat_cols.append(f"{label} N = {cnt:,}")
        strat_masks.append(mask)
        strat_denoms.append(cnt)

    rows: list[dict] = []
    state = {"skip": False}

    def section(label):
        state["skip"] = label in omit_sections
        if state["skip"]:
            return
        rows.append({"Characteristic": label, col_all: "",
                     **{c: "" for c in strat_cols}})

    def row(label, mask):
        if state["skip"]:
            return
        mask = mask.fillna(False)
        rec = {"Characteristic": label, col_all: _fmt(mask.sum(), d_all)}
        for col, sm, denom in zip(strat_cols, strat_masks, strat_denoms):
            rec[col] = _fmt((mask & sm).sum(), denom)
        rows.append(rec)

    section("Case counts")
    row("Total SUID cases", pd.Series(True, index=data.index))

    section(SEX_SECTION)
    row("Male", data["sex"].eq("M"))
    row("Female", data["sex"].eq("F"))

    if include_bedshare_section:
        bs = data["bed_share"].astype(str).str.upper().str[0]
        section("Bed sharing")
        row("Yes", bs.eq("Y"))
        row("No", bs.eq("N"))

    if "caucausedth" in data.columns:
        section("Cause of death (CAUcausedth)")
        cau = pd.to_numeric(data["caucausedth"], errors="coerce")
        for code, label in CAUSE_LABELS.items():
            row(f"{label} ({code})", cau.eq(code))
        row("Other / not in map", ~cau.isin(list(CAUSE_LABELS)))

    section("Maternal smoking during pregnancy")
    row("Yes", data["cig_rec"].eq("Y"))
    row("No", data["cig_rec"].eq("N"))

    section("Gestational age at birth")
    ga = pd.to_numeric(data["inf3gestage"], errors="coerce")
    row("Extremely/very preterm, <32 weeks", ga < 32)
    row("Moderate/late preterm, 32-36 weeks", (ga >= 32) & (ga < 37))
    row("Term, >=37 weeks", ga >= 37)

    section("Postconceptional age window")
    pca = pd.to_numeric(data["pca"], errors="coerce")
    row(f"Early rising, PCA < {peak1}", pca < peak1)
    row(f"Peak/transition, {peak1}-{peak2 - 1}", (pca >= peak1) & (pca < peak2))
    row(f"Late declining, PCA >= {peak2}", pca >= peak2)

    section("Postnatal age at death")
    wk = pd.to_numeric(data["infageweeks"], errors="coerce")
    wpm = 365.25 / 12 / 7
    row("<1 month", wk < wpm)
    row("1 to <2 months", (wk >= wpm) & (wk < 2 * wpm))
    row("2 to <4 months", (wk >= 2 * wpm) & (wk < 4 * wpm))
    row("4 to <6 months", (wk >= 4 * wpm) & (wk < 6 * wpm))
    row(">=6 months", wk >= 6 * wpm)

    if BW_COL in data.columns:
        section(BW_SECTION)
        bw = pd.to_numeric(data[BW_COL], errors="coerce")
        row("<2500", bw < 2500)
        row("2500-3499", (bw >= 2500) & (bw < 3500))
        row(">=3500", bw >= 3500)
        row("Missing (no gestage/sex to impute)", bw.isna())

    if "infdodyear" in data.columns:
        section("Year of death")
        yr = pd.to_numeric(data["infdodyear"], errors="coerce")
        for lo, hi in [(2004, 2007), (2008, 2011), (2012, 2015), (2016, 2019)]:
            m = (yr >= lo) & (yr <= hi)
            if m.any():   # drop bins outside the applied --year-min/--year-max window
                row(f"{lo}-{hi}", m)

    return pd.DataFrame(rows)


def table1_by_bedshare(df: pd.DataFrame, *, peak1: int = PEAK1,
                       peak2: int = PEAK2) -> pd.DataFrame:
    bs = df["bed_share"].astype(str).str.upper().str[0]
    strata = [("Bed sharing", bs.eq("Y")), ("Not bed sharing", bs.eq("N"))]
    return build_cohort_table(df, strata, peak1=peak1, peak2=peak2)


def write_table(tab: pd.DataFrame, outdir: Path, stem: str,
                log: logging.Logger = LOG) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(outdir / f"{stem}.csv", index=False)
    with open(outdir / f"{stem}.md", "w") as f:
        f.write(tab.to_markdown(index=False))
    log.info("Wrote %s", outdir / f"{stem}.csv")
    print(tab.to_string(index=False))


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-path", type=Path,
                   default=Path("standardized/NCFRP_SCH_LR_BW_pop.csv"))
    p.add_argument("--pca-min", type=int, default=36)
    p.add_argument("--pca-max", type=int, default=90)
    p.add_argument("--peak1", type=int, default=PEAK1)
    p.add_argument("--peak2", type=int, default=PEAK2)
    p.add_argument("--year-min", type=int, default=None)
    p.add_argument("--year-max", type=int, default=None)
    p.add_argument("--outdir", type=Path, default=Path("output_data/ncfrp"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    df, _ = load_ncfrp_data(args.data_path, pca_min=args.pca_min, pca_max=args.pca_max,
                            year_min=args.year_min, year_max=args.year_max, audit_csv=None)
    LOG.info("Table 1 population (pre peak-split analytic cohort): n=%d", len(df))
    tab = table1_by_bedshare(df, peak1=args.peak1, peak2=args.peak2)
    write_table(tab, args.outdir, "table1_by_bedshare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

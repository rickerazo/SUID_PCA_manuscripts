#!/usr/bin/env python3
"""
audit_subject_breakdown.py

Standalone audit script for the SIDS/SUID sex-differences manuscript pipeline.
Derived from SIDS_Collaboration_manuscript_pipeline_v4.ipynb, corrected and
restructured to run outside the notebook (e.g. on the HPC where the real data
lives) and to leave a paper trail of exactly how the analytic sample was built.

WHAT THIS FIXES vs. the notebook
---------------------------------
1. `load_case_control()` in the notebook builds the "exclude first-week
   deaths" filter like this:

        case_df.loc[case_df['infage'] <= 1, infage] = np.nan   # infage is a
        case_df.dropna(subset=['infage'], inplace=True)        # *parameter*,
                                                                 # bound to
                                                                 # 'pca' when
                                                                 # called from
                                                                 # main()

   Because `infage` (the parameter) held the string 'pca' rather than the
   literal column name 'infage', the NaN got written into a freshly-created
   'pca' column instead of the real 'infage' column, and the subsequent
   `dropna(subset=['infage'])` removed nothing. The first-week exclusion was
   silently a no-op. `clean_case_series()` below fixes this by always keying
   off the *actual* column names and never aliasing them through a
   caller-supplied "primary variable" argument.

2. `make_suid_table1_by_sex()` defaulted `output_csv` and `output_excel` to
   the SAME path ('output_data/aje', no extension), so the Excel write
   silently clobbered the CSV write and neither file had a sane extension.
   `table1_by_sex()` below uses distinct, explicit filenames.

3. The upper PCA bound was 75 in the notebook. `pca_boundary_report()`
   (added in this script) showed that PCA 34-90 is the contiguous block
   where sex-specific weekly counts stay at or above the manuscript's own
   stated stability threshold (min_cell=5) — i.e. 75 was well inside the
   region the data can support, leaving usable non-sparse weeks unused.
   The default `--pca-max` here is now 90. Pass `--pca-max 75` to reproduce
   the original notebook window if you need it for comparison.

Everything else (bin definitions, Table 1 structure) is kept faithful to
the original notebook logic.

USAGE
-----
    # 1. First, sanity-check that your column names match what this script
    #    expects (they should, if your file matches matched_cases_cdc.xlsx):
    python audit_subject_breakdown.py --input /path/to/matched_cases_cdc.xlsx --inspect-only

    # 2. Run the corrected audit + Table 1 + PCA-window boundary diagnostics
    #    (uses pca_max=90 by default; see point 3 above):
    python audit_subject_breakdown.py \
        --input /path/to/matched_cases_cdc.xlsx \
        --outdir ./audit_output

    # 3. Also see how much the original bug + original 75wk window changed
    #    the numbers (writes a side-by-side comparison report):
    python audit_subject_breakdown.py \
        --input /path/to/matched_cases_cdc.xlsx \
        --outdir ./audit_output \
        --compare-buggy

Only pandas / numpy / openpyxl (for .xlsx I/O) are required — no plotting,
no statsmodels, so it's cheap to run repeatedly while you're re-orienting.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("audit_subject_breakdown")


# ---------------------------------------------------------------------------
# Audit trail bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class AuditTrail:
    """Records an ordered, CONSORT-style flow of n at each cleaning step."""

    label: str = "trail"
    steps: list = field(default_factory=list)
    total_n: int | None = None  # n at step 0, i.e. the raw starting total

    def record(self, step_label: str, df: pd.DataFrame, note: str = "") -> None:
        n = len(df)
        prev_n = self.steps[-1]["n_after"] if self.steps else None
        removed = (prev_n - n) if prev_n is not None else 0

        if self.total_n is None:
            self.total_n = n  # first call to record() defines the 100% baseline

        pct_removed_of_total = 100 * removed / self.total_n if self.total_n else 0.0
        pct_retained_of_total = 100 * n / self.total_n if self.total_n else 0.0

        self.steps.append(
            {
                "step": step_label,
                "n_before": prev_n if prev_n is not None else n,
                "n_after": n,
                "n_removed": removed,
                "pct_removed_of_total": round(pct_removed_of_total, 2),
                "pct_retained_of_total": round(pct_retained_of_total, 2),
                "note": note,
            }
        )
        LOG.info(
            "[%-55s] n=%7d  (-%d, -%.1f%% of total)  retained=%.1f%% of total  %s",
            step_label, n, removed, pct_removed_of_total, pct_retained_of_total, note,
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unrecognized input extension '{suffix}' for {path}. "
                      f"Expected .xlsx, .parquet, or .csv.")


def inspect(df: pd.DataFrame) -> None:
    print(f"\nShape: {df.shape}\n")
    print("Columns and dtypes:")
    print(df.dtypes.to_string())
    print("\nHead:")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.head(10))


# ---------------------------------------------------------------------------
# Corrected cleaning pipeline
# ---------------------------------------------------------------------------

def clean_through_pca(
    raw_df: pd.DataFrame,
    *,
    infage_col: str = "infage",
    combgest_col: str = "combgest",
    reproduce_original_bug: bool = False,
    bug_shadow_col: str = "pca",
) -> tuple[pd.DataFrame, AuditTrail]:
    """
    Steps 0-3 of load_case_control(): load, exclude first-week deaths,
    exclude unknown gestational/infant age, compute PCA. Deliberately stops
    BEFORE the 36-75 PCA-window restriction, so the returned frame still
    contains the full observed PCA range. `pca_boundary_report()` needs that
    full range to diagnose the window boundaries themselves; `clean_case_series()`
    consumes this and continues on to apply the window.
    """
    df = raw_df.copy()
    trail = AuditTrail(label="buggy" if reproduce_original_bug else "corrected")
    trail.record("0. Raw case file as loaded", df)

    CONTROL_FILE_MARKERS = {"propensity_score", "outcome", "dweekday"}
    for col in (infage_col, combgest_col):
        if col not in df.columns:
            hint = ""
            if CONTROL_FILE_MARKERS & set(df.columns):
                hint = (
                    "\nHint: this input has columns typical of the matched "
                    "CONTROL/denominator file (e.g. 'propensity_score', 'outcome'), "
                    "not the case series. Case files have no death-age column "
                    "because controls don't die. Point --input at the cases file "
                    "instead (in the notebook this is "
                    "'output_data/controls/matched_cases_cdc.xlsx', NOT "
                    "'matched_controls.xlsx')."
                )
            raise KeyError(
                f"Expected column '{col}' not found in input.{hint}\n"
                f"Available columns: {list(df.columns)}"
            )

    # --- 1. Exclude deaths in the first week of life (infage <= 1 week) ----
    if reproduce_original_bug:
        # Faithful reproduction of the notebook bug: NaN gets written into
        # `bug_shadow_col` (whatever the caller's "primary variable" name
        # was, e.g. 'pca') instead of the real infage column, so the
        # following dropna on infage_col removes nothing.
        df.loc[df[infage_col] <= 1, bug_shadow_col] = np.nan
        df = df.dropna(subset=[infage_col])
        trail.record(
            "1. Exclude infage<=1wk [REPRODUCING ORIGINAL BUG: no-op]", df
        )
    else:
        df.loc[df[infage_col] <= 1, infage_col] = np.nan
        df = df.dropna(subset=[infage_col])
        trail.record("1. Exclude first-week deaths (infage<=1wk)", df)

    # --- 2. Exclude unknown gestational age (combgest >= 99) ---------------
    df.loc[df[combgest_col] >= 99, combgest_col] = np.nan
    df = df.dropna(subset=[combgest_col])
    trail.record("2. Exclude unknown gestational age (combgest>=99)", df)

    # --- 2b. Exclude unknown infant age (infage >= 99) ----------------------
    df.loc[df[infage_col] >= 99, infage_col] = np.nan
    df = df.dropna(subset=[infage_col])
    trail.record("2b. Exclude unknown infant age (infage>=99)", df)

    # --- 3. Compute post-conceptional age (PCA) -----------------------------
    df["pca"] = df[combgest_col] + df[infage_col]
    df = df.dropna(subset=["pca"])
    trail.record("3. Compute PCA = combgest + infage; drop missing", df)

    return df, trail


def clean_case_series(
    raw_df: pd.DataFrame,
    *,
    infage_col: str = "infage",
    combgest_col: str = "combgest",
    pca_min: int = 36,
    pca_max: int = 90,
    pca_max_inclusive: bool = False,
    reproduce_original_bug: bool = False,
    bug_shadow_col: str = "pca",
) -> tuple[pd.DataFrame, pd.DataFrame, AuditTrail]:
    """
    Rebuild the analytic case series with the same steps as the notebook's
    load_case_control(), corrected.

    Set reproduce_original_bug=True to intentionally reproduce the historical
    no-op first-week filter (bug_shadow_col mimics whatever `var1` the
    notebook was called with — 'pca' in production) purely for comparison
    purposes. Do not use that path for anything you intend to report.

    Set pca_max_inclusive=True to exclude 'PCA >= pca_max' instead of the
    current notebook's 'PCA > pca_max'. Published Figure 1 (n=1,086 excluded
    at the upper bound, pca_max=75) was generated with the inclusive rule;
    the notebook has since been edited to the exclusive rule. Use
    pca_max_inclusive=True together with pca_max=75 and
    reproduce_original_bug=True to reproduce the published Figure 1 exactly.

    Returns (final_df, pre_window_df, trail):
        final_df       - after the full pipeline including the PCA window
        pre_window_df  - after steps 0-3 only (see clean_through_pca), i.e.
                          before the 36-75 restriction; feed this to
                          pca_boundary_report() to diagnose the window itself
        trail          - AuditTrail covering all steps, 0 through 6
    """
    df, trail = clean_through_pca(
        raw_df,
        infage_col=infage_col,
        combgest_col=combgest_col,
        reproduce_original_bug=reproduce_original_bug,
        bug_shadow_col=bug_shadow_col,
    )
    pre_window_df = df.copy()

    # --- 5. Drop PCA below the analytic window -------------------------------
    df.loc[df["pca"] < pca_min, "pca"] = np.nan
    df = df.dropna(subset=["pca"])
    trail.record(f"5. Exclude PCA < {pca_min}", df)

    # --- 6. Drop PCA above the analytic window --------------------------------
    if pca_max_inclusive:
        df.loc[df["pca"] >= pca_max, "pca"] = np.nan
        step_label = f"6. Exclude PCA >= {pca_max}"
    else:
        df.loc[df["pca"] > pca_max, "pca"] = np.nan
        step_label = f"6. Exclude PCA > {pca_max}"
    df = df.dropna(subset=["pca"])
    trail.record(step_label, df)

    return df.reset_index(drop=True), pre_window_df, trail


# ---------------------------------------------------------------------------
# Steps 7-8: valid smoking status + early/late window split
# (completes the case-selection flow shown in the manuscript's Figure 1)
# ---------------------------------------------------------------------------

def exclude_invalid_smoking(
    df: pd.DataFrame,
    trail: AuditTrail,
    *,
    smoking_col: str = "cig_rec",
    valid_values: tuple[str, ...] = ("Y", "N"),
) -> pd.DataFrame:
    """
    Step 7: keep only rows with a valid maternal-smoking value. Mirrors the
    notebook's count_data(), which implicitly drops any row whose cig_rec is
    not exactly 'Y' or 'N' by only ever selecting those two values when
    building the modeling strata.
    """
    out = df.loc[df[smoking_col].isin(valid_values)].copy()
    trail.record(
        f"7. Exclude missing/invalid {smoking_col} (not in {valid_values})", out
    )
    return out


def split_analytic_windows(
    df: pd.DataFrame,
    trail: AuditTrail,
    *,
    pca_col: str = "pca",
    peak1: int = 43,
    peak2: int = 47,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Step 8: split into the early window (PCA < peak1), late window
    (PCA >= peak2), and the peak interval [peak1, peak2) that is summarized
    descriptively but excluded from the windowed negative binomial models.
    Mirrors prepare_data() in the notebook.

    Returns (early_df, late_df, peak_df, combined_df) where combined_df is
    early_df + late_df, i.e. the final analytic series.
    """
    early = df.loc[df[pca_col] < peak1].copy()
    late = df.loc[df[pca_col] >= peak2].copy()
    peak = df.loc[(df[pca_col] >= peak1) & (df[pca_col] < peak2)].copy()
    combined = pd.concat([early, late], ignore_index=True)

    trail.record(
        f"8. Exclude peak-phase [{peak1},{peak2}) weeks "
        f"(descriptive only, excluded from windowed models)",
        combined,
        note=f"early(<{peak1}) n={len(early)}, late(>={peak2}) n={len(late)}, "
             f"peak(excluded) n={len(peak)}",
    )
    return early, late, peak, combined


# Published Figure 1 (SUID_manuscript_v3.docx / Erazo_SUID_Figure1.pdf), for
# automated validation of the reproduction below.
PUBLISHED_FIGURE1 = {
    "start": 25257,
    "excl_pca_low": 336,
    "excl_pca_high": 1086,
    "excl_invalid_smoking": 255,
    "excl_peak": 4494,
    "final": 19086,
    "early": 3833,
    "late": 15253,
}


def figure1_reconciliation(
    raw_df: pd.DataFrame,
    outdir: Path,
    *,
    infage_col: str = "infage",
    combgest_col: str = "combgest",
    smoking_col: str = "cig_rec",
    sex_col: str = "sex",
    pca_min: int = 36,
    pca_max: int = 90,
    peak1: int = 43,
    peak2: int = 47,
) -> None:
    """
    Two full runs of the case-selection flow shown in the manuscript's
    Figure 1:

    1. An EXACT reproduction of the original pipeline (bug active, pca_max=75,
       inclusive upper bound) — validated line-by-line against the numbers
       actually printed in the published Figure 1 (PUBLISHED_FIGURE1 above).
       If every line matches, that's strong confirmation this script's
       understanding of the pipeline is correct, not just plausible.
    2. The CORRECTED flow (bug fixed, pca_max widened per pca_boundary_report
       findings), producing the true final analytic N to use going forward.

    Writes:
      figure1_reproduction_check.txt   step-by-step PASS/MISMATCH vs. the
                                        published numbers
      audit_trail_figure1_reproduction.csv
      figure1_corrected.txt            the new flow, same format
      audit_trail_figure1_corrected.csv
      table1_by_sex_final_analytic.csv/.xlsx   Table 1 on the corrected FINAL
                                        analytic series (early + late windows
                                        combined) — this is the population
                                        "Among N eligible deaths..." should
                                        describe in Results, not the broader
                                        post-PCA-window set.
      table1_by_sex_early_window.csv / table1_by_sex_late_window.csv
    """
    # --- 1. Exact reproduction of the published pipeline -------------------
    repro_df, _, repro_trail = clean_case_series(
        raw_df,
        infage_col=infage_col,
        combgest_col=combgest_col,
        pca_min=pca_min,
        pca_max=75,
        pca_max_inclusive=True,
        reproduce_original_bug=True,
    )
    repro_df = exclude_invalid_smoking(repro_df, repro_trail, smoking_col=smoking_col)
    repro_early, repro_late, repro_peak, repro_final = split_analytic_windows(
        repro_df, repro_trail, peak1=peak1, peak2=peak2
    )
    repro_trail.to_frame().to_csv(outdir / "audit_trail_figure1_reproduction.csv", index=False)

    # Pull exact removed-counts straight off the trail steps by label prefix,
    # rather than hardcoding step indices (robust to step-order edits above).
    def _removed(label_prefix: str) -> int:
        return _trail_removed(repro_trail, label_prefix)

    check_rows = [
        ("start", len(raw_df), PUBLISHED_FIGURE1["start"]),
        ("excl_pca_low", _removed("5. Exclude PCA <"), PUBLISHED_FIGURE1["excl_pca_low"]),
        ("excl_pca_high", _removed("6. Exclude PCA"), PUBLISHED_FIGURE1["excl_pca_high"]),
        ("excl_invalid_smoking", _removed("7. Exclude missing/invalid"), PUBLISHED_FIGURE1["excl_invalid_smoking"]),
        ("excl_peak", len(repro_peak), PUBLISHED_FIGURE1["excl_peak"]),
        ("final", len(repro_final), PUBLISHED_FIGURE1["final"]),
        ("early", len(repro_early), PUBLISHED_FIGURE1["early"]),
        ("late", len(repro_late), PUBLISHED_FIGURE1["late"]),
    ]

    check_path = outdir / "figure1_reproduction_check.txt"
    with open(check_path, "w") as f:
        f.write("Figure 1 reproduction check (bug ACTIVE, pca_max=75, inclusive "
                "upper bound — this should reproduce the PUBLISHED figure)\n")
        f.write("=" * 78 + "\n\n")
        all_match = True
        for label, computed, published in check_rows:
            status = "PASS " if computed == published else "MISMATCH"
            if computed != published:
                all_match = False
            f.write(f"  [{status}] {label:22s} computed={computed:<8d} published={published}\n")
        f.write("\n")
        if all_match:
            f.write("ALL ROWS MATCH. This script's reproduction of the case-selection "
                    "pipeline (including the bug) is confirmed correct — the corrected "
                    "flow below can be trusted as the true fix, not a guess.\n")
        else:
            f.write("AT LEAST ONE ROW DID NOT MATCH. Do not treat the corrected numbers "
                    "below as final until the mismatch is understood — something about "
                    "this reproduction differs from what actually generated the "
                    "published Figure 1 (e.g. a filter order, a threshold, or a data "
                    "vintage difference). Check the MISMATCH row(s) first.\n")
    LOG.info("Figure 1 reproduction check: %s (see %s)",
              "ALL PASS" if all_match else "MISMATCH FOUND", check_path)

    # --- 2. Corrected flow ---------------------------------------------------
    corr_df, _, corr_trail = clean_case_series(
        raw_df,
        infage_col=infage_col,
        combgest_col=combgest_col,
        pca_min=pca_min,
        pca_max=pca_max,
        pca_max_inclusive=False,
        reproduce_original_bug=False,
    )
    corr_df = exclude_invalid_smoking(corr_df, corr_trail, smoking_col=smoking_col)
    corr_early, corr_late, corr_peak, corr_final = split_analytic_windows(
        corr_df, corr_trail, peak1=peak1, peak2=peak2
    )
    corr_trail.to_frame().to_csv(outdir / "audit_trail_figure1_corrected.csv", index=False)

    corrected_path = outdir / "figure1_corrected.txt"
    with open(corrected_path, "w") as f:
        f.write(f"Corrected case-selection flow (first-week bug fixed, "
                f"PCA window {pca_min}-{pca_max})\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Start: SUID deaths (raw, before any exclusion)      n = {len(raw_df)}\n")
        f.write(f"Excluded: first-week deaths (infage<=1wk)           n = {_trail_removed(corr_trail, '1. Exclude first-week')}\n")
        f.write(f"Excluded: unknown gestational/infant age            n = "
                f"{_trail_removed(corr_trail, '2. Exclude unknown') + _trail_removed(corr_trail, '2b. Exclude unknown')}\n")
        f.write(f"Excluded: PCA < {pca_min}                                n = {_trail_removed(corr_trail, '5. Exclude PCA <')}\n")
        f.write(f"Excluded: PCA > {pca_max}                                n = {_trail_removed(corr_trail, '6. Exclude PCA')}\n")
        f.write(f"Excluded: missing/invalid {smoking_col}                    n = {_trail_removed(corr_trail, '7. Exclude missing')}\n")
        f.write(f"Excluded: peak phase [{peak1},{peak2}) weeks (descriptive only)   n = {len(corr_peak)}\n\n")
        f.write(f"Final analytic case series:                          N = {len(corr_final)}\n")
        f.write(f"  Early window (<{peak1}):                                n = {len(corr_early)}\n")
        f.write(f"  Late window (>={peak2}):                                n = {len(corr_late)}\n\n")
        f.write(f"For comparison, published Figure 1 final N was "
                f"{PUBLISHED_FIGURE1['final']} "
                f"({PUBLISHED_FIGURE1['early']} early + {PUBLISHED_FIGURE1['late']} late).\n")
    LOG.info("Wrote corrected case-selection flow to %s (final N=%d)",
              corrected_path, len(corr_final))

    # --- Table 1 on the true final analytic series, and each window --------
    for name, sub in (
        ("final_analytic", corr_final),
        ("early_window", corr_early),
        ("late_window", corr_late),
    ):
        if sub.empty:
            continue
        t1 = table1_by_sex(sub, sex_col=sex_col, postnatal_age_col=infage_col,
                            gestational_age_col=combgest_col, smoking_col=smoking_col)
        t1.to_csv(outdir / f"table1_by_sex_{name}.csv", index=False)
    LOG.info("Wrote table1_by_sex_final_analytic.csv, _early_window.csv, _late_window.csv")


def _trail_removed(trail: AuditTrail, label_prefix: str) -> int:
    for step in trail.steps:
        if step["step"].startswith(label_prefix):
            return step["n_removed"]
    raise KeyError(f"No trail step starting with {label_prefix!r}")


# ---------------------------------------------------------------------------
# PCA window boundary diagnostics
# (answers "why 36-75, and why isn't the exclusion symmetric?")
# ---------------------------------------------------------------------------

def pca_boundary_report(
    pre_window_df: pd.DataFrame,
    outdir: Path,
    *,
    sex_col: str = "sex",
    combgest_col: str = "combgest",
    infage_col: str = "infage",
    pca_min: int = 36,
    pca_max: int = 90,
    min_cell: int = 5,
    tail_pct: float = 1.5,
) -> None:
    """
    Diagnose the PCA-window boundary choice directly from the data, to answer
    a reviewer question like:

        "Why 36-75? The tails aren't symmetric (X% vs Y% excluded) — if this
         is about sparse sex-specific counts, show it; if not, justify the
         asymmetry, or widen the window."

    Takes the PRE-window frame (post first-week/unknown-age cleaning, but
    before the 36-75 restriction) so the full observed PCA range is visible.

    Writes:
      pca_weekly_counts.csv        n_total/n_male/n_female by PCA week, with a
                                    flag for weeks where either sex's count is
                                    below `min_cell` — the manuscript's own
                                    stated "sparse counts" rationale, made
                                    concrete and auditable, on BOTH tails.
      pca_low_tail_excluded.csv    demographic breakdown of everyone excluded
                                    by PCA < pca_min (gestational age x
                                    postnatal age), so the manuscript can
                                    describe concretely who this excludes —
                                    e.g. a 30-week-gestation infant who died
                                    at postnatal week 4 is exactly this group.
      pca_high_tail_excluded.csv   same, for PCA > pca_max.
      pca_boundary_diagnostics.txt narrative summary: percentile-based cutoffs
                                    for comparison, the exact >=cutoff vs
                                    >cutoff sensitivity at the chosen boundary
                                    (tests whether an inclusive vs exclusive
                                    upper bound explains a discrepancy against
                                    previously reported/published counts), and
                                    draft language for the Methods section.
    """
    df = pre_window_df.copy()
    df["pca"] = pd.to_numeric(df["pca"], errors="coerce")
    df[combgest_col] = pd.to_numeric(df[combgest_col], errors="coerce")
    df[infage_col] = pd.to_numeric(df[infage_col], errors="coerce")
    df = df.dropna(subset=["pca"])
    df["pca"] = df["pca"].astype(int)

    sex_upper = df[sex_col].astype(str).str.upper()
    is_male = sex_upper.str.startswith("M")
    is_female = sex_upper.str.startswith("F")

    n_total = len(df)
    if n_total == 0:
        LOG.warning("pca_boundary_report: empty input, skipping.")
        return

    # --- weekly counts across the FULL observed PCA range, not just the window
    full_index = range(int(df["pca"].min()), int(df["pca"].max()) + 1)
    weekly = (
        df.groupby("pca").size().reindex(full_index, fill_value=0).rename("n_total").to_frame()
    )
    weekly["n_male"] = df.loc[is_male].groupby("pca").size().reindex(weekly.index, fill_value=0)
    weekly["n_female"] = df.loc[is_female].groupby("pca").size().reindex(weekly.index, fill_value=0)
    weekly["sparse_by_sex"] = (weekly["n_male"] < min_cell) | (weekly["n_female"] < min_cell)
    weekly["in_chosen_window"] = (weekly.index >= pca_min) & (weekly.index <= pca_max)
    weekly["cum_pct_at_or_below"] = 100 * weekly["n_total"].cumsum() / n_total
    weekly["cum_pct_at_or_above"] = 100 * weekly["n_total"][::-1].cumsum()[::-1] / n_total
    weekly.index.name = "pca_week"
    weekly.to_csv(outdir / "pca_weekly_counts.csv")

    # --- where would a *symmetric* percentile trim actually cut? -----------
    s = df["pca"].sort_values()
    lo_cut = s.quantile(tail_pct / 100)
    hi_cut = s.quantile(1 - tail_pct / 100)

    # --- who exactly gets excluded at each end of the CHOSEN window? -------
    def _tail_breakdown(mask: pd.Series, label: str) -> pd.DataFrame:
        sub = df.loc[mask]
        if sub.empty:
            return pd.DataFrame(columns=["tail", "gestational_age_bin", "postnatal_age_bin", "n"])
        ga_bin = pd.cut(
            sub[combgest_col],
            bins=[0, 27, 31, 33, 36, 200],
            labels=["<28wk (extremely preterm)", "28-31wk (very preterm)",
                    "32-33wk (moderate preterm)", "34-36wk (late preterm)",
                    ">=37wk (term)"],
        )
        pna_bin = pd.cut(
            sub[infage_col],
            bins=[0, 1, 4, 8, 12, 1000],
            labels=["<=1wk", "2-4wk", "5-8wk", "9-12wk", ">12wk"],
        )
        out = (
            sub.assign(gestational_age_bin=ga_bin, postnatal_age_bin=pna_bin)
            .groupby(["gestational_age_bin", "postnatal_age_bin"], observed=True)
            .size()
            .rename("n")
            .reset_index()
        )
        out = out.loc[out["n"] > 0].sort_values("n", ascending=False)
        out.insert(0, "tail", label)
        return out

    low_tail = _tail_breakdown(df["pca"] < pca_min, f"PCA < {pca_min}")
    high_tail = _tail_breakdown(df["pca"] > pca_max, f"PCA > {pca_max}")
    low_tail.to_csv(outdir / "pca_low_tail_excluded.csv", index=False)
    high_tail.to_csv(outdir / "pca_high_tail_excluded.csv", index=False)

    # --- sensitivity: does >=cutoff vs >cutoff explain a discrepancy? ------
    n_exactly_at_min = int((df["pca"] == pca_min).sum())
    n_exactly_at_max = int((df["pca"] == pca_max).sum())
    n_low_excl = int((df["pca"] < pca_min).sum())
    n_high_excl = int((df["pca"] > pca_max).sum())
    n_high_excl_inclusive = int((df["pca"] >= pca_max).sum())  # what '>=' would exclude instead

    n_sparse_weeks_low = int(weekly.loc[weekly.index < pca_min, "sparse_by_sex"].sum())
    n_sparse_weeks_in_window = int(weekly.loc[weekly["in_chosen_window"], "sparse_by_sex"].sum())
    n_sparse_weeks_high = int(weekly.loc[weekly.index > pca_max, "sparse_by_sex"].sum())

    diag_path = outdir / "pca_boundary_diagnostics.txt"
    with open(diag_path, "w") as f:
        f.write("PCA window boundary diagnostics (for reviewer response)\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Sample at this stage (post first-week/unknown-age cleaning, "
                f"PRE window restriction): n = {n_total}\n\n")

        f.write(f"Chosen window: {pca_min} <= PCA <= {pca_max}\n")
        f.write(f"  Excluded PCA < {pca_min}: n = {n_low_excl} "
                f"({100 * n_low_excl / n_total:.2f}% of n={n_total})\n")
        f.write(f"  Excluded PCA > {pca_max}: n = {n_high_excl} "
                f"({100 * n_high_excl / n_total:.2f}% of n={n_total})\n\n")

        f.write("Boundary-operator sensitivity check ('>' vs '>=') --\n")
        f.write("this matters if a previously reported/published exclusion "
                "count doesn't match this run's output:\n")
        f.write(f"  n with PCA exactly == {pca_max}: {n_exactly_at_max}\n")
        f.write(f"  Current rule excludes PCA > {pca_max}: n_excluded = {n_high_excl}\n")
        f.write(f"  If the upper bound were '>= {pca_max}' instead, "
                f"n_excluded would be {n_high_excl_inclusive} "
                f"(+{n_exactly_at_max}). Compare this to whatever count is "
                f"currently written in the manuscript/Figure 1 — if it matches "
                f"{n_high_excl_inclusive} rather than {n_high_excl}, the "
                f"published figure was generated with an inclusive upper "
                f"bound ('>=') that the code has since changed away from.\n")
        f.write(f"  n with PCA exactly == {pca_min}: {n_exactly_at_min} "
                f"(currently INCLUDED, since the lower bound is '< {pca_min}', "
                f"not '<= {pca_min}')\n\n")

        f.write(f"Symmetric-percentile comparison (reviewer's implicit "
                f"benchmark of ~{tail_pct}% per tail):\n")
        f.write(f"  A symmetric {tail_pct}% trim from each tail of the observed "
                f"PCA distribution would cut at approximately "
                f"PCA <= {lo_cut:.1f} and PCA >= {hi_cut:.1f} weeks.\n")
        f.write(f"  That is NOT close to the chosen {pca_min}/{pca_max} window, "
                f"because the case PCA distribution is right-skewed: SUID risk "
                f"rises quickly after the neonatal period, peaks in the first "
                f"few months, then declines over a much longer tail than it "
                f"rose. Equal WEEKS from the peak does not correspond to equal "
                f"PROBABILITY MASS in each tail, and equal probability mass "
                f"does not correspond to equal weeks. Trimming by week (fixed "
                f"boundary) and trimming by percentile (data-adaptive boundary) "
                f"are two different, defensible choices — but they will not "
                f"agree with each other, and the manuscript should say which "
                f"one it's doing and why, rather than let a reviewer assume "
                f"symmetry was the goal.\n\n")

        f.write(f"Data-driven check of the manuscript's own stated rationale "
                f"('sparse sex-specific counts', min_cell={min_cell}):\n")
        f.write(f"  Weeks BELOW {pca_min} with a sex-specific count < {min_cell}: "
                f"{n_sparse_weeks_low} of {int((weekly.index < pca_min).sum())} weeks\n")
        f.write(f"  Weeks ABOVE {pca_max} with a sex-specific count < {min_cell}: "
                f"{n_sparse_weeks_high} of {int((weekly.index > pca_max).sum())} weeks\n")
        f.write(f"  Weeks INSIDE [{pca_min},{pca_max}] with a sex-specific "
                f"count < {min_cell}: {n_sparse_weeks_in_window} "
                f"(should be 0, or near-0, if the window is doing the job the "
                f"Limitations section says it's doing)\n\n")
        f.write(f"  If the low-tail and high-tail sparse-week counts above are "
                f"very different relative to how many weeks are available on "
                f"each side, that is itself a legitimate, citable reason for "
                f"an asymmetric cutoff — but it should be stated as such (with "
                f"these numbers) in Methods, not left for a reviewer to "
                f"infer.\n\n")

        f.write("See pca_weekly_counts.csv for the full week-by-week table "
                "(n_total/n_male/n_female/sparse flag) — use it to show "
                "reviewers exactly where sex-specific counts drop below the "
                "stability threshold on each side.\n\n")
        f.write("See pca_low_tail_excluded.csv / pca_high_tail_excluded.csv for "
                "who is actually excluded at each boundary, broken down by "
                "gestational age x postnatal age at death — use this to answer "
                "the reviewer's concrete example directly (a 30-week-gestation "
                "infant dying at postnatal weeks 1-6 falls in the '28-31wk' "
                "gestational-age row, '2-4wk' or '5-8wk' postnatal-age row of "
                "pca_low_tail_excluded.csv).\n")

    LOG.info("Wrote PCA boundary diagnostics to %s", diag_path)
    LOG.info(
        "  Low tail: %d excluded (%.2f%%) | High tail: %d excluded (%.2f%%) | "
        "n at exactly PCA=%d: %d",
        n_low_excl, 100 * n_low_excl / n_total,
        n_high_excl, 100 * n_high_excl / n_total,
        pca_max, n_exactly_at_max,
    )


# ---------------------------------------------------------------------------
# Table 1 — sex-stratified subject breakdown
# (adapted from make_suid_table1_by_sex in the notebook; logic unchanged,
# only the output-path bug is fixed)
# ---------------------------------------------------------------------------

def table1_by_sex(
    df: pd.DataFrame,
    *,
    sex_col: str = "sex",
    postnatal_age_col: str = "infage",
    postnatal_age_unit: str = "weeks",
    gestational_age_col: str = "combgest",
    pca_col: str = "pca",
    smoking_col: str = "cig_rec",
    cause_col: str | None = None,
    year_col: str = "dob_yy",
    restrict_pca_range: bool = False,
    pca_min: int = 36,
    pca_max: int = 90,
    include_calendar_time: bool = True,
) -> pd.DataFrame:
    """
    Build "Table 1. Characteristics of SUID cases by infant sex".
    Columns: Characteristic | Overall N = ... | Female N = ... | Male N = ...
    Each cell is "n (%)" with column percentages.
    """
    data = df.copy()
    LOG.info("table1_by_sex: input shape %s", data.shape)

    def _standardize_sex(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().upper()
        if s in {"F", "FEMALE", "0", "0.0"}:
            return "Female"
        if s in {"M", "MALE", "1", "1.0"}:
            return "Male"
        return np.nan

    def _standardize_yes_no(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().upper()
        if s in {"Y", "YES", "1", "1.0", "TRUE", "T"}:
            return "Yes"
        if s in {"N", "NO", "0", "0.0", "FALSE", "F"}:
            return "No"
        return np.nan

    def _fmt_n_pct(n, denom):
        n, denom = int(n), int(denom)
        if denom == 0:
            return f"{n:,} (NA)"
        return f"{n:,} ({100 * n / denom:.1f})"

    rows: list[dict] = []

    def _add_section(label):
        rows.append({"Characteristic": label, overall_col: "", female_col: "", male_col: ""})

    def _add_row(label, mask):
        mask = mask.fillna(False)
        overall_n = int(mask.sum())
        female_n = int((mask & female_mask).sum())
        male_n = int((mask & male_mask).sum())
        rows.append(
            {
                "Characteristic": label,
                overall_col: _fmt_n_pct(overall_n, overall_denom),
                female_col: _fmt_n_pct(female_n, female_denom),
                male_col: _fmt_n_pct(male_n, male_denom),
            }
        )

    def _find_cause_col(dataframe):
        if cause_col is not None and cause_col in dataframe.columns:
            return cause_col
        for col in ("cause_group", "icd10", "icd_10", "ucod", "underlying_cause",
                    "underlying_cause_code", "cod", "cause", "cause_code"):
            if col in dataframe.columns:
                vals = dataframe[col].astype(str).str.upper()
                if vals.str.contains("R95|W75|R99", regex=True, na=False).any():
                    return col
        return None

    def _cause_mask(code, col):
        vals = data[col].astype(str).str.upper().str.strip()
        return vals.str.contains(code, regex=False, na=False)

    if sex_col not in data.columns:
        raise ValueError(f"Missing required sex column: {sex_col}")

    data["_table1_sex"] = data[sex_col].apply(_standardize_sex)
    n_unrecognized_sex = data["_table1_sex"].isna().sum()
    if n_unrecognized_sex:
        LOG.warning("%d rows had an unrecognized sex value and will be dropped "
                    "from Table 1", n_unrecognized_sex)
    data = data.loc[data["_table1_sex"].isin(["Female", "Male"])].copy()
    if data.empty:
        raise ValueError("No valid Female/Male rows found after standardizing sex.")

    if pca_col not in data.columns:
        if gestational_age_col in data.columns and postnatal_age_col in data.columns:
            if postnatal_age_unit != "weeks":
                raise ValueError(
                    f"{pca_col} is missing and can only be constructed from "
                    "gestational age + postnatal age when postnatal_age_unit='weeks'."
                )
            data[pca_col] = data[gestational_age_col] + data[postnatal_age_col]
        else:
            raise ValueError(
                f"{pca_col} is missing and cannot be constructed because "
                f"{gestational_age_col} and/or {postnatal_age_col} are missing."
            )

    if restrict_pca_range:
        data = data.loc[(data[pca_col] >= pca_min) & (data[pca_col] <= pca_max)].copy()

    if smoking_col in data.columns:
        data["_table1_smoking"] = data[smoking_col].apply(_standardize_yes_no)

    inferred_cause_col = _find_cause_col(data)

    overall_denom = len(data)
    female_mask = data["_table1_sex"] == "Female"
    male_mask = data["_table1_sex"] == "Male"
    female_denom = int(female_mask.sum())
    male_denom = int(male_mask.sum())

    overall_col = f"Overall N = {overall_denom:,}"
    female_col = f"Female N = {female_denom:,}"
    male_col = f"Male N = {male_denom:,}"

    _add_section("Case counts")
    _add_row("Total SUID cases", pd.Series(True, index=data.index))

    if inferred_cause_col is not None:
        _add_section("Cause of death, ICD-10")
        _add_row("R95, Sudden infant death syndrome", _cause_mask("R95", inferred_cause_col))
        _add_row("W75, Accidental suffocation and strangulation in bed", _cause_mask("W75", inferred_cause_col))
        _add_row("R99, Other ill-defined and unspecified causes of mortality", _cause_mask("R99", inferred_cause_col))
    else:
        LOG.warning("No cause-of-death column found/inferred; skipping that Table 1 section.")

    if postnatal_age_col in data.columns:
        age = pd.to_numeric(data[postnatal_age_col], errors="coerce")
        _add_section("Postnatal age at death")
        if postnatal_age_unit.lower() == "days":
            _add_row("7-27 days", (age >= 7) & (age <= 27))
            _add_row("28 days to <2 months", (age >= 28) & (age < 61))
            _add_row("2 to <4 months", (age >= 61) & (age < 122))
            _add_row("4 to <6 months", (age >= 122) & (age < 183))
            _add_row("6 to <12 months", (age >= 183) & (age < 365))
        elif postnatal_age_unit.lower() == "weeks":
            weeks_per_month = 365.25 / 12 / 7
            _add_row("7-27 days", (age >= 1) & (age < 4))
            _add_row("28 days to <2 months", (age >= 4) & (age < 2 * weeks_per_month))
            _add_row("2 to <4 months", (age >= 2 * weeks_per_month) & (age < 4 * weeks_per_month))
            _add_row("4 to <6 months", (age >= 4 * weeks_per_month) & (age < 6 * weeks_per_month))
            _add_row("6 to <12 months", (age >= 6 * weeks_per_month) & (age < 12 * weeks_per_month))
        else:
            raise ValueError("postnatal_age_unit must be either 'weeks' or 'days'.")

    if pca_col in data.columns:
        pca = pd.to_numeric(data[pca_col], errors="coerce")
        _add_section("Postconceptional age window")
        _add_row("36-42 weeks, early rising window", (pca >= 36) & (pca <= 42))
        _add_row("43-46 weeks, peak/transition interval", (pca >= 43) & (pca <= 46))
        _add_row(">=47 weeks, late declining window", pca >= 47)

    if gestational_age_col in data.columns:
        ga = pd.to_numeric(data[gestational_age_col], errors="coerce")
        _add_section("Gestational age at birth")
        _add_row("Preterm, <37 weeks", ga < 37)
        _add_row("Term, >=37 weeks", ga >= 37)

    if smoking_col in data.columns:
        _add_section("Maternal smoking during pregnancy")
        _add_row("Yes", data["_table1_smoking"] == "Yes")
        _add_row("No", data["_table1_smoking"] == "No")

    if include_calendar_time and year_col in data.columns:
        year = pd.to_numeric(data[year_col], errors="coerce")
        _add_section("Year of birth or death")
        _add_row("2014-2015", (year >= 2014) & (year <= 2015))
        _add_row("2016-2017", (year >= 2016) & (year <= 2017))
        _add_row("2018-2019", (year >= 2018) & (year <= 2019))
        _add_row("2020-2021", (year >= 2020) & (year <= 2021))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison: corrected vs. original buggy filter
# ---------------------------------------------------------------------------

def compare_corrected_vs_buggy(
    raw_df: pd.DataFrame,
    outdir: Path,
    **clean_kwargs,
) -> None:
    corrected_df, _, corrected_trail = clean_case_series(
        raw_df, reproduce_original_bug=False, **clean_kwargs
    )
    buggy_df, _, buggy_trail = clean_case_series(
        raw_df, reproduce_original_bug=True, **clean_kwargs
    )

    corrected_trail.to_frame().to_csv(outdir / "audit_trail_corrected.csv", index=False)
    buggy_trail.to_frame().to_csv(outdir / "audit_trail_original_buggy.csv", index=False)

    delta_n = len(buggy_df) - len(corrected_df)
    raw_total = corrected_trail.total_n or len(raw_df)  # 100% baseline, both trails agree
    pct_buggy = 100 * len(buggy_df) / raw_total
    pct_corrected = 100 * len(corrected_df) / raw_total
    pct_delta_of_raw = 100 * delta_n / raw_total

    LOG.info("=" * 78)
    LOG.info("CORRECTED vs. ORIGINAL BUGGY FIRST-WEEK FILTER")
    LOG.info("  Raw total:          n = %d (100%%)", raw_total)
    LOG.info("  Original (buggy):   n = %d (%.1f%% of raw)", len(buggy_df), pct_buggy)
    LOG.info("  Corrected:          n = %d (%.1f%% of raw)", len(corrected_df), pct_corrected)
    LOG.info(
        "  Extra subjects retained by the bug (should have been excluded): "
        "%d (%.1f%% of raw total)",
        delta_n, pct_delta_of_raw,
    )
    LOG.info("=" * 78)

    t1_corrected = table1_by_sex(corrected_df)
    t1_buggy = table1_by_sex(buggy_df)
    t1_corrected.to_csv(outdir / "table1_by_sex_corrected.csv", index=False)
    t1_buggy.to_csv(outdir / "table1_by_sex_original_buggy.csv", index=False)

    summary_path = outdir / "comparison_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Comparison: corrected first-week exclusion vs. original notebook bug\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Raw total                   : {raw_total} (100.0%)\n")
        f.write(f"Original (buggy) analytic n : {len(buggy_df)} ({pct_buggy:.1f}% of raw)\n")
        f.write(f"Corrected analytic n        : {len(corrected_df)} ({pct_corrected:.1f}% of raw)\n")
        f.write(f"Subjects that should have been excluded (infage<=1wk) "
                f"but were not, in the original pipeline: {delta_n} "
                f"({pct_delta_of_raw:.1f}% of raw total)\n\n")
        f.write("See table1_by_sex_corrected.csv vs. table1_by_sex_original_buggy.csv "
                "for the full side-by-side subject breakdown.\n")
    LOG.info("Wrote comparison summary to %s", summary_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit and report the subject breakdown for the SIDS/SUID case series.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                    help="Path to the raw case file (.xlsx, .parquet, or .csv), "
                         "e.g. matched_cases_cdc.xlsx")
    p.add_argument("--outdir", type=Path, default=Path("./audit_output"),
                    help="Directory to write audit trail / Table 1 outputs into")
    p.add_argument("--inspect-only", action="store_true",
                    help="Print columns/dtypes/head of the input and exit, without "
                         "running any cleaning. Use this first to confirm column names.")
    p.add_argument("--compare-buggy", action="store_true",
                    help="Also run the original (buggy) first-week filter and write a "
                         "side-by-side comparison report.")
    p.add_argument("--skip-boundary-report", action="store_true",
                    help="Skip the PCA-window boundary diagnostics (pca_weekly_counts.csv, "
                         "pca_low_tail_excluded.csv, pca_high_tail_excluded.csv, "
                         "pca_boundary_diagnostics.txt). Runs by default.")
    p.add_argument("--skip-figure1-reconciliation", action="store_true",
                    help="Skip reproducing the published Figure 1 (validated against the "
                         "exact published numbers) and the corrected case-selection flow "
                         "(invalid-smoking exclusion + early/late window split). Runs by "
                         "default.")

    p.add_argument("--infage-col", default="infage")
    p.add_argument("--combgest-col", default="combgest")
    p.add_argument("--sex-col", default="sex")
    p.add_argument("--smoking-col", default="cig_rec")
    p.add_argument("--year-col", default="dob_yy")
    p.add_argument("--cause-col", default=None,
                    help="Explicit cause-of-death column name; auto-detected if omitted.")
    p.add_argument("--pca-min", type=int, default=36)
    p.add_argument("--pca-max", type=int, default=90,
                    help="Upper PCA bound. Raised from the original 75 to 90 based on "
                         "pca_boundary_diagnostics.txt: PCA 34-90 is the contiguous block "
                         "where sex-specific weekly counts stay >= min_cell, so 75 was "
                         "leaving usable, non-sparse data on the table. Override with "
                         "--pca-max 75 to reproduce the original window.")
    p.add_argument("--min-cell", type=int, default=5,
                    help="Minimum sex-specific weekly count considered 'stable' for the "
                         "PCA-window boundary diagnostics (matches the notebook's own "
                         "get_sex() suppression threshold).")
    p.add_argument("--tail-pct", type=float, default=1.5,
                    help="Percentage used for the symmetric-trim comparison in the PCA "
                         "boundary diagnostics.")
    p.add_argument("--peak1", type=int, default=43,
                    help="Upper edge (exclusive) of the early analytic window (PCA < peak1).")
    p.add_argument("--peak2", type=int, default=47,
                    help="Lower edge (inclusive) of the late analytic window (PCA >= peak2). "
                         "PCA in [peak1, peak2) is the descriptive-only peak interval, "
                         "excluded from both windowed models.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.outdir / "audit_run.log"),
        ],
    )

    if not args.input.exists():
        LOG.error("Input file not found: %s", args.input)
        return 1

    LOG.info("Loading %s", args.input)
    raw_df = read_any(args.input)
    LOG.info("Raw shape: %s", raw_df.shape)

    if args.inspect_only:
        inspect(raw_df)
        return 0

    clean_kwargs = dict(
        infage_col=args.infage_col,
        combgest_col=args.combgest_col,
        pca_min=args.pca_min,
        pca_max=args.pca_max,
    )

    if args.compare_buggy:
        compare_corrected_vs_buggy(raw_df, args.outdir, **clean_kwargs)

    case_df, pre_window_df, trail = clean_case_series(
        raw_df, reproduce_original_bug=False, **clean_kwargs
    )
    trail.to_frame().to_csv(args.outdir / "audit_trail.csv", index=False)

    if not args.skip_boundary_report:
        pca_boundary_report(
            pre_window_df,
            args.outdir,
            sex_col=args.sex_col,
            combgest_col=args.combgest_col,
            infage_col=args.infage_col,
            pca_min=args.pca_min,
            pca_max=args.pca_max,
            min_cell=args.min_cell,
            tail_pct=args.tail_pct,
        )

    table1 = table1_by_sex(
        case_df,
        sex_col=args.sex_col,
        postnatal_age_col=args.infage_col,
        gestational_age_col=args.combgest_col,
        smoking_col=args.smoking_col,
        year_col=args.year_col,
        cause_col=args.cause_col,
    )
    table1_csv = args.outdir / "table1_by_sex.csv"
    table1_xlsx = args.outdir / "table1_by_sex.xlsx"
    table1.to_csv(table1_csv, index=False)
    table1.to_excel(table1_xlsx, index=False)

    if not args.skip_figure1_reconciliation:
        figure1_reconciliation(
            raw_df,
            args.outdir,
            infage_col=args.infage_col,
            combgest_col=args.combgest_col,
            smoking_col=args.smoking_col,
            sex_col=args.sex_col,
            pca_min=args.pca_min,
            pca_max=args.pca_max,
            peak1=args.peak1,
            peak2=args.peak2,
        )

    LOG.info("=" * 78)
    LOG.info("DONE. Final analytic sample (post PCA-window only): n=%d", len(case_df))
    LOG.info("  Audit trail : %s", args.outdir / "audit_trail.csv")
    LOG.info("  Table 1 (csv)  : %s", table1_csv)
    LOG.info("  Table 1 (xlsx) : %s", table1_xlsx)
    if not args.skip_boundary_report:
        LOG.info("  PCA boundary diagnostics: %s", args.outdir / "pca_boundary_diagnostics.txt")
        LOG.info("  PCA weekly counts       : %s", args.outdir / "pca_weekly_counts.csv")
        LOG.info("  PCA low/high tail cases : %s / %s",
                 args.outdir / "pca_low_tail_excluded.csv",
                 args.outdir / "pca_high_tail_excluded.csv")
    if not args.skip_figure1_reconciliation:
        LOG.info("  Figure 1 reproduction check (validates against published numbers): %s",
                 args.outdir / "figure1_reproduction_check.txt")
        LOG.info("  Corrected case-selection flow (TRUE final analytic N): %s",
                 args.outdir / "figure1_corrected.txt")
        LOG.info("  Table 1 on the true final analytic series: %s",
                 args.outdir / "table1_by_sex_final_analytic.csv")
    if args.compare_buggy:
        LOG.info("  Comparison report: %s", args.outdir / "comparison_summary.txt")
    LOG.info("=" * 78)

    print("\n" + table1.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

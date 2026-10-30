#!/usr/bin/env python3
"""
ncfrp_manuscript_v5.py

NCFRP SIDS/SUID developmental manuscript pipeline (bed-sharing / sleep environment).

Core model:
    stratified weekly counts over PCA x bed_share x smoke x preterm (cases only,
    no zero-count rows), fed to
      1) piecewise negative binomial models (early rising / late declining window),
         with the peak interval excluded from the regression and summarized
         descriptively (two-population design, matching manuscript 1)
      2) a full-curve negative binomial spline GAM across PCA
    run separately for: all infants, females only, males only.

Also: descriptive chi-square smoke x bed-share co-occurrence AMONG CASES.

CHANGES vs. the archived version (see manuscript_2_bedsharing_pca/AUDIT_FINDINGS.md
for the full rationale and the manuscript-1 reference behaviour each maps to):

  A1.6  docstring filename corrected (was "v4"); duplicate numpy/pandas imports
        removed; a single random seed is set in main().
  A9    ONE case loader (load_ncfrp_data) that records an explicit CONSORT-style
        AuditTrail for every exclusion and writes it to CSV on every run. The old
        second loader (NaN-poison + terminal dropna, no trail) is gone. main()
        always emits output_data/ncfrp_flow_audit.csv.
  A2.10 / G3  gestational age and infant age: per the NCFRP codebook (T1) both
        "default to blank", so unknown values are MISSING (NaN), not numeric
        sentinels, and are dropped by the notna() steps (plus inf3gestunk == 1 if
        present). Implausible values (a decode error) are EXCLUDED, not recoded:
        inf3gestage > GESTAGE_MAX (47) or < GESTAGE_MIN (20), infagedays >=
        MAX_INFAGE_DAYS (365). A documented cap is kept only for GENUINE
        post-term gestation (> POSTTERM_CAP = 42 -> 42; previously everything
        > 42, sentinels included, was collapsed to 40 = "term").
  A3.21 / G4  the negative binomial and GAM models are fit on RAW INTEGER counts.
        The pre-fit rolling-mean of stratum counts is removed. Rolling means are
        used only for plotted descriptive curves (smooth_rate_table), as in
        manuscript 1.
  A3.23 the piecewise design now carries prespecified interaction terms
        (bed_share x smoke, bed_share x preterm), reported in every IRR table,
        mirroring manuscript 1's smoke x preterm / sex x preterm terms and
        reviewer 1's request for complete interaction reporting.
  A3.25 the late-window upper truncation (peak3) defaults to PCA_MAX and the
        late window is now inclusive of PCA == peak3 (>= peak2 and <= peak3),
        matching manuscript 1's convention. The stale peak3 = 100 is gone.
  A4.30 every figure and table reads IRRs from ONE saved coefficient table per
        model; an assertion checks that each reported CI brackets its point
        estimate before anything is written.
  A4.31 the FEMALES and MALES runs are enabled (were commented out).
  A4.32 Figure 3 TIFF is written with format="tiff"; Figure 4 is saved with
        bbox_inches="tight" so the forest-plot row labels are not clipped;
        Figure 3 panel B is a fitted rate-ratio curve (was a raw count
        difference with a crude lo_bs - hi_ns CI band; Ed E10).
  Ed E7 Figure 2 panels A and B share a y-axis range.

STILL OUTSTANDING (need the HPC / data / a user decision - see AUDIT_FINDINGS.md):
  A1.1/A1.2  the case-selection ("SIDS") code and the standardization step that
             produces standardized/NCFRP_SCH_LR_BW_pop.csv are not in the repo.
  A3.17      PCA_MIN / PCA_MAX should be set from a count-stability boundary
             report (pca_boundary_report_ncfrp.py), not hardcoded.
  A3.19/G2   the multiplicative offset from marginal exposure prevalences should
             be replaced by a joint exposure distribution where the denominator
             data allow. Left as-is here; flagged at the offset-construction site.
  A3.22      MIN_COUNT (per-stratum threshold) is kept at 10 to preserve the
             currently-written analytic choice; manuscript 1 uses no threshold.
             Quantify what it removes, then decide.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import forestplot as fp

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.gam.generalized_additive_model import GLMGam
from statsmodels.gam.smooth_basis import BSplines
from statsmodels.genmod.families import NegativeBinomial as NB
from scipy.stats import chi2_contingency
from scipy.linalg import qr as _qr

LOG = logging.getLogger("ncfrp_manuscript")

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)


# -----------------------------------------------------------------------------
# Global settings (CLI-overridable in main())
# -----------------------------------------------------------------------------

# Directory the process was launched from, captured before main() chdir's into
# --outdir. Input trees (standardized/...) are relative to this, not to --outdir.
ORIG_CWD = Path.cwd()

DATA_PATH = Path("standardized/NCFRP_SCH_LR_BW_pop.csv")
EXPOSURE_DIR = Path("output_data/risk_exposure")
FIG_DIR = Path("output_figures/ncfrp")
GAM_FIG_DIR = FIG_DIR / "gam"
OUT_DATA_DIR = Path("output_data/ncfrp")
AUDIT_CSV = Path("output_data/ncfrp_flow_audit.csv")

SEED = 20260902

MIN_COUNT = 10          # per-stratum-week count threshold. A3.22: M1 uses none.
PCA_MIN = 36            # A3.17: should come from pca_boundary_report_ncfrp.py
PCA_MAX = 75            # A3.17: M1 widened its equivalent bound to 90
PEAK1 = 43             # early window is PCA < PEAK1
PEAK2 = 47             # late window is PEAK2 <= PCA <= peak3
# T3: NCFRP_SCH_pop.csv is NOT year-restricted (verified: 24,314 rows, INFdodyear
# from 2004 on), and the archived analysis read that file, so the default is NO
# year filter (reproduces the manuscript's 24,314 starting N). preprocess's
# select_sample() restricts _sample.csv to 2008-2019; pass --year-min 2008
# --year-max 2019 here to match that, and Figure 1's start N changes accordingly.
YEAR_MIN = None
YEAR_MAX = None
FIRST_WEEK_CUT = 1     # exclude infageweeks <= FIRST_WEEK_CUT (matches M1's infage<=1). G5 pending.
# Display-only centred moving-average windows for Figure 2 (A3.21/G4: the NB and
# GAM models are always fit on raw integer counts; smoothing is visualization
# only). Set from --smooth-window-pna / --smooth-window-pca in main().
SMOOTH_WINDOW_PNA = 12
SMOOTH_WINDOW_PCA = 5
# T1 (codebook): NCFRP INF3gestage "defaults to blank" - unknown gestational age
# is MISSING (NaN), not a numeric sentinel, and is dropped by the notna() step.
# GESTAGE_MAX is just an implausible-value guard (no human gestation is 99 weeks;
# if a decode error produced one it must not become a valid "term" value).
GESTAGE_MAX = 47       # inf3gestage > this is implausible -> EXCLUDED (was: recoded to 40)
GESTAGE_MIN = 20       # implausibly low gestational age -> EXCLUDED
POSTTERM_CAP = 42      # genuine post-term gestation capped to this (recode only, not a drop)
# T1 (codebook): INFage also "defaults to blank"; there is no numeric unknown
# code. MAX_INFAGE_DAYS is an implausible-value guard: an infant death is within
# the first year, and the PCA window keeps this well inside that anyway.
MAX_INFAGE_DAYS = 365


# -----------------------------------------------------------------------------
# Audit trail (ported from manuscript_1_sex_pca/src/audit_subject_breakdown.py)
# -----------------------------------------------------------------------------

@dataclass
class AuditTrail:
    """Ordered, CONSORT-style record of n at each cleaning step."""

    label: str = "ncfrp_case_selection"
    steps: list = field(default_factory=list)
    total_n: int | None = None

    def record(self, step_label: str, df: pd.DataFrame, note: str = "") -> None:
        n = len(df)
        prev_n = self.steps[-1]["n_after"] if self.steps else None
        removed = (prev_n - n) if prev_n is not None else 0
        if self.total_n is None:
            self.total_n = n
        pct_removed = 100 * removed / self.total_n if self.total_n else 0.0
        pct_retained = 100 * n / self.total_n if self.total_n else 0.0
        self.steps.append({
            "step": step_label,
            "n_before": prev_n if prev_n is not None else n,
            "n_after": n,
            "n_removed": removed,
            "pct_removed_of_total": round(pct_removed, 2),
            "pct_retained_of_total": round(pct_retained, 2),
            "note": note,
        })
        LOG.info("[%-48s] n=%7d  (-%d, -%.1f%% of total)  %s",
                 step_label, n, removed, pct_removed, note)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def ensure_dirs() -> None:
    for d in (FIG_DIR, GAM_FIG_DIR, OUT_DATA_DIR, AUDIT_CSV.parent):
        d.mkdir(parents=True, exist_ok=True)


def load_pickle(path: Path):
    path = Path(path)
    if not path.is_absolute():
        path = ORIG_CWD / path       # exposure inputs live under the launch dir,
    with open(path, "rb") as f:      # not under --outdir (main() chdir's there)
        return pickle.load(f)


def load_exposure(exposure_dir: Path = EXPOSURE_DIR):
    """
    Marginal exposure prevalence dicts (built by src/risk_exposure_joint.py):
        smoke_exposure   : {0: nonsmoker, 1: smoker}
        sex_exposure     : {0: female,    1: male}
        term_exposure    : {0: term,      1: preterm}
        bedshare_exposure: {0: non-bed-share, 1: bed-share}
    """
    return (
        load_pickle(Path(exposure_dir) / "smoke_exposure.pkl"),
        load_pickle(Path(exposure_dir) / "sex_exposure.pkl"),
        load_pickle(Path(exposure_dir) / "term_exposure.pkl"),
        load_pickle(Path(exposure_dir) / "bedshare_exposure.pkl"),
    )


def load_joint_exposure(exposure_dir: Path = EXPOSURE_DIR):
    """
    G2: joint P(smoke, preterm) from the CDC live-birth denominator (dict keyed
    by (smoke, preterm) tuples) plus the marginal bed-share prevalence from
    PRAMS. The stratum offset is joint[(smoke, preterm)] * bedshare[bed_share] -
    smoking x preterm are no longer assumed independent in the population.
    Build with src/risk_exposure_joint.py.
    """
    joint = load_pickle(Path(exposure_dir) / "joint_smoke_preterm_exposure.pkl")
    bedshare = load_pickle(Path(exposure_dir) / "bedshare_exposure.pkl")
    return joint, bedshare


# -----------------------------------------------------------------------------
# A9 - single, audited case loader
# -----------------------------------------------------------------------------

REQUIRED_COLS = ["infagedays", "inf3gestage", "smoker", "infsex", "bed_sharing"]


def load_ncfrp_data(
    data_path: Path = DATA_PATH,
    *,
    pca_min: int = PCA_MIN,
    pca_max: int = PCA_MAX,
    first_week_cut: int = FIRST_WEEK_CUT,
    gestage_max: int = GESTAGE_MAX,
    max_infage_days: int = MAX_INFAGE_DAYS,
    gestage_min: int = GESTAGE_MIN,
    postterm_cap: int = POSTTERM_CAP,
    year_min: int | None = YEAR_MIN,
    year_max: int | None = YEAR_MAX,
    audit_csv: Path | None = AUDIT_CSV,
    apply_window: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the analytic case series with an explicit exclusion at each step and a
    CONSORT-style audit trail. Returns (clean_df, audit_frame).

    T1 (codebook): NCFRP INF3gestage and INFage both "default to blank", so
    unknown values are MISSING (NaN), not numeric sentinel codes. The notna()
    steps handle the real unknowns; the gestage_max / max_infage_days bounds are
    implausible-value guards so a decode error can never masquerade as a valid
    term gestation or infant age. If the standardized file carries an
    INF3gestunk flag it is also honoured.

    Steps, each applied to the survivors of the previous step:
      0  raw load
      1  drop missing postnatal age (infagedays)
      2  exclude implausible infant age (infagedays >= max_infage_days)
      3  exclude first postnatal week(s) (infageweeks <= first_week_cut)       [A2.8/G5]
      4  drop missing gestational age (inf3gestage NaN, or inf3gestunk == 1)
      5  exclude implausible gestational age (inf3gestage > gestage_max)       [A2.10/G3]
      6  exclude implausibly low gestational age (inf3gestage < gestage_min)
      7  cap genuine post-term gestation (inf3gestage > postterm_cap -> cap)   [recode only, G3]
      8  compute PCA = infageweeks + inf3gestage
      9  exclude PCA < pca_min
      10 exclude PCA > pca_max
      11 exclude missing/invalid maternal smoking (cig_rec not in {Y, N})
      12 exclude missing/invalid infant sex
      13 exclude missing/invalid bed sharing
      14 final safeguard: any remaining missing analytic field
    """
    trail = AuditTrail()

    df = pd.read_csv(data_path, low_memory=False)
    df.columns = df.columns.str.lower()
    # drop a stray unnamed index column if preprocess wrote one (A2.15 upstream)
    df = df.loc[:, ~df.columns.str.match(r"unnamed: \d+$")]
    trail.record("0. raw load", df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present: {list(df.columns)}")

    # 0b. restrict to the NCFRP death-year window (T3: preprocess's select_sample
    # applied 2008-2019; the standardized file may or may not already be
    # restricted, so do it explicitly and audibly here if the column is present).
    if (year_min is None) != (year_max is None):
        LOG.warning("Only one of year_min/year_max given (min=%s, max=%s); the "
                    "death-year restriction needs BOTH and is being SKIPPED - the "
                    "cohort is NOT year-restricted.", year_min, year_max)
    if year_min is not None and year_max is not None and "infdodyear" in df.columns:
        df = df.loc[df["infdodyear"].between(year_min, year_max)].copy()
        trail.record(f"0b. restrict to death years {year_min}-{year_max}", df)

    # 1. missing postnatal age
    df = df.loc[df["infagedays"].notna()].copy()
    trail.record("1. drop missing infagedays", df)

    # 2. implausible infant age (a decode error, e.g. INFagecat=Years mishandled)
    df = df.loc[df["infagedays"] < max_infage_days].copy()
    trail.record(f"2. exclude implausible infant age (infagedays >= {max_infage_days})", df)

    df["infageweeks"] = np.floor(df["infagedays"] / 7)

    # 3. first postnatal week(s)  (A2.8 - matches M1's infage <= 1)
    df = df.loc[df["infageweeks"] > first_week_cut].copy()
    trail.record(f"3. exclude first postnatal week(s) (infageweeks <= {first_week_cut})", df)

    # 4. missing / flagged-unknown gestational age
    if "inf3gestunk" in df.columns:
        df = df.loc[df["inf3gestunk"] != 1].copy()
        trail.record("4a. exclude gestational age flagged unknown (inf3gestunk == 1)", df)
    df = df.loc[df["inf3gestage"].notna()].copy()
    trail.record("4. drop missing gestational age", df)

    # 5. implausible gestational age (was silently recoded to 40 - A2.10/G3)
    df = df.loc[df["inf3gestage"] <= gestage_max].copy()
    trail.record(f"5. exclude implausible gestational age (inf3gestage > {gestage_max})", df)

    # 6. implausibly low gestational age
    df = df.loc[df["inf3gestage"] >= gestage_min].copy()
    trail.record(f"6. exclude gestational age < {gestage_min}", df)

    # 7. cap genuine post-term (recode only, not a row drop; G3 - was -> 40)
    n_capped = int((df["inf3gestage"] > postterm_cap).sum())
    df.loc[df["inf3gestage"] > postterm_cap, "inf3gestage"] = postterm_cap
    trail.record(f"7. cap post-term gestation > {postterm_cap} -> {postterm_cap}", df,
                 note=f"recode only, {n_capped} rows capped")

    # 8. PCA
    df["pca"] = df["infageweeks"] + df["inf3gestage"]

    # 9-10. PCA window. apply_window=False (used by pca_boundary_report_ncfrp.py)
    # leaves the full observed PCA range so the boundary itself can be diagnosed.
    if apply_window:
        df = df.loc[df["pca"] >= pca_min].copy()
        trail.record(f"9. exclude PCA < {pca_min}", df)
        df = df.loc[df["pca"] <= pca_max].copy()
        trail.record(f"10. exclude PCA > {pca_max}", df)

    # 11. maternal smoking
    df["cig_rec"] = pd.Series(np.nan, index=df.index, dtype=object)  # object: str codes assigned below (pandas>=3 safe)
    df.loc[df["smoker"] == 1, "cig_rec"] = "Y"
    df.loc[df["smoker"] == 2, "cig_rec"] = "N"
    df["smoke"] = np.nan
    df.loc[df["cig_rec"] == "Y", "smoke"] = 1
    df.loc[df["cig_rec"] == "N", "smoke"] = 0
    df = df.loc[df["cig_rec"].isin(["Y", "N"])].copy()
    trail.record("11. exclude missing/invalid maternal smoking", df)

    # 12. infant sex
    df["sex"] = pd.Series(np.nan, index=df.index, dtype=object)  # object: str codes assigned below (pandas>=3 safe)
    df.loc[df["infsex"] == 1, "sex"] = "M"
    df.loc[df["infsex"] == 2, "sex"] = "F"
    df = df.loc[df["sex"].isin(["M", "F"])].copy()
    trail.record("12. exclude missing/invalid infant sex", df)

    # 13. bed sharing
    df["bed_share"] = pd.Series(np.nan, index=df.index, dtype=object)  # object: str codes assigned below (pandas>=3 safe)
    df.loc[df["bed_sharing"] == 1, "bed_share"] = "Y"
    df.loc[df["bed_sharing"] == 2, "bed_share"] = "N"
    df["bed_share_bin"] = np.nan
    df.loc[df["bed_share"] == "Y", "bed_share_bin"] = 1
    df.loc[df["bed_share"] == "N", "bed_share_bin"] = 0
    df = df.loc[df["bed_share"].isin(["Y", "N"])].copy()
    trail.record("13. exclude missing/invalid bed sharing", df)

    # preterm (all survivors now have a valid, non-sentinel gestational age)
    df["preterm"] = np.where(df["inf3gestage"] < 37, "Y", "N")
    df["preterm_bin"] = (df["preterm"] == "Y").astype(int)

    # low birthweight (<2500 g) for the birthweight-stratified / interaction
    # analyses (lab action item 1). inf3biwt is the LR_BW-imputed column, present
    # only when the pipeline is fed standardized/NCFRP_SCH_LR_BW_pop.csv. NOT a
    # covariate in the primary models; NOT in keep_cols, so it never drops a row.
    if "inf3biwt" in df.columns:
        _bw = pd.to_numeric(df["inf3biwt"], errors="coerce")
        df["lbw_bin"] = (_bw < 2500).astype(float)
        df.loc[_bw.isna(), "lbw_bin"] = np.nan
        df["lbw"] = df["lbw_bin"].map({1.0: "Y", 0.0: "N"})

    # 14. final safeguard
    keep_cols = ["pca", "infageweeks", "inf3gestage", "cig_rec", "smoke",
                 "sex", "bed_share", "bed_share_bin", "preterm", "preterm_bin"]
    before = len(df)
    df = df.dropna(subset=keep_cols).copy()
    trail.record("14. final safeguard: drop any remaining missing analytic field", df,
                 note=f"{before - len(df)} rows")

    df["pca"] = df["pca"].astype(int)
    df["smoke"] = df["smoke"].astype(int)
    df["bed_share_bin"] = df["bed_share_bin"].astype(int)
    df["preterm_bin"] = df["preterm_bin"].astype(int)

    audit = trail.to_frame()
    if audit_csv is not None:
        Path(audit_csv).parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_csv, index=False)
        LOG.info("Wrote case-selection audit trail to %s", audit_csv)

    return df.reset_index(drop=True), audit


# -----------------------------------------------------------------------------
# Two-population split (descriptive/GAM cohort vs windowed-regression cohort)
# A3.18 - report all four Ns explicitly.
# -----------------------------------------------------------------------------

def split_analytic_windows(df: pd.DataFrame, *, pca_col: str = "pca",
                           peak1: int = PEAK1, peak2: int = PEAK2) -> dict:
    early = df.loc[df[pca_col] < peak1].copy()
    late = df.loc[df[pca_col] >= peak2].copy()
    peak = df.loc[(df[pca_col] >= peak1) & (df[pca_col] < peak2)].copy()
    LOG.info("Two-population split: full analytic N=%d | early(<%d) n=%d | "
             "late(>=%d) n=%d | peak[%d,%d) excluded-from-regression n=%d",
             len(df), peak1, len(early), peak2, len(late), peak1, peak2, len(peak))
    return {"full": df, "early": early, "late": late, "peak": peak}


# -----------------------------------------------------------------------------
# Count structuring (cases only, no zero rows)
# -----------------------------------------------------------------------------

def structure_ncfrp_data(c0: pd.Series, bed_share: int, smoke: int, preterm: int,
                         var1: str) -> pd.DataFrame:
    return pd.DataFrame({
        var1: c0.index.values,
        "count": c0.values,
        "bed_share": bed_share,
        "smoke": smoke,
        "preterm": preterm,
    }).reset_index(drop=True)


def count_ncfrp_data(df_: pd.DataFrame, var1: str, *, min_count: int = MIN_COUNT) -> pd.DataFrame:
    """
    Build the PCA x bed_share x smoke x preterm count table from cases only.
    No zero rows are created. A3.22: min_count is applied ONCE here (M1 applies
    no such threshold); kept at 10 to match the currently-written analytic choice.
    """
    groups = []
    for bed_val, bed_label in [(0, "N"), (1, "Y")]:
        df_bed = df_.loc[df_["bed_share"] == bed_label]
        for smoke_val, smoke_label in [(0, "N"), (1, "Y")]:
            df_smoke = df_bed.loc[df_bed["cig_rec"] == smoke_label]
            for preterm_val, preterm_label in [(0, "N"), (1, "Y")]:
                df_sub = df_smoke.loc[df_smoke["preterm"] == preterm_label]
                counts = df_sub[var1].value_counts().sort_index()
                if len(counts) == 0:
                    continue
                groups.append(structure_ncfrp_data(
                    counts, bed_share=bed_val, smoke=smoke_val,
                    preterm=preterm_val, var1=var1))

    if not groups:
        raise ValueError("No grouped count data were created. Check filtering and coding.")

    input1 = pd.concat(groups, ignore_index=True)
    n_cells, n_cases = len(input1), int(input1["count"].sum())
    input1 = input1.loc[input1["count"] >= min_count].copy()
    # Manuscript-facing number (M1 R2 standard: state what the cell threshold
    # removes): log cells and cases dropped by the min_count rule.
    LOG.info("min_count=%d on %s: cells %d -> %d, cases %d -> %d (dropped %d, %.1f%%)",
             min_count, var1, n_cells, len(input1), n_cases, int(input1["count"].sum()),
             n_cases - int(input1["count"].sum()),
             100.0 * (n_cases - int(input1["count"].sum())) / max(n_cases, 1))
    input1 = input1.sort_values([var1, "bed_share", "smoke", "preterm"]).reset_index(drop=True)
    return input1


def make_ncfrp_input(var1: str, case_df: pd.DataFrame, *, exposure_dir: Path = EXPOSURE_DIR,
                     min_count: int = MIN_COUNT, offset_mode: str = "joint") -> pd.DataFrame:
    """
    Stratified count table + multiplicative exposure offset.

    G2 / A3.19. offset_mode:
      "joint"    (default) - exposure = P_joint(smoke, preterm) [CDC live births]
                 * P(bed_share) [PRAMS]. smoking x preterm are NOT assumed
                 independent. Needs joint_smoke_preterm_exposure.pkl from
                 risk_exposure_joint.py.
      "marginal" - exposure = P(smoke) * P(preterm) * P(bed_share), all
                 marginals. The archived behaviour; kept as a sensitivity
                 comparison.
    """
    input1 = count_ncfrp_data(case_df, var1, min_count=min_count)
    input1 = input1.dropna(subset=[var1]).copy()

    if offset_mode == "joint":
        joint, bedshare_exp = load_joint_exposure(exposure_dir)
        smoke_preterm = list(zip(input1["smoke"].astype(int), input1["preterm"].astype(int)))
        input1["exposure"] = (
            np.array([joint[k] for k in smoke_preterm], dtype=float)
            * input1["bed_share"].map(bedshare_exp).astype(float)
        )
    elif offset_mode == "marginal":
        smoke_exp, _sex_exp, preterm_exp, bedshare_exp = load_exposure(exposure_dir)
        input1["exposure"] = (
            input1["smoke"].map(smoke_exp).astype(float)
            * input1["preterm"].map(preterm_exp).astype(float)
            * input1["bed_share"].map(bedshare_exp).astype(float)
        )
    else:
        raise ValueError(f"offset_mode must be 'joint' or 'marginal', got {offset_mode!r}")

    if (input1["exposure"] <= 0).any() or input1["exposure"].isna().any():
        bad = input1.loc[(input1["exposure"] <= 0) | (input1["exposure"].isna())]
        raise ValueError(f"Non-positive / NaN exposure encountered (offset_mode="
                         f"{offset_mode}):\n{bad}")

    input1["log_exposure"] = np.log(input1["exposure"])
    return input1


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def check_overdispersion(input_: pd.DataFrame, group_col: str | None = None) -> None:
    if group_col is None:
        LOG.info("Overall count mean=%.3f var=%.3f",
                 input_["count"].mean(), input_["count"].var(ddof=1))
        return
    for level, sub in input_.groupby(group_col):
        LOG.info("%s=%s: mean=%.3f var=%.3f", group_col, level,
                 sub["count"].mean(), sub["count"].var(ddof=1))


def smoke_bedshare_case_association(df: pd.DataFrame, out_dir: Path = OUT_DATA_DIR) -> pd.DataFrame:
    """
    A3.24: this is co-occurrence AMONG CASES, not a risk association. Reported as
    such. No denominator is involved.
    """
    ctab = pd.crosstab(df["smoke"], df["bed_share"])
    chi2, p, dof, _ = chi2_contingency(ctab, correction=False)
    LOG.info("Smoke x bed-share CO-OCCURRENCE AMONG CASES:\n%s\nchi2=%.4f dof=%d p=%.6g",
             ctab, chi2, dof, p)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ctab.to_csv(Path(out_dir) / "smoke_bedshare_case_cooccurrence.csv")
    with open(Path(out_dir) / "smoke_bedshare_case_cooccurrence.txt", "w") as f:
        f.write("Smoke x bed-share CO-OCCURRENCE AMONG CASES (not a risk association)\n")
        f.write(f"chi2={chi2:.4f}, dof={dof}, p={p:.6g}\n\n{ctab.to_string()}\n")
    return ctab


# -----------------------------------------------------------------------------
# Model output savers + a single-source IRR table (A4.30)
# -----------------------------------------------------------------------------

def _coef_table(model) -> pd.DataFrame:
    conf = model.conf_int().copy()
    conf.columns = ["coef_ci_low", "coef_ci_high"]
    tab = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "stat": getattr(model, "tvalues", getattr(model, "zvalues", np.nan)),
        "p_value": model.pvalues,
    }).join(conf)
    tab["IRR"] = np.exp(tab["coef"])
    tab["IRR_ci_low"] = np.exp(tab["coef_ci_low"])
    tab["IRR_ci_high"] = np.exp(tab["coef_ci_high"])
    tab.index.name = "term"
    return tab


def _assert_ci_brackets_estimate(tab: pd.DataFrame, label: str) -> None:
    """A4.30 / R1.16-R1.17: every reported CI must contain its point estimate."""
    bad = tab.loc[~((tab["IRR_ci_low"] <= tab["IRR"] + 1e-9)
                    & (tab["IRR"] <= tab["IRR_ci_high"] + 1e-9))]
    if len(bad):
        raise AssertionError(
            f"[{label}] {len(bad)} coefficient CI(s) do not bracket the point "
            f"estimate:\n{bad[['IRR', 'IRR_ci_low', 'IRR_ci_high']]}")


# A3.28: general-purpose pre-fit collinearity/rank-deficiency guard. Added
# alongside the A3.27 close-out (see the retired gam_res_int block, formerly
# here, and gam_bedshare_pca_varycoef.py) after tracing the degenerate
# bed_share x PCA GAM variant to an unidentified design: matching, exploded
# std errors (~329) between the bed_share term and several spline-basis terms
# is the textbook signature of a flat/near-null ridge in the Fisher
# information that no amount of roughness-penalty (CV alpha) selection can
# regularize, because that penalty never ties a parametric term to a smooth
# term's basis. Call this on any new NB/GAM design before trusting a fit.
SE_SANITY_MAX = 20.0  # log-scale std err above this is not a usable estimate


def check_design_rank(X: np.ndarray, colnames: list[str], label: str, *,
                      cond_threshold: float = 1e8) -> None:
    """Log-and-raise pre-fit collinearity/rank check. Cheap relative to any
    NB/GAM fit: O(min(n,p)^2 * max(n,p))."""
    X = np.asarray(X, dtype=float)
    rank = np.linalg.matrix_rank(X)
    cond = np.linalg.cond(X)
    if rank < X.shape[1] or cond > cond_threshold:
        # QR with column pivoting names which columns are implicated.
        _, _, piv = _qr(X, mode="economic", pivoting=True)
        suspect = [colnames[i] for i in piv[rank:]]
        msg = (f"[{label}] design rank={rank}/{X.shape[1]}, cond={cond:.3g} "
              f"(threshold {cond_threshold:.0e}); suspect columns: {suspect}")
        if rank < X.shape[1]:
            raise ValueError(msg)   # hard block: truly singular
        LOG.warning(msg)           # soft warn: ill-conditioned but full rank


def save_nb_model_outputs(model, out_prefix: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    try:
        model.summary2().tables[1].to_csv(f"{out_prefix}_summary.csv")
    except Exception:
        _coef_table(model).to_csv(f"{out_prefix}_summary.csv")
    irr = _coef_table(model)
    _assert_ci_brackets_estimate(irr, out_prefix)
    irr.to_csv(f"{out_prefix}_irr.csv")
    LOG.info("IRR %s\n%s", out_prefix,
             irr[["IRR", "IRR_ci_low", "IRR_ci_high", "p_value"]].to_string())
    return irr


def save_gam_model_outputs(model, out_prefix: str) -> pd.DataFrame:
    """A3.28: this used to write unconditionally - the exact gap that let the
    degenerate gam_res_int variant (SE~329, CI spanning ~560 orders of
    magnitude) reach disk with no warning. Now matches save_nb_model_outputs's
    CI-bracket assertion, plus a max-SE sanity bound (a CI can technically
    bracket its point estimate and still be numerically meaningless)."""
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    try:
        model.summary2().tables[1].to_csv(f"{out_prefix}_summary.csv")
    except Exception:
        _coef_table(model).to_csv(f"{out_prefix}_summary.csv")
    gam_exp = _coef_table(model)
    _assert_ci_brackets_estimate(gam_exp, out_prefix)
    huge = gam_exp.loc[gam_exp["std_err"] > SE_SANITY_MAX]
    if len(huge):
        raise AssertionError(
            f"[{out_prefix}] {len(huge)} term(s) have std_err > {SE_SANITY_MAX} "
            f"(likely rank-deficient design):\n{huge[['coef', 'std_err']]}")
    gam_exp.to_csv(f"{out_prefix}_exp.csv")
    LOG.info("GAM exp(coef) %s\n%s", out_prefix,
             gam_exp[["IRR", "IRR_ci_low", "IRR_ci_high", "p_value"]].to_string())
    return gam_exp


# -----------------------------------------------------------------------------
# Denominator smoothing helper (DISPLAY ONLY - A3.21/G4)
# -----------------------------------------------------------------------------

def get_stratum_offset_from_input(input1: pd.DataFrame, bed_share: int, smoke: int,
                                  preterm: int) -> float:
    sub = input1.loc[(input1["bed_share"] == bed_share) & (input1["smoke"] == smoke)
                     & (input1["preterm"] == preterm)]
    if sub.empty:
        raise ValueError(f"No rows for stratum bed_share={bed_share}, smoke={smoke}, "
                         f"preterm={preterm}")
    vals = sub["log_exposure"].dropna().unique()
    if len(vals) == 0:
        raise ValueError(f"No log_exposure for stratum bed_share={bed_share}, "
                         f"smoke={smoke}, preterm={preterm}")
    if len(vals) > 1:
        LOG.warning("Multiple log_exposure values for stratum bed_share=%d smoke=%d "
                    "preterm=%d: %s. Using the first.", bed_share, smoke, preterm, vals)
    return float(vals[0])


def smooth_rate_table(rate_df: pd.DataFrame, xvar: str, window: int = 3,
                      min_periods: int = 1, center: bool = True) -> pd.DataFrame:
    """Rolling mean of PLOTTED descriptive rates only. Not used for modelling."""
    out = []
    for _, sub in rate_df.groupby("bed_share", dropna=False):
        sub = sub.sort_values(xvar).copy()
        sub["rate_raw"] = sub["rate"].astype(float)
        sub["rate"] = sub["rate_raw"].rolling(window=window, min_periods=min_periods,
                                              center=center).mean()
        out.append(sub)
    return (pd.concat(out, ignore_index=True)
            .sort_values([xvar, "bed_share"]).reset_index(drop=True))


# -----------------------------------------------------------------------------
# Descriptive rates / Figure 2
# -----------------------------------------------------------------------------

def get_total_births_denominator(
        path: str = "standardized/consolidated/denominator.parquet") -> int:
    path = Path(path)
    if not path.is_absolute():
        path = ORIG_CWD / path       # relative to the launch dir, not --outdir
    den = pd.read_parquet(path)
    if "combgest" in den.columns:
        den = den.copy()
        den.loc[den["combgest"] > 44, "combgest"] = np.nan
        den = den.dropna(subset=["combgest"])
    return int(den.shape[0])


def build_observed_rate_table(case_df: pd.DataFrame, week_var: str,
                              total_births: int | None = None,
                              rate_scale: float = 100000.0,
                              exposure_dir: Path = EXPOSURE_DIR) -> pd.DataFrame:
    """
    Descriptive weekly rate stratified by bed sharing.

    A3.20: numerator = NCFRP cases (participating jurisdictions only);
    denominator = national live births x bed-share prevalence. The two are not
    the same population, so the absolute per-100,000 scale is not a true
    incidence rate. Kept for the shape of the curves; the manuscript must state
    this. (Flagged, no code fix here without a jurisdiction-matched denominator.)
    """
    _, _, _, bedshare_exp = load_exposure(exposure_dir)
    if total_births is None:
        total_births = get_total_births_denominator()

    df = case_df.dropna(subset=[week_var, "bed_share"]).copy()
    df = df.loc[df["bed_share"].isin(["Y", "N"])]
    counts = (df.groupby([week_var, "bed_share"]).size().reset_index(name="cases")
              .sort_values([week_var, "bed_share"]).reset_index(drop=True))
    counts["bed_share_num"] = counts["bed_share"].map({"N": 0, "Y": 1})
    counts["denominator"] = counts["bed_share_num"].map(
        lambda x: total_births * float(bedshare_exp[x]))
    counts["rate"] = (counts["cases"] / counts["denominator"]) * rate_scale
    return counts[[week_var, "bed_share", "cases", "denominator", "rate"]]


def plot_incidence_curves_pna_pca(case_df: pd.DataFrame, title: str,
                                  total_births: int | None = None,
                                  rate_scale: float = 100000.0, *,
                                  smooth_window_pna: int | None = None,
                                  smooth_window_pca: int | None = None) -> None:
    """
    Figure 2: A. weekly incidence by postnatal age; B. by postconceptional age.
    Ed E7: panels A and B share a y-axis range; trend prose stays out of the
    figure (goes in the manuscript text).

    A3.21/G4 + lab feedback on reporting noise: the solid line is a centred
    moving average for VISUALISATION ONLY (windows below); the faint dots are the
    raw weekly values. All modelling uses the raw integer counts.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if total_births is None:
        total_births = get_total_births_denominator()
    w_pna = SMOOTH_WINDOW_PNA if smooth_window_pna is None else smooth_window_pna
    w_pca = SMOOTH_WINDOW_PCA if smooth_window_pca is None else smooth_window_pca

    pna_tbl = smooth_rate_table(
        build_observed_rate_table(case_df, "infageweeks", total_births, rate_scale),
        "infageweeks", window=w_pna, min_periods=1, center=True)
    pca_tbl = smooth_rate_table(
        build_observed_rate_table(case_df, "pca", total_births, rate_scale),
        "pca", window=w_pca, min_periods=1, center=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    COLORS = {"N": "tab:blue", "Y": "tab:orange"}

    def _panel(ax, tbl, xvar, xlabel, panel_title):
        for key, lbl in [("N", "Not bed sharing"), ("Y", "Bed sharing")]:
            sub = tbl.loc[tbl["bed_share"] == key].sort_values(xvar)
            ax.plot(sub[xvar], sub["rate_raw"], linestyle="none", marker="o",
                    markersize=3, alpha=0.22, color=COLORS[key])
            ax.plot(sub[xvar], sub["rate"], linewidth=2.5, color=COLORS[key],
                    label=lbl)
        ax.set_title(panel_title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)

    _panel(axes[0], pna_tbl, "infageweeks", "Postnatal age (weeks)",
           "Postnatal age scale")
    _panel(axes[1], pca_tbl, "pca", "Postconceptional age (weeks)",
           "Postconceptional age scale")

    # shared y-range (Ed E7). Use the smoothed max plus a robust raw quantile so
    # most raw dots are visible while the extreme month-reporting spikes clip.
    ymax = max(pna_tbl["rate"].max(), pca_tbl["rate"].max(),
               pna_tbl["rate_raw"].quantile(0.98), pca_tbl["rate_raw"].quantile(0.98))
    for ax in axes:
        ax.set_ylim(0, ymax * 1.05)
        ax.set_ylabel(f"Weekly SUID incidence per {int(rate_scale):,} live births",
                      fontsize=12)
    axes[0].legend(frameon=False, fontsize=10, loc="upper right")
    # The smoother/raw-dot explanation lives in the figure legend (caption), not
    # on the plot: the in-panel note overlapped the panel B traces (M1 R2 / Ed).
    for ax, label in zip(axes, ["A", "B"]):
        ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left",
                fontsize=14, fontweight="bold")

    fig.savefig(FIG_DIR / f"Figure2_{title}.tiff", dpi=300, format="tiff",
                bbox_inches="tight")
    fig.savefig(FIG_DIR / f"Figure2_{title}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"Figure2_{title}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 4: forest plot of IRRs from the early / late NB models
# -----------------------------------------------------------------------------

TERM_LABELS = {
    "pca": "PCA (per week)",
    "bed_share": "Bed sharing",
    "smoke": "Maternal smoking",
    "preterm": "Preterm birth",
    "bed_share:smoke": "Bed sharing \u00d7 maternal smoking",
    "bed_share:preterm": "Bed sharing \u00d7 preterm birth",
}
TERM_GROUPS = {
    "pca": "Developmental factor",
    "bed_share": "Sleep environment",
    "smoke": "Covariates",
    "preterm": "Covariates",
    "bed_share:smoke": "Interactions",
    "bed_share:preterm": "Interactions",
}
PREDICTOR_ORDER = ["pca", "bed_share", "smoke", "preterm",
                   "bed_share:smoke", "bed_share:preterm"]


def irr_table_from_csv(irr_csv: str, window_label: str) -> pd.DataFrame:
    """A4.30: the forest plot reads the SAME saved IRR table the model wrote."""
    raw = pd.read_csv(irr_csv)
    raw = raw.loc[raw["term"].isin(PREDICTOR_ORDER)].copy()
    raw["variable"] = raw["term"].map(TERM_LABELS)
    raw["group"] = raw["term"].map(TERM_GROUPS)
    raw = raw.rename(columns={"IRR": "rate_ratio", "IRR_ci_low": "rr_ci_low",
                              "IRR_ci_high": "rr_ci_high", "p_value": "pvalue"})
    raw["window"] = window_label
    # journal-style P values (M1 R2 / Ed: no "0.0***" renderings)
    raw["p_fmt"] = raw["pvalue"].map(format_p)
    # PCA is a per-week ratio close to 1: three decimals, otherwise two
    raw["est_ci_fmt"] = [(f"{e:.3f} ({lo:.3f}\u2013{hi:.3f})" if t == "pca" else
                          f"{e:.2f} ({lo:.2f}\u2013{hi:.2f})") for t, e, lo, hi in
                         zip(raw["term"], raw["rate_ratio"], raw["rr_ci_low"], raw["rr_ci_high"])]
    raw["term"] = pd.Categorical(raw["term"], categories=PREDICTOR_ORDER, ordered=True)
    return raw.sort_values("term").reset_index(drop=True)


def format_p(p: float) -> str:
    """AJE style: <.001, otherwise 2 decimals (3 below .01), no leading zero."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<.001"
    return (f"{p:.3f}" if p < 0.01 else f"{p:.2f}").lstrip("0")


def draw_nb_forest_panel(irr_csv: str, ax, title: str, xmin: float = 0.5, xmax: float = 25):
    """Forest panel drawn directly with matplotlib (the forestplot library put
    fixed-width gaps between its text columns and the plot). Reads the SAME
    saved IRR table the model wrote (A4.30). Columns: label | IRR (95% CI) |
    plot (log x) | P value. Group headers in bold."""
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    df = irr_table_from_csv(irr_csv, title)
    groups = [g for g in ["Developmental factor", "Sleep environment",
                          "Covariates", "Interactions"]
              if g in set(df["group"])]
    rows = []                       # (kind, text, row) top to bottom
    for g in groups:
        rows.append(("group", g, None))
        for _, r in df.loc[df["group"] == g].iterrows():
            rows.append(("term", r["variable"], r))
    n = len(rows)
    ys = list(range(n, 0, -1))
    trans = ax.get_yaxis_transform()            # x in axes fraction, y in data
    for (kind, text, r), y in zip(rows, ys):
        if kind == "group":
            ax.text(-1.30, y, text, transform=trans, ha="left", va="center",
                    fontsize=10, fontweight="bold")
            continue
        ax.text(-1.25, y, text, transform=trans, ha="left", va="center", fontsize=10)
        ax.text(-0.42, y, r["est_ci_fmt"], transform=trans, ha="left", va="center", fontsize=10)
        ax.text(1.04, y, r["p_fmt"], transform=trans, ha="left", va="center", fontsize=10)
        ax.plot([r["rr_ci_low"], r["rr_ci_high"]], [y, y], color="0.35", linewidth=1.4)
        ax.plot([r["rate_ratio"]], [y], marker="s", markersize=6, color="#2f4f4f")
    top = n + 0.9
    ax.text(-0.42, top, "IRR (95% CI)", transform=trans, ha="left", va="center",
            fontsize=10, fontweight="bold")
    ax.text(1.04, top, "P value", transform=trans, ha="left", va="center",
            fontsize=10, fontweight="bold")
    ax.axvline(1, color="gray", linestyle="--", alpha=0.8, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim(0.4, n + 1.3)
    ticks = [t for t in (0.5, 1, 2, 5, 10, 20) if xmin <= t <= xmax]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Adjusted incidence rate ratio (log scale)", fontsize=10)
    return ax


def plot_nb_forest(early_irr_csv: str, late_irr_csv: str, title: str,
                   filename_prefix: str = "Figure4") -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8.5))
    fig.subplots_adjust(left=0.50, right=0.86, top=0.93, bottom=0.07, hspace=0.42)
    draw_nb_forest_panel(early_irr_csv, axs[0], "Early developmental window", 0.5, 25)
    draw_nb_forest_panel(late_irr_csv, axs[1], "Late developmental window", 0.5, 25)
    panel_titles = [f"A. Early window (PCA <{PEAK1} weeks)",
                    f"B. Late window (PCA \u2265{PEAK2} weeks)"]
    for ax, label in zip(axs, panel_titles):
        ax.set_title(label, loc="left", x=-1.30, fontsize=12, fontweight="bold", pad=14)
    # A4.32 / Ed E11: tight bbox so the left-hand variable labels are not clipped
    fig.savefig(FIG_DIR / f"{filename_prefix}_{title}.tiff", dpi=300, format="tiff",
                bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{filename_prefix}_{title}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{filename_prefix}_{title}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Piecewise NB model  (A3.23 interactions; A3.25 inclusive peak3; A3.21 raw counts)
# -----------------------------------------------------------------------------

DESIGN_COLS = ["pca", "bed_share", "smoke", "preterm",
               "bed_share:smoke", "bed_share:preterm"]
# reduced design for smoking-stratified windows: `smoke` (and its interaction) is
# constant within a stratum, so it is dropped (lab action items 2 & 3).
DESIGN_COLS_NOSMOKE = ["pca", "bed_share", "preterm", "bed_share:preterm"]


def _add_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["bed_share:smoke"] = out["bed_share"] * out["smoke"]
    out["bed_share:preterm"] = out["bed_share"] * out["preterm"]
    return out


def prepare_piecewise_data(input1: pd.DataFrame, var1: str, peak1: int, peak2: int,
                           peak3: int, design_cols: list[str] | None = None):
    design_cols = DESIGN_COLS if design_cols is None else design_cols
    df = _add_interactions(input1)
    early = df.loc[df[var1] < peak1].copy()
    # A3.25: inclusive upper bound (>= peak2 and <= peak3), matching manuscript 1
    late = df.loc[(df[var1] >= peak2) & (df[var1] <= peak3)].copy()

    x1 = sm.add_constant(early[design_cols], has_constant="add")
    x2 = sm.add_constant(late[design_cols], has_constant="add")
    y1 = early["count"].to_numpy(dtype=float)   # A3.21/G4: RAW integer counts
    y2 = late["count"].to_numpy(dtype=float)
    return x1, x2, y1, y2, early["log_exposure"], late["log_exposure"], early, late


# -----------------------------------------------------------------------------
# Full-curve NB GAM
# -----------------------------------------------------------------------------

def make_knots_from_data(x: np.ndarray, step: int = 6) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    x_min, x_max = float(np.min(x)), float(np.max(x))
    knots = np.arange(np.ceil(x_min) + step, np.floor(x_max), step, dtype=float)
    return knots[(knots > x_min) & (knots < x_max)]


def prepare_curve_data_ncfrp(input1: pd.DataFrame, var1: str):
    """
    count ~ linear(bed_share + smoke + preterm) + smooth(PCA_main)
          + smooth(PCA_bedshare_deviation)
    A3.27: the deviation smooth is non-standard and carries the "bed sharing
    peaks earlier" claim; a standard interaction-spline variant is fit
    separately in run_piecewise_and_gam for robustness.
    A3.21/G4: y is the RAW integer count.
    """
    input1 = input1.dropna(subset=[var1, "log_exposure", "count"]).copy()
    x_linear = pd.DataFrame({
        "Intercept": 1.0,
        "bed_share": input1["bed_share"].astype(int).values,
        "smoke": input1["smoke"].astype(int).values,
        "preterm": input1["preterm"].astype(int).values,
    }, index=input1.index)
    y = input1["count"].to_numpy(dtype=float)
    offset = input1["log_exposure"].to_numpy(dtype=float)

    pca_vals = input1[var1].astype(float).to_numpy()
    bed_vals = input1["bed_share"].astype(int).to_numpy()
    pca_anchor = float(np.median(pca_vals))
    input1["pca_main"] = pca_vals
    input1["pca_beddev"] = np.where(bed_vals == 1, pca_vals, pca_anchor)
    x_smooth = input1[["pca_main", "pca_beddev"]].to_numpy(dtype=float)
    return x_linear, x_smooth, y, offset, input1


def fit_nb_gam(x_linear: pd.DataFrame, x_smooth: np.ndarray, y: np.ndarray,
               offset: np.ndarray, *, step: int = 6, nb_alpha: float = 0.10,
               alpha_grid_size: int = 0, cv_folds: int = 5):
    """
    A1.5: if alpha_grid_size > 0, cross-validate the spline penalty like
    manuscript 1's generate_figure3_gam_curves.py (select_penweight_kfold);
    otherwise use the fixed nb_alpha. The seed is set in main().
    """
    degree = 3
    knots_main = make_knots_from_data(x_smooth[:, 0], step=step)
    knots_beddev = make_knots_from_data(x_smooth[:, 1], step=step)
    if len(knots_main) < 2 or len(knots_beddev) < 2:
        raise ValueError(f"Too few interior knots (main={knots_main}, beddev={knots_beddev}).")

    df_main = len(knots_main) + degree + 1
    df_beddev = len(knots_beddev) + degree + 1
    smoother = BSplines(x_smooth, df=[df_main, df_beddev], degree=[degree, degree],
                        knot_kwds=[{"knots": knots_main}, {"knots": knots_beddev}],
                        include_intercept=False)
    family = NB(alpha=nb_alpha)
    model = GLMGam(y, exog=x_linear, smoother=smoother, family=family, offset=offset)

    if alpha_grid_size and alpha_grid_size > 0:
        alphas = np.logspace(-2, 3, alpha_grid_size)
        LOG.info("GAM: cross-validating spline penalty (%d^2 alphas x %d folds)",
                 alpha_grid_size, cv_folds)
        alpha_cv, _ = model.select_penweight_kfold(alphas=[alphas, alphas], k_folds=cv_folds)
        LOG.info("GAM: selected alpha_cv=%s", alpha_cv)
        model = GLMGam(y, exog=x_linear, smoother=smoother, family=family,
                       offset=offset, alpha=alpha_cv)

    res = model.fit()
    return res, {"knots_main": knots_main, "knots_beddev": knots_beddev}


# -----------------------------------------------------------------------------
# Prediction and plotting (Figure 3)
# -----------------------------------------------------------------------------

def make_prediction_design(pca_grid: np.ndarray, bed_share: int, smoke: int, preterm: int,
                           offset_value: float = 0.0, pca_anchor: float = 46.0):
    exog = pd.DataFrame({
        "Intercept": np.ones_like(pca_grid, dtype=float),
        "bed_share": np.full_like(pca_grid, bed_share, dtype=float),
        "smoke": np.full_like(pca_grid, smoke, dtype=float),
        "preterm": np.full_like(pca_grid, preterm, dtype=float),
    })
    pca_main = pca_grid.astype(float)
    pca_beddev = np.where(np.full_like(pca_grid, bed_share, dtype=int) == 1,
                          pca_grid.astype(float),
                          np.full_like(pca_grid, pca_anchor, dtype=float))
    exog_smooth = np.column_stack([pca_main, pca_beddev])
    offset = np.full_like(pca_grid, offset_value, dtype=float)
    return exog, exog_smooth, offset


def clip_exog_smooth_to_training_range(res, exog_smooth: np.ndarray) -> np.ndarray:
    x_new = np.asarray(exog_smooth, dtype=float).copy()
    x_train = np.asarray(res.model.smoother.x, dtype=float)
    for j in range(x_new.shape[1]):
        x_new[:, j] = np.clip(x_new[:, j], np.min(x_train[:, j]), np.max(x_train[:, j]))
    return x_new


def pred_with_ci(res, exog: pd.DataFrame, exog_smooth: np.ndarray, offset: np.ndarray):
    exog_smooth_safe = clip_exog_smooth_to_training_range(res, exog_smooth)
    try:
        pr = res.get_prediction(exog=exog, exog_smooth=exog_smooth_safe, offset=offset)
        sf = pr.summary_frame(alpha=0.05)
        return (sf["mean"].to_numpy(), sf["mean_ci_lower"].to_numpy(),
                sf["mean_ci_upper"].to_numpy())
    except Exception as e:
        LOG.warning("GAM CI not available via get_prediction: %r", e)
        mu = res.predict(exog=exog, exog_smooth=exog_smooth_safe, offset=offset)
        return np.asarray(mu), None, None


def plot_gam_curve(input1: pd.DataFrame, title: str, res, *, var1: str = "pca",
                   smoke_: int = 0, preterm_: int = 0, pca_min: int = PCA_MIN,
                   pca_max: int = PCA_MAX) -> None:
    pca_grid = np.arange(pca_min, pca_max + 1)
    off_ns = get_stratum_offset_from_input(input1, 0, smoke_, preterm_)
    off_bs = get_stratum_offset_from_input(input1, 1, smoke_, preterm_)
    exog_ns, xs_ns, o_ns = make_prediction_design(pca_grid, 0, smoke_, preterm_, off_ns)
    exog_bs, xs_bs, o_bs = make_prediction_design(pca_grid, 1, smoke_, preterm_, off_bs)
    mu_ns, lo_ns, hi_ns = pred_with_ci(res, exog_ns, xs_ns, o_ns)
    mu_bs, lo_bs, hi_bs = pred_with_ci(res, exog_bs, xs_bs, o_bs)

    obs_ns = input1.loc[(input1["bed_share"] == 0) & (input1["smoke"] == smoke_)
                        & (input1["preterm"] == preterm_)].sort_values(var1)
    obs_bs = input1.loc[(input1["bed_share"] == 1) & (input1["smoke"] == smoke_)
                        & (input1["preterm"] == preterm_)].sort_values(var1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), constrained_layout=True)

    ax = axes[0]
    ax.plot(obs_ns[var1], obs_ns["count"], marker="v", linewidth=.5,
            label="Observed non-bed sharing", color="blue")
    ax.plot(obs_bs[var1], obs_bs["count"], marker="^", linewidth=.5,
            label="Observed bed sharing", color="orange")
    ax.plot(pca_grid, mu_ns, linewidth=3, linestyle="-.",
            label="Predicted mean non-bed sharing", color="blue")
    ax.plot(pca_grid, mu_bs, linewidth=3, linestyle="--",
            label="Predicted mean bed sharing", color="orange")
    if lo_ns is not None:
        ax.fill_between(pca_grid, lo_ns, hi_ns, alpha=0.15, color="blue")
    if lo_bs is not None:
        ax.fill_between(pca_grid, lo_bs, hi_bs, alpha=0.15, color="orange")
    ax.set_xlabel("PCA (weeks)")
    ax.set_ylabel("SUID case count")
    ax.set_title("A. GAM-estimated SUID incidence trajectories by sleep environment")
    ax.set_xlim(pca_min - 1, pca_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    # Panel B (A4.32c / Ed E10): fitted rate ratio (bed-share / non) vs PCA with
    # a log-scale delta CI, instead of the previous raw count difference whose
    # CI band was lo_bs - hi_ns. Falls back to a plain ratio line if no CI.
    ax2 = axes[1]
    ratio = np.divide(mu_bs, mu_ns, out=np.full_like(mu_bs, np.nan, dtype=float),
                      where=mu_ns > 0)
    ax2.plot(pca_grid, ratio, linewidth=3, linestyle="--", color="k",
             label="Fitted bed-sharing / non-sharing ratio")
    if lo_ns is not None and lo_bs is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_lo = np.log(lo_bs) - np.log(hi_ns)
            log_hi = np.log(hi_bs) - np.log(lo_ns)
        ax2.fill_between(pca_grid, np.exp(log_lo), np.exp(log_hi), alpha=0.15, color="k")
    ax2.axhline(1.0, linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.set_xlabel("PCA (weeks)")
    ax2.set_ylabel("Fitted incidence rate ratio (bed sharing vs not)")
    ax2.set_title("B. Bed-sharing vs non-sharing fitted rate ratio across PCA")
    ax2.set_xlim(pca_min - 1, pca_max)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False)

    for ax_i, label in zip(axes, ["A", "B"]):
        ax_i.text(0.02, 0.98, label, transform=ax_i.transAxes, va="top", ha="left",
                  fontsize=14, fontweight="bold")

    fig.savefig(GAM_FIG_DIR / f"Figure3_{title}.png", dpi=300, bbox_inches="tight")
    fig.savefig(GAM_FIG_DIR / f"Figure3_{title}.tiff", dpi=300, format="tiff",
                bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def run_piecewise_and_gam(df_run: pd.DataFrame, label: str, *,
                          pca_min: int = PCA_MIN, pca_max: int = PCA_MAX,
                          peak1: int = PEAK1, peak2: int = PEAK2,
                          peak3: int | None = None, min_count: int = MIN_COUNT,
                          exposure_dir: Path = EXPOSURE_DIR,
                          gam_alpha_grid: int = 0, offset_mode: str = "joint") -> dict:
    peak3 = pca_max if peak3 is None else peak3   # A3.25
    LOG.info("=" * 78)
    LOG.info("RUN: %s  (PCA %d-%d, early<%d, late %d-%d, offset=%s)", label, pca_min,
             pca_max, peak1, peak2, peak3, offset_mode)

    var1 = "pca"
    input_df = make_ncfrp_input(var1, df_run, exposure_dir=exposure_dir,
                                min_count=min_count, offset_mode=offset_mode)
    check_overdispersion(input_df)
    check_overdispersion(input_df, "bed_share")

    # A3.21/G4: NO pre-fit smoothing. Models fit on raw integer counts.
    x1, x2, y1, y2, off1, off2, early, late = prepare_piecewise_data(
        input_df, var1, peak1, peak2, peak3)

    model_early = NegativeBinomial(y1, x1, offset=off1).fit(maxiter=200, disp=False)
    model_late = NegativeBinomial(y2, x2, offset=off2).fit(maxiter=200, disp=False)
    LOG.info("%s early window: %d strata rows, %d cases", label, len(x1), int(y1.sum()))
    LOG.info("%s late  window: %d strata rows, %d cases", label, len(x2), int(y2.sum()))

    # full-curve GAM (deviation-smooth spec)
    x_linear, x_smooth, y, offset, curve_df = prepare_curve_data_ncfrp(input_df, var1)
    gam_res, knots = fit_nb_gam(x_linear, x_smooth, y, offset, alpha_grid_size=gam_alpha_grid)
    LOG.info("%s GAM knots: %s", label, knots)

    # A3.27: the exploratory bed_share x PCA interaction-spline robustness
    # variant that used to be fit here (gam_res_int) was retired 2026-09-04 -
    # it was numerically degenerate (SE~329, CI spanning ~560 orders of
    # magnitude; see AUDIT_FINDINGS.md A3.27 close-out for the root cause: its
    # knots were placed from a zero-inflated bed_share*pca product instead of
    # the bed_share==1 subset's real PCA range, and it multiplied by
    # bed_share BEFORE basis expansion instead of after). The corrected
    # replacement is a standalone varying-coefficient model, fit as a plain
    # NB MLE rather than through GLMGam: see gam_bedshare_pca_varycoef.py.

    # save model outputs (single source of numbers - A4.30)
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    slug = label.lower()
    early_irr = save_nb_model_outputs(model_early, f"{OUT_DATA_DIR}/{slug}_early")
    late_irr = save_nb_model_outputs(model_late, f"{OUT_DATA_DIR}/{slug}_late")
    save_gam_model_outputs(gam_res, f"{OUT_DATA_DIR}/{slug}_gam")

    with open(f"{OUT_DATA_DIR}/{slug}_early_summary.txt", "w") as f:
        f.write(str(model_early.summary()))
    with open(f"{OUT_DATA_DIR}/{slug}_late_summary.txt", "w") as f:
        f.write(str(model_late.summary()))

    # figures (Figure 4 reads the saved IRR CSVs - A4.30)
    plot_incidence_curves_pna_pca(df_run, label, total_births=None, rate_scale=100000.0)
    plot_nb_forest(f"{OUT_DATA_DIR}/{slug}_early_irr.csv",
                   f"{OUT_DATA_DIR}/{slug}_late_irr.csv", label, filename_prefix="Figure4")
    plot_gam_curve(curve_df, label, gam_res, var1="pca", smoke_=0, preterm_=0,
                   pca_min=pca_min, pca_max=pca_max)

    return {"input_df": input_df, "model_early": model_early, "model_late": model_late,
            "gam_res": gam_res, "knots": knots,
            "early_data": early, "late_data": late, "curve_data": curve_df,
            "early_irr": early_irr, "late_irr": late_irr}


# -----------------------------------------------------------------------------
# Lab-meeting add-ons: smoking-stratified windows, <1-month table, bed_share x LBW
# interaction. Supplementary to the primary models; each fails soft.
# -----------------------------------------------------------------------------

def run_windowed_stratified(df_run: pd.DataFrame, by: str = "smoke", *,
                            pca_min: int = PCA_MIN, pca_max: int = PCA_MAX,
                            peak1: int = PEAK1, peak2: int = PEAK2,
                            peak3: int | None = None, min_count: int = MIN_COUNT,
                            exposure_dir: Path = EXPOSURE_DIR,
                            offset_mode: str = "joint") -> dict:
    """
    Lab action items 2 & 3: re-fit the early AND late windowed NB models
    separately for each level of `by` (default maternal smoking). `by` is dropped
    from the design when it is constant within a stratum (smoke). The primary
    models keep the bed_share x smoke interaction; this is for interpreting it.
    Outputs -> output_data/ncfrp/stratified/{by}{level}_{early,late}_irr.csv and
    output_figures/ncfrp/Figure5_{by}_stratified.{png,tiff}.
    """
    peak3 = pca_max if peak3 is None else peak3
    var1 = "pca"
    design = DESIGN_COLS_NOSMOKE if by == "smoke" else DESIGN_COLS
    strat_dir = OUT_DATA_DIR / "stratified"
    strat_dir.mkdir(parents=True, exist_ok=True)
    levels = sorted(pd.Series(df_run[by]).dropna().unique())
    saved: dict = {}
    for lvl in levels:
        lvl_i = int(lvl)
        sub = df_run.loc[df_run[by] == lvl].copy()
        try:
            inp = make_ncfrp_input(var1, sub, exposure_dir=exposure_dir,
                                   min_count=min_count, offset_mode=offset_mode)
            x1, x2, y1, y2, off1, off2, _e, _l = prepare_piecewise_data(
                inp, var1, peak1, peak2, peak3, design_cols=design)
            m_e = NegativeBinomial(y1, x1, offset=off1).fit(maxiter=200, disp=False)
            m_l = NegativeBinomial(y2, x2, offset=off2).fit(maxiter=200, disp=False)
            for win, m in [("early", m_e), ("late", m_l)]:
                pref = f"{strat_dir}/{by}{lvl_i}_{win}"
                saved[(lvl_i, win)] = save_nb_model_outputs(m, pref)
                with open(f"{pref}_summary.txt", "w") as f:
                    f.write(str(m.summary()))
            LOG.info("stratified %s=%d: early %d strata rows / late %d strata rows",
                     by, lvl_i, len(x1), len(x2))
        except Exception as e:
            LOG.warning("stratified %s=%s failed: %r", by, lvl, e)

    try:
        levs = sorted({k[0] for k in saved})
        if len(levs) == 2:
            fig, axs = plt.subplots(2, 2, figsize=(15, 12))
            fig.subplots_adjust(left=0.30, right=0.96, top=0.95, bottom=0.07,
                                hspace=0.5, wspace=0.6)
            for r, lvl_i in enumerate(levs):
                for c, win in enumerate(["early", "late"]):
                    draw_nb_forest_panel(f"{strat_dir}/{by}{lvl_i}_{win}_irr.csv",
                                         axs[r, c], f"{by}={lvl_i}, {win} window",
                                         0.3, 40)
            fig.savefig(FIG_DIR / f"Figure5_{by}_stratified.png", dpi=300,
                        bbox_inches="tight")
            fig.savefig(FIG_DIR / f"Figure5_{by}_stratified.tiff", dpi=300,
                        format="tiff", bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        LOG.warning("stratified forest (%s) failed: %r", by, e)
    return saved


def under1mo_by_smoke_bedshare(df: pd.DataFrame, out_dir: Path = OUT_DATA_DIR) -> pd.DataFrame:
    """
    Lab action item 2: descriptive count of deaths before 1 month of age, by
    maternal smoking x bed sharing. Not a model. -> output_data/ncfrp/stratified/.
    """
    wpm = 365.25 / 12 / 7
    d = df.copy()
    u1 = pd.to_numeric(d["infageweeks"], errors="coerce") < wpm
    smoker = np.where(d["smoke"] == 1, "Smoker", "Non-smoker")
    bs = np.where(d["bed_share"].astype(str).str.upper().str[0] == "Y",
                  "Bed sharing", "Not bed sharing")
    rows = []
    for sm_lbl in ["Smoker", "Non-smoker"]:
        m_sm = smoker == sm_lbl
        rec = {"Maternal smoking": sm_lbl, "N in group": f"{int(m_sm.sum()):,}"}
        for bs_lbl in ["Bed sharing", "Not bed sharing"]:
            m = m_sm & (bs == bs_lbl)
            n_grp = int(m.sum())
            n_u1 = int((u1 & m).sum())
            rec[f"{bs_lbl}: deaths <1 mo"] = (
                f"{n_u1:,} / {n_grp:,} ({100 * n_u1 / n_grp:.1f}%)"
                if n_grp else "0 / 0 (NA)")
        rows.append(rec)
    tab = pd.DataFrame(rows)
    strat_dir = Path(out_dir) / "stratified"
    strat_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(strat_dir / "under1mo_by_smoke_bedshare.csv", index=False)
    try:
        with open(strat_dir / "under1mo_by_smoke_bedshare.md", "w") as f:
            f.write(tab.to_markdown(index=False))
    except Exception as e:
        LOG.warning("under1mo markdown write failed: %r", e)
    LOG.info("Deaths <1 month by smoking x bed sharing:\n%s", tab.to_string(index=False))
    return tab


def run_windowed_lbw_interaction(df_run: pd.DataFrame, *, pca_min: int = PCA_MIN,
                                 pca_max: int = PCA_MAX, peak1: int = PEAK1,
                                 peak2: int = PEAK2, peak3: int | None = None,
                                 min_count: int = MIN_COUNT,
                                 exposure_dir: Path = EXPOSURE_DIR,
                                 offset_mode: str = "joint") -> None:
    """
    Lab action item 1 (interaction form): early/late windowed NB with a <2500 g
    birthweight indicator and a bed_share x lbw term added to the prespecified
    design. Birthweight has no population offset, so lbw enters offset-neutral
    (log_exposure joined on the smoke/preterm/bed_share keys). Self-contained;
    does not touch the primary models. The early cell may be underpowered once
    split by lbw - a failure is logged, not raised.
    -> output_data/ncfrp/lbw_interaction_{early,late}_irr.csv
    """
    if "lbw_bin" not in df_run.columns or pd.Series(df_run["lbw_bin"]).isna().all():
        LOG.warning("lbw_bin unavailable - skipping bed_share x LBW interaction")
        return
    peak3 = pca_max if peak3 is None else peak3
    var1 = "pca"
    base = make_ncfrp_input(var1, df_run, exposure_dir=exposure_dir,
                            min_count=1, offset_mode=offset_mode)
    key = ["bed_share", "smoke", "preterm"]
    off = base[key + ["log_exposure"]].drop_duplicates(subset=key)

    d = df_run.dropna(subset=["lbw_bin"]).copy()
    d["_bed"] = (d["bed_share"].astype(str).str.upper().str[0] == "Y").astype(int)
    d["_smk"] = d["smoke"].astype(int)
    d["_pt"] = (d["preterm"].astype(str).str.upper().str[0] == "Y").astype(int)
    d["_lbw"] = d["lbw_bin"].astype(int)
    cnt = (d.groupby([var1, "_bed", "_smk", "_pt", "_lbw"]).size()
           .reset_index(name="count")
           .rename(columns={"_bed": "bed_share", "_smk": "smoke",
                            "_pt": "preterm", "_lbw": "lbw"}))
    cnt = (cnt.loc[cnt["count"] >= min_count]
           .merge(off, on=key, how="left").dropna(subset=["log_exposure"]))
    cnt["bed_share:smoke"] = cnt["bed_share"] * cnt["smoke"]
    cnt["bed_share:preterm"] = cnt["bed_share"] * cnt["preterm"]
    cnt["bed_share:lbw"] = cnt["bed_share"] * cnt["lbw"]
    cols = DESIGN_COLS + ["lbw", "bed_share:lbw"]

    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    windows = {"early": cnt.loc[cnt[var1] < peak1],
               "late": cnt.loc[(cnt[var1] >= peak2) & (cnt[var1] <= peak3)]}
    for win, part in windows.items():
        try:
            x = sm.add_constant(part[cols], has_constant="add")
            y = part["count"].to_numpy(dtype=float)
            m = NegativeBinomial(
                y, x, offset=part["log_exposure"].to_numpy(dtype=float)
            ).fit(maxiter=200, disp=False)
            save_nb_model_outputs(m, f"{OUT_DATA_DIR}/lbw_interaction_{win}")
            with open(f"{OUT_DATA_DIR}/lbw_interaction_{win}_summary.txt", "w") as f:
                f.write(str(m.summary()))
            LOG.info("lbw interaction %s window: %d strata rows", win, len(x))
        except Exception as e:
            LOG.warning("lbw interaction %s window failed: %r", win, e)


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NCFRP bed-sharing developmental SUID pipeline (audited).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-path", type=Path, default=DATA_PATH)
    p.add_argument("--exposure-dir", type=Path, default=EXPOSURE_DIR)
    p.add_argument("--outdir", type=Path, default=Path("."),
                   help="working directory for output_figures/ and output_data/")
    p.add_argument("--pca-min", type=int, default=PCA_MIN)
    p.add_argument("--pca-max", type=int, default=PCA_MAX,
                   help="A3.17: set from pca_boundary_report_ncfrp.py, do not assume 75")
    p.add_argument("--peak1", type=int, default=PEAK1)
    p.add_argument("--peak2", type=int, default=PEAK2)
    p.add_argument("--peak3", type=int, default=None,
                   help="late-window upper bound; defaults to --pca-max")
    p.add_argument("--min-count", type=int, default=MIN_COUNT,
                   help="A3.22: per-stratum count threshold; M1 uses none")
    p.add_argument("--first-week-cut", type=int, default=FIRST_WEEK_CUT)
    p.add_argument("--gestage-max", type=int, default=GESTAGE_MAX,
                   help="inf3gestage > this is an implausible-value exclusion (T1)")
    p.add_argument("--max-infage-days", type=int, default=MAX_INFAGE_DAYS)
    p.add_argument("--year-min", type=int, default=YEAR_MIN,
                   help="restrict to death years >= this (infdodyear) if the column exists")
    p.add_argument("--year-max", type=int, default=YEAR_MAX)
    p.add_argument("--offset-mode", choices=["joint", "marginal"], default="joint",
                   help="G2: 'joint' uses P(smoke,preterm) from CDC births x P(bed-share) "
                        "from PRAMS; 'marginal' is the product of 3 marginals (sensitivity)")
    p.add_argument("--gam-alpha-grid", type=int, default=0,
                   help="A1.5: >0 cross-validates the GAM spline penalty over this many alphas")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-sex-strata", action="store_true",
                   help="skip the FEMALES / MALES runs (A4.31: they run by default)")
    p.add_argument("--no-extra-strata", action="store_true",
                   help="skip the lab-meeting add-ons: LBW/NONLBW runs, the "
                        "months-age sensitivity run, smoking-stratified windows, "
                        "the <1-month table, and the bed_share x LBW interaction")
    p.add_argument("--smooth-window-pna", type=int, default=SMOOTH_WINDOW_PNA,
                   help="Figure 2A display-only centred moving-average window (weeks)")
    p.add_argument("--smooth-window-pca", type=int, default=SMOOTH_WINDOW_PCA,
                   help="Figure 2B display-only centred moving-average window (weeks)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    # Resolve input paths to absolute BEFORE the chdir below. Otherwise relative
    # paths (CLI args and the denominator default) are silently reinterpreted
    # against --outdir once we cd into it.
    global ORIG_CWD
    ORIG_CWD = Path.cwd()
    args.data_path = args.data_path.resolve()
    args.exposure_dir = args.exposure_dir.resolve()

    if args.outdir != Path("."):
        args.outdir.mkdir(parents=True, exist_ok=True)
        os.chdir(args.outdir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    np.random.seed(args.seed)   # A1.5

    global PCA_MIN, PCA_MAX, PEAK1, PEAK2, MIN_COUNT
    PCA_MIN, PCA_MAX, PEAK1, PEAK2, MIN_COUNT = (
        args.pca_min, args.pca_max, args.peak1, args.peak2, args.min_count)
    global SMOOTH_WINDOW_PNA, SMOOTH_WINDOW_PCA
    SMOOTH_WINDOW_PNA, SMOOTH_WINDOW_PCA = args.smooth_window_pna, args.smooth_window_pca

    ensure_dirs()

    df, _audit = load_ncfrp_data(
        args.data_path, pca_min=args.pca_min, pca_max=args.pca_max,
        first_week_cut=args.first_week_cut, gestage_max=args.gestage_max,
        max_infage_days=args.max_infage_days, year_min=args.year_min,
        year_max=args.year_max, audit_csv=AUDIT_CSV)
    LOG.info("Cleaned NCFRP analytic dataset: %s", df.shape)

    windows = split_analytic_windows(df, peak1=args.peak1, peak2=args.peak2)
    pd.DataFrame([{
        "full_analytic_N": len(windows["full"]),
        "early_n": len(windows["early"]),
        "late_n": len(windows["late"]),
        "peak_excluded_n": len(windows["peak"]),
    }]).to_csv(OUT_DATA_DIR / "two_population_Ns.csv", index=False)

    smoke_bedshare_case_association(df)

    runs = {"ALL": df}
    if not args.no_sex_strata:
        runs["FEMALES"] = df.loc[df["sex"] == "F"].copy()
        runs["MALES"] = df.loc[df["sex"] == "M"].copy()

    for label, sub in runs.items():
        run_piecewise_and_gam(
            sub, label, pca_min=args.pca_min, pca_max=args.pca_max,
            peak1=args.peak1, peak2=args.peak2, peak3=args.peak3,
            min_count=args.min_count, exposure_dir=args.exposure_dir,
            gam_alpha_grid=args.gam_alpha_grid, offset_mode=args.offset_mode)

    # ---- lab-meeting add-ons (supplementary; each fails soft) ----------------
    if not args.no_extra_strata:
        extra: dict = {}
        if "lbw_bin" in df.columns and not df["lbw_bin"].isna().all():
            extra["LBW"] = df.loc[df["lbw_bin"] == 1].copy()
            extra["NONLBW"] = df.loc[df["lbw_bin"] == 0].copy()
        # (a months-age sensitivity run is not viable here: ~91% of NCFRP infant
        #  ages are reported in whole months, so dropping infagecat == 2 leaves
        #  too few cases to fit the windowed models. The reporting-artifact
        #  attribution in compare_time_axes.py carries this instead.)
        for label, sub in extra.items():
            try:
                # gam_alpha_grid=0 for the supplementary strata: they do not need
                # CV spline-penalty selection and it is the dominant cost.
                run_piecewise_and_gam(
                    sub, label, pca_min=args.pca_min, pca_max=args.pca_max,
                    peak1=args.peak1, peak2=args.peak2, peak3=args.peak3,
                    min_count=args.min_count, exposure_dir=args.exposure_dir,
                    gam_alpha_grid=0, offset_mode=args.offset_mode)
            except Exception as e:
                LOG.warning("extra stratum run %s failed: %r", label, e)

        if args.offset_mode == "joint":
            try:
                run_windowed_stratified(
                    df, "smoke", pca_min=args.pca_min, pca_max=args.pca_max,
                    peak1=args.peak1, peak2=args.peak2, peak3=args.peak3,
                    min_count=args.min_count, exposure_dir=args.exposure_dir,
                    offset_mode=args.offset_mode)
            except Exception as e:
                LOG.warning("run_windowed_stratified(smoke) failed: %r", e)
            try:
                under1mo_by_smoke_bedshare(df)
            except Exception as e:
                LOG.warning("under1mo_by_smoke_bedshare failed: %r", e)
            try:
                run_windowed_lbw_interaction(
                    df, pca_min=args.pca_min, pca_max=args.pca_max,
                    peak1=args.peak1, peak2=args.peak2, peak3=args.peak3,
                    min_count=args.min_count, exposure_dir=args.exposure_dir,
                    offset_mode=args.offset_mode)
            except Exception as e:
                LOG.warning("run_windowed_lbw_interaction failed: %r", e)

    LOG.info("DONE. Outputs under %s", Path("output_data/ncfrp").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

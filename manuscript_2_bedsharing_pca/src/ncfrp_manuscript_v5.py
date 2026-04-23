# ncfrp_manuscript_v4.py
"""
NCFRP SIDS/SUID developmental manuscript pipeline

Core model:
    stratified counts over PCA x smoke x preterm x bed_share
    no zero-count rows
    minimum count threshold enforced
    separate runs:
        1) all infants
        2) females only
        3) males only

Main outputs:
    - piecewise negative binomial models
    - full-curve NB spline model
    - descriptive chi-square smoke x bed-share association among cases
"""

from __future__ import annotations

import os
import pickle
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
import matplotlib.ticker as mticker
from scipy.stats import chi2_contingency


# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)

DATA_PATH = Path("standardized/NCFRP_SCH_LR_BW_pop.csv")
EXPOSURE_DIR = Path("output_data/risk_exposure")
FIG_DIR = Path("output_figures/ncfrp")
GAM_FIG_DIR = FIG_DIR / "gam"

MIN_COUNT = 10
PCA_MIN = 36
PCA_MAX = 75


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    GAM_FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_exposure():
    """
    Expected coding of exposure dictionaries:
        smoke_exposure   : {0: nonsmoker, 1: smoker}
        sex_exposure     : {0: female, 1: male}
        term_exposure    : {0: term, 1: preterm}
        bedshare_exposure: {0: non-bed-share, 1: bed-share}
    """
    smoking_exposure = load_pickle(EXPOSURE_DIR / "smoke_exposure.pkl")
    sex_exposure = load_pickle(EXPOSURE_DIR / "sex_exposure.pkl")
    preterm_exposure = load_pickle(EXPOSURE_DIR / "term_exposure.pkl")
    bedshare_exposure = load_pickle(EXPOSURE_DIR / "bedshare_exposure.pkl")
    return smoking_exposure, sex_exposure, preterm_exposure, bedshare_exposure


# -----------------------------------------------------------------------------
# Cleaning and recoding
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd


def load_ncfrp_data_with_audit(
    data_path,
    pca_min=36,
    pca_max=75,
    save_report_path=None,
):
    """
    Load and clean NCFRP data while producing an auditable row-survival report.

    The report is sequential:
    each step is applied to the rows that survived the previous step.

    Returns
    -------
    df_final : pd.DataFrame
        Final cleaned analytic dataset.

    audit_report : pd.DataFrame
        One row per filtering step with counts before/failed/dropped/remaining.
    """

    audit_rows = []

    def log_step(step_name, before_n, fail_mask, action="drop"):
        """
        Record one sequential filtering step.

        Parameters
        ----------
        step_name : str
            Human-readable label for the step.
        before_n : int
            Number of rows entering the step.
        fail_mask : pd.Series[bool]
            Boolean mask on current dataframe: True means row fails this step.
        action : str
            Usually 'drop'. Kept for readability in the report.
        """
        failed_n = int(fail_mask.sum())
        after_n = int(before_n - failed_n)

        audit_rows.append({
            "step": step_name,
            "action": action,
            "rows_before": before_n,
            "rows_failing_step": failed_n,
            "rows_dropped_this_step": failed_n,
            "rows_after": after_n,
        })

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    df = pd.read_csv(data_path, low_memory=False)
    df.columns = df.columns.str.lower()

    audit_rows.append({
        "step": "raw_load",
        "action": "load",
        "rows_before": len(df),
        "rows_failing_step": 0,
        "rows_dropped_this_step": 0,
        "rows_after": len(df),
    })

    # ------------------------------------------------------------------
    # Required columns check
    # ------------------------------------------------------------------
    required_cols = ["infagedays", "inf3gestage", "smoker", "infsex", "bed_sharing"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # ------------------------------------------------------------------
    # Derived variables before sequential dropping
    # ------------------------------------------------------------------
    df["infageweeks"] = np.floor(df["infagedays"] / 7)

    # smoker recode
    df["cig_rec"] = np.nan
    df.loc[df["smoker"] == 1, "cig_rec"] = "Y"
    df.loc[df["smoker"] == 2, "cig_rec"] = "N"

    df["smoke"] = np.nan
    df.loc[df["cig_rec"] == "Y", "smoke"] = 1
    df.loc[df["cig_rec"] == "N", "smoke"] = 0

    # sex recode
    df["sex"] = np.nan
    df.loc[df["infsex"] == 1, "sex"] = "M"
    df.loc[df["infsex"] == 2, "sex"] = "F"

    # bed sharing recode
    df["bed_share"] = np.nan
    df.loc[df["bed_sharing"] == 1, "bed_share"] = "Y"
    df.loc[df["bed_sharing"] == 2, "bed_share"] = "N"

    df["bed_share_bin"] = np.nan
    df.loc[df["bed_share"] == "Y", "bed_share_bin"] = 1
    df.loc[df["bed_share"] == "N", "bed_share_bin"] = 0

    # ------------------------------------------------------------------
    # Sequential filtering
    # ------------------------------------------------------------------

    # 1. Missing postnatal age input
    before_n = len(df)
    fail = df["infagedays"].isna()
    log_step("missing_infagedays", before_n, fail)
    df = df.loc[~fail].copy()

    # refresh derived weeks after filtering
    df["infageweeks"] = np.floor(df["infagedays"] / 7)

    # 2. First postnatal week exclusion (<= 1 completed week, per your current code)
    before_n = len(df)
    fail = df["infageweeks"] <= 1
    log_step("exclude_first_postnatal_week", before_n, fail)
    df = df.loc[~fail].copy()

    # 3. Missing gestational age
    before_n = len(df)
    fail = df["inf3gestage"].isna()
    log_step("missing_gestational_age", before_n, fail)
    df = df.loc[~fail].copy()

    # 4. Unrealistic gestational age <20 weeks
    before_n = len(df)
    fail = df["inf3gestage"] < 20
    log_step("exclude_gestational_age_lt20", before_n, fail)
    df = df.loc[~fail].copy()

    # 5. Cap gestational age >42 to 40, but DO NOT drop
    # This is a data correction step, not a row-loss step.
    n_capped = int((df["inf3gestage"] > 42).sum())
    before_n = len(df)
    df.loc[df["inf3gestage"] > 42, "inf3gestage"] = 40
    audit_rows.append({
        "step": "cap_gestational_age_gt42_to_40",
        "action": "recode_only",
        "rows_before": before_n,
        "rows_failing_step": n_capped,
        "rows_dropped_this_step": 0,
        "rows_after": before_n,
    })

    # 6. Build PCA
    df["pca"] = df["infageweeks"] + df["inf3gestage"]

    # 7. PCA below lower bound
    before_n = len(df)
    fail = df["pca"] < pca_min
    log_step(f"exclude_pca_lt_{pca_min}", before_n, fail)
    df = df.loc[~fail].copy()

    # 8. PCA above upper bound
    before_n = len(df)
    fail = df["pca"] > pca_max
    log_step(f"exclude_pca_gt_{pca_max}", before_n, fail)
    df = df.loc[~fail].copy()

    # 9. Bad smoking data
    before_n = len(df)
    fail = df["cig_rec"].isna()
    log_step("missing_or_invalid_smoking", before_n, fail)
    df = df.loc[~fail].copy()

    # refresh smoke after filtering
    df["smoke"] = np.nan
    df.loc[df["cig_rec"] == "Y", "smoke"] = 1
    df.loc[df["cig_rec"] == "N", "smoke"] = 0

    # 10. Bad sex data
    before_n = len(df)
    fail = df["sex"].isna()
    log_step("missing_or_invalid_sex", before_n, fail)
    df = df.loc[~fail].copy()

    # 11. Bad bed-share data
    before_n = len(df)
    fail = df["bed_share"].isna()
    log_step("missing_or_invalid_bedsharing", before_n, fail)
    df = df.loc[~fail].copy()

    # refresh bin
    df["bed_share_bin"] = np.nan
    df.loc[df["bed_share"] == "Y", "bed_share_bin"] = 1
    df.loc[df["bed_share"] == "N", "bed_share_bin"] = 0

    # 12. Preterm derivation should now be valid for all surviving rows
    df["preterm"] = np.nan
    df.loc[df["inf3gestage"] < 37, "preterm"] = "Y"
    df.loc[df["inf3gestage"] >= 37, "preterm"] = "N"

    df["preterm_bin"] = np.nan
    df.loc[df["preterm"] == "Y", "preterm_bin"] = 1
    df.loc[df["preterm"] == "N", "preterm_bin"] = 0

    # 13. Final safeguard: any missing analytic field still left?
    keep_cols = [
        "pca", "infageweeks", "inf3gestage",
        "cig_rec", "smoke",
        "sex",
        "bed_share", "bed_share_bin",
        "preterm", "preterm_bin",
    ]
    before_n = len(df)
    fail = df[keep_cols].isna().any(axis=1)
    log_step("final_missing_any_required_analytic_field", before_n, fail)
    df = df.loc[~fail].copy()

    # ------------------------------------------------------------------
    # Final typing
    # ------------------------------------------------------------------
    df["pca"] = df["pca"].astype(int)
    df["smoke"] = df["smoke"].astype(int)
    df["bed_share_bin"] = df["bed_share_bin"].astype(int)
    df["preterm_bin"] = df["preterm_bin"].astype(int)

    audit_report = pd.DataFrame(audit_rows)

    if save_report_path is not None:
        audit_report.to_csv(save_report_path, index=False)

    return df, audit_report

def load_ncfrp_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f'\nLoaded df original shape: {df.shape}')
    df.columns = df.columns.str.lower()

    # Postnatal age in completed weeks
    if "infagedays" not in df.columns:
        raise ValueError("Expected column 'infagedays' not found in NCFRP file.")
    df["infageweeks"] = np.floor(df["infagedays"] / 7)

    # Exclude first postnatal week
    print(f'\nDrop the first week cases: ', df.loc[df['infageweeks']<=1].shape)
    df.loc[df["infageweeks"] <= 1, "infageweeks"] = np.nan

    # Gestational age cleanup
    if "inf3gestage" not in df.columns:
        raise ValueError("Expected column 'inf3gestage' not found in NCFRP file.")
    print(f'\nDrop unrealistic gestational ages (less than 20 weeks): ',df.loc[df['inf3gestage']<20].shape)
    df.loc[df["inf3gestage"] > 42, "inf3gestage"] = 40
    df.loc[df["inf3gestage"] < 20, "inf3gestage"] = np.nan

    # PCA
    df["pca"] = df["infageweeks"] + df["inf3gestage"]
    print(f'\nDropping PCA bad data. less than {PCA_MIN}: ',df.loc[df['pca']<PCA_MIN].shape, f', more than {PCA_MAX}: ',df.loc[df['pca']>PCA_MAX].shape)
    df.loc[df["pca"] < PCA_MIN, "pca"] = np.nan
    df.loc[df["pca"] > PCA_MAX, "pca"] = np.nan

    # Smoking
    print(df['smoker'].value_counts())
    df["cig_rec"] = np.nan
    df.loc[df["smoker"] == 1, "cig_rec"] = "Y"
    df.loc[df["smoker"] == 2, "cig_rec"] = "N"
    print(f'\nBad smoker data: ',df.loc[df['cig_rec'].isna()].shape)

    df["smoke"] = np.nan
    df.loc[df["cig_rec"] == "Y", "smoke"] = 1
    df.loc[df["cig_rec"] == "N", "smoke"] = 0

    # Sex
    df["sex"] = np.nan
    df.loc[df["infsex"] == 1, "sex"] = "M"
    df.loc[df["infsex"] == 2, "sex"] = "F"
    print(f'\nBad sex data: ',df.loc[df['sex'].isna()].shape)

    # Bed sharing
    df["bed_share"] = np.nan
    df.loc[df["bed_sharing"] == 1, "bed_share"] = "Y"
    df.loc[df["bed_sharing"] == 2, "bed_share"] = "N"
    print(f'\nBad bed share data: ',df.loc[df['bed_share'].isna()].shape)

    df["bed_share_bin"] = np.nan
    df.loc[df["bed_share"] == "Y", "bed_share_bin"] = 1
    df.loc[df["bed_share"] == "N", "bed_share_bin"] = 0

    # Preterm
    # FIXED: preterm is Y if gestational age < 37
    df["preterm"] = np.nan
    df.loc[df["inf3gestage"] < 37, "preterm"] = "Y"
    df.loc[df["inf3gestage"] >= 37, "preterm"] = "N"

    df["preterm_bin"] = np.nan
    df.loc[df["preterm"] == "Y", "preterm_bin"] = 1
    df.loc[df["preterm"] == "N", "preterm_bin"] = 0

    # Final cleanup
    keep_cols = [
        "pca", "infageweeks", "inf3gestage",
        "cig_rec", "smoke",
        "sex",
        "bed_share", "bed_share_bin",
        "preterm", "preterm_bin"
    ]
    # print(f'\nRunning total: ',df.loc[df[keep_cols]].isna())
    df = df.dropna(subset=keep_cols).copy()

    df["pca"] = df["pca"].astype(int)
    df["smoke"] = df["smoke"].astype(int)
    df["bed_share_bin"] = df["bed_share_bin"].astype(int)
    df["preterm_bin"] = df["preterm_bin"].astype(int)

    return df


# -----------------------------------------------------------------------------
# Count structuring
# -----------------------------------------------------------------------------

def structure_ncfrp_data(c0: pd.Series, bed_share: int, smoke: int, preterm: int, var1: str) -> pd.DataFrame:
    out = pd.DataFrame({
        var1: c0.index.values,
        "count": c0.values,
        "bed_share": bed_share,
        "smoke": smoke,
        "preterm": preterm,
    })
    return out.reset_index(drop=True)


def count_ncfrp_data(df_: pd.DataFrame, var1: str) -> pd.DataFrame:
    """
    Build PCA x smoke x preterm x bed_share count table from cases only.
    No zero rows are created.
    Minimum count threshold is enforced after aggregation.
    """
    groups = []

    for bed_share_val, bed_label in [(0, "N"), (1, "Y")]:
        df_bed = df_.loc[df_["bed_share"] == bed_label]

        for smoke_val, smoke_label in [(0, "N"), (1, "Y")]:
            df_smoke = df_bed.loc[df_bed["cig_rec"] == smoke_label]

            for preterm_val, preterm_label in [(0, "N"), (1, "Y")]:
                df_sub = df_smoke.loc[df_smoke["preterm"] == preterm_label]
                counts = df_sub[var1].value_counts().sort_index()

                if len(counts) == 0:
                    continue

                groups.append(
                    structure_ncfrp_data(
                        counts,
                        bed_share=bed_share_val,
                        smoke=smoke_val,
                        preterm=preterm_val,
                        var1=var1
                    )
                )

    if not groups:
        raise ValueError("No grouped count data were created. Check filtering and variable coding.")

    input1 = pd.concat(groups, ignore_index=True)

    # User-validated modeling choice
    input1 = input1.loc[input1["count"] >= MIN_COUNT].copy()
    input1 = input1.sort_values([var1, "bed_share", "smoke", "preterm"]).reset_index(drop=True)
    return input1


def make_ncfrp_input(var1: str, case_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create stratified count table and attach multiplicative exposure offset.
    """
    smoking_exposure, sex_exposure, preterm_exposure, bedshare_exposure = load_exposure()

    input1 = count_ncfrp_data(case_df, var1)
    input1 = input1.dropna(subset=[var1]).copy()

    # Exposure offset:
    # for the all-sex run, do not multiply by sex prevalence
    # for sex-stratified runs, the case_df already defines sex and the sex prevalence
    # should be applied outside this function only if desired.
    input1["exposure"] = (
        input1["smoke"].map(smoking_exposure).astype(float)
        * input1["preterm"].map(preterm_exposure).astype(float)
        * input1["bed_share"].map(bedshare_exposure).astype(float)
    )

    if (input1["exposure"] <= 0).any():
        bad = input1.loc[input1["exposure"] <= 0]
        raise ValueError(f"Non-positive exposure encountered:\n{bad}")

    input1["log_exposure"] = np.log(input1["exposure"])
    return input1


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def check_overdispersion(input_: pd.DataFrame, group_col: str | None = None) -> None:
    if group_col is None:
        mean_ = input_["count"].mean()
        var_ = input_["count"].var(ddof=1)
        print(f"Overall count mean={mean_:.3f}, variance={var_:.3f}")
        return

    for level, sub in input_.groupby(group_col):
        mean_ = sub["count"].mean()
        var_ = sub["count"].var(ddof=1)
        print(f"{group_col}={level}: mean={mean_:.3f}, variance={var_:.3f}")


def smoke_bedshare_case_association(df: pd.DataFrame) -> None:
    ctab = pd.crosstab(df["smoke"], df["bed_share"])
    chi2, p, dof, expected = chi2_contingency(ctab, correction=False)

    print("\nSmoke x bed-share case-only association")
    print(ctab)
    print(f"chi2={chi2:.4f}, dof={dof}, p={p:.6g}")


# -----------------------------------------------------------------------------
# Model output saver wrapper helpers
# -----------------------------------------------------------------------------

def save_nb_model_outputs(model, out_prefix):
    """
    Save:
      1) raw coefficient summary as CSV
      2) IRR table as CSV

    Parameters
    ----------
    model : fitted statsmodels NegativeBinomial result
    out_prefix : str
        Example:
            output_tables/ncfrp/all_early
        This will create:
            all_early_summary.csv
            all_early_irr.csv
    """
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    # -----------------------------
    # 1) raw summary table
    # -----------------------------
    try:
        raw = model.summary2().tables[1].copy()
        raw.to_csv(f"{out_prefix}_summary.csv")
    except Exception:
        # fallback if summary2 acts up
        conf = model.conf_int()
        conf.columns = ["coef_ci_low", "coef_ci_high"]

        raw = pd.DataFrame({
            "coef": model.params,
            "std_err": model.bse,
            "stat": getattr(model, "tvalues", getattr(model, "zvalues", np.nan)),
            "p_value": model.pvalues,
        })
        raw = raw.join(conf)
        raw.index.name = "term"
        raw.to_csv(f"{out_prefix}_summary.csv")

    # -----------------------------
    # 2) IRR table built directly
    # -----------------------------
    conf = model.conf_int().copy()
    conf.columns = ["coef_ci_low", "coef_ci_high"]

    irr = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "stat": getattr(model, "tvalues", getattr(model, "zvalues", np.nan)),
        "p_value": model.pvalues,
    })

    irr = irr.join(conf)

    irr["IRR"] = np.exp(irr["coef"])
    irr["IRR_ci_low"] = np.exp(irr["coef_ci_low"])
    irr["IRR_ci_high"] = np.exp(irr["coef_ci_high"])

    irr.index.name = "term"
    irr.to_csv(f"{out_prefix}_irr.csv")

    print(f'\nIRR {out_prefix}\n{irr.to_string(index=False)}')

    return raw, irr

def save_gam_model_outputs(model, out_prefix):
    """
    Save raw GAM coefficient summary and exponentiated coefficients.
    Note: spline basis terms are usually not interpreted as standalone IRRs,
    but this preserves the fitted output.
    """
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    try:
        raw = model.summary2().tables[1].copy()
        raw.to_csv(f"{out_prefix}_summary.csv")
    except Exception:
        conf = model.conf_int()
        conf.columns = ["coef_ci_low", "coef_ci_high"]

        raw = pd.DataFrame({
            "coef": model.params,
            "std_err": model.bse,
            "stat": getattr(model, "tvalues", getattr(model, "zvalues", np.nan)),
            "p_value": model.pvalues,
        })
        raw = raw.join(conf)
        raw.index.name = "term"
        raw.to_csv(f"{out_prefix}_summary.csv")

    conf = model.conf_int().copy()
    conf.columns = ["coef_ci_low", "coef_ci_high"]

    gam_exp = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "stat": getattr(model, "tvalues", getattr(model, "zvalues", np.nan)),
        "p_value": model.pvalues,
    })

    gam_exp = gam_exp.join(conf)

    gam_exp["exp_coef"] = np.exp(gam_exp["coef"])
    gam_exp["exp_ci_low"] = np.exp(gam_exp["coef_ci_low"])
    gam_exp["exp_ci_high"] = np.exp(gam_exp["coef_ci_high"])

    gam_exp.index.name = "term"
    gam_exp.to_csv(f"{out_prefix}_exp.csv")
    print(f'\nGAM IRR\n{gam_exp.to_string(index=False)}')

    return raw, gam_exp

# -----------------------------------------------------------------------------
# Denominator smoothing helper function
# -----------------------------------------------------------------------------
def get_stratum_offset_from_input(
    input1: pd.DataFrame,
    bed_share: int,
    smoke: int,
    preterm: int,
) -> float:
    sub = input1.loc[
        (input1["bed_share"] == bed_share) &
        (input1["smoke"] == smoke) &
        (input1["preterm"] == preterm)
    ]

    if sub.empty:
        raise ValueError(
            f"No rows found for stratum bed_share={bed_share}, smoke={smoke}, preterm={preterm}"
        )

    vals = sub["log_exposure"].dropna().unique()

    if len(vals) == 0:
        raise ValueError(
            f"No log_exposure found for stratum bed_share={bed_share}, smoke={smoke}, preterm={preterm}"
        )

    if len(vals) > 1:
        print(
            f"Warning: multiple log_exposure values found for stratum "
            f"bed_share={bed_share}, smoke={smoke}, preterm={preterm}: {vals}. "
            f"Using the first one."
        )

    return float(vals[0])

def smooth_rate_table(
    rate_df: pd.DataFrame,
    xvar: str,
    window: int = 3,
    min_periods: int = 1,
    center: bool = True,
) -> pd.DataFrame:
    out = []

    for bed, sub in rate_df.groupby("bed_share", dropna=False):
        sub = sub.sort_values(xvar).copy()
        sub["rate_raw"] = sub["rate"].astype(float)
        sub["rate"] = (
            sub["rate_raw"]
            .rolling(window=window, min_periods=min_periods, center=center)
            .mean()
        )
        out.append(sub)

    out = pd.concat(out, ignore_index=True)
    return out.sort_values([xvar, "bed_share"]).reset_index(drop=True)

def smooth_stratified_counts(
    input_df: pd.DataFrame,
    var1: str = "pca",
    window: int = 3,
    min_periods: int = 1,
    center: bool = True,
    round_counts: bool = False,
) -> pd.DataFrame:
    """
    Smooth aggregated case counts within each PCA x exposure stratum.

    IMPORTANT FOR THIS PROJECT:
    - smooths counts only
    - leaves exposure and log_exposure unchanged
    - appropriate because the jaggedness is in weekly case counts,
      while the offset is a fixed stratum-level exposure weight

    Parameters
    ----------
    input_df : pd.DataFrame
        Must contain:
            [var1, count, exposure, log_exposure, bed_share, smoke, preterm]
    """
    req = [var1, "count", "exposure", "log_exposure", "bed_share", "smoke", "preterm"]
    missing = [c for c in req if c not in input_df.columns]
    if missing:
        raise ValueError(f"smooth_stratified_counts missing required columns: {missing}")

    strata_cols = ["bed_share", "smoke", "preterm"]
    out = []

    for _, sub in input_df.groupby(strata_cols, dropna=False):
        sub = sub.sort_values(var1).copy()
        sub["count_raw"] = sub["count"].astype(float)

        sub["count"] = (
            sub["count_raw"]
            .rolling(window=window, min_periods=min_periods, center=center)
            .mean()
        )

        if round_counts:
            sub["count"] = np.round(sub["count"])

        out.append(sub)

    smoothed = pd.concat(out, ignore_index=True)
    smoothed = smoothed.sort_values([var1, "bed_share", "smoke", "preterm"]).reset_index(drop=True)
    return smoothed

def apply_min_count_threshold(input_df: pd.DataFrame, min_count: float = MIN_COUNT) -> pd.DataFrame:
    out = input_df.loc[input_df["count"] >= min_count].copy()
    return out.reset_index(drop=True)

# -----------------------------------------------------------------------------
# Denominator Figure helpers
# -----------------------------------------------------------------------------

def get_total_births_denominator() -> int:
    """
    Total live births in the denominator population used for descriptive rates.
    """
    den = pd.read_parquet("standardized/consolidated/denominator.parquet")

    # match the same broad gestational-age cleaning logic used in your older workflow
    if "combgest" in den.columns:
        den = den.copy()
        den.loc[den["combgest"] > 44, "combgest"] = np.nan
        den = den.dropna(subset=["combgest"])

    return int(den.shape[0])
# -----------------------------------------------------------------------------
# Figure 2: observed incidence curves by PNA and PCA, stratified by bed sharing
# -----------------------------------------------------------------------------

def build_observed_rate_table(
    case_df: pd.DataFrame,
    week_var: str,
    total_births: int | None = None,
    rate_scale: float = 100000.0,
) -> pd.DataFrame:
    """
    Build descriptive weekly observed rate table stratified by bed sharing.

    For Figure 2, rates are:
        cases / (total live births * bed-share prevalence) * 100,000

    This gives a manuscript-friendly descriptive rate scale.

    Parameters
    ----------
    case_df : pd.DataFrame
        Cleaned case-level NCFRP dataframe.
    week_var : str
        'infageweeks' for postnatal age in weeks, or 'pca' for post-conceptional age.
    total_births : int or None
        Total births in denominator population. If None, will be loaded from
        standardized/consolidated/denominator.parquet.
    rate_scale : float
        Usually 100000.0 for rates per 100,000 live births.
    """
    _, _, _, bedshare_exposure = load_exposure()

    if total_births is None:
        total_births = get_total_births_denominator()

    df = case_df.copy()
    df = df.dropna(subset=[week_var, "bed_share"])
    df = df.loc[df["bed_share"].isin(["Y", "N"])].copy()

    counts = (
        df.groupby([week_var, "bed_share"])
        .size()
        .reset_index(name="cases")
        .sort_values([week_var, "bed_share"])
        .reset_index(drop=True)
    )

    counts["bed_share_num"] = counts["bed_share"].map({"N": 0, "Y": 1})

    # descriptive denominator for each sleep-environment stratum
    counts["denominator"] = counts["bed_share_num"].map(
        lambda x: total_births * float(bedshare_exposure[x])
    )

    counts["rate"] = (counts["cases"] / counts["denominator"]) * rate_scale

    return counts[[week_var, "bed_share", "cases", "denominator", "rate"]].copy()


def plot_incidence_curves_pna_pca(
    case_df: pd.DataFrame,
    title: str,
    total_births: int | None = None,
    rate_scale: float = 100000.0,
):
    """
    Figure 2:
      A. Weekly SIDS incidence by postnatal age (PNA), bed sharing vs non-sharing
      B. Weekly SIDS incidence by post-conceptional age (PCA), bed sharing vs non-sharing

    Rates are descriptive rates per 100,000 live births using:
        total births × bed-share prevalence
    """
    os.makedirs("output_figures/ncfrp", exist_ok=True)

    if total_births is None:
        total_births = get_total_births_denominator()

    pna_tbl = build_observed_rate_table(
        case_df,
        "infageweeks",
        total_births=total_births,
        rate_scale=rate_scale,
    )
    pca_tbl = build_observed_rate_table(
        case_df,
        "pca",
        total_births=total_births,
        rate_scale=rate_scale,
    )

    # smooth plotted rates only
    pna_tbl = smooth_rate_table(pna_tbl, "infageweeks", window=12, min_periods=1, center=True)
    pca_tbl = smooth_rate_table(pca_tbl, "pca", window=5, min_periods=1, center=True)

    # IMPORTANT: do not share y if one curve dominates the panel
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    def _plot_panel(ax, tbl, xvar, xlabel, title):
        ns = tbl.loc[tbl["bed_share"] == "N"].sort_values(xvar)
        bs = tbl.loc[tbl["bed_share"] == "Y"].sort_values(xvar)

        ax.plot(
            ns[xvar], ns["rate"],
            linewidth=2.5, marker="o", markersize=4.5,
            label="Non–bed sharing"
        )
        ax.plot(
            bs[xvar], bs["rate"],
            linewidth=2.5, marker="o", markersize=4.5,
            label="Bed sharing"
        )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)

    _plot_panel(
        axes[0],
        pna_tbl,
        xvar="infageweeks",
        xlabel="Postnatal age (weeks)",
        title="A. SIDS incidence by PNA",
    )

    _plot_panel(
        axes[1],
        pca_tbl,
        xvar="pca",
        xlabel="Post-conceptional age (weeks)",
        title="B. SIDS incidence by PCA",
    )

    axes[0].set_ylabel(f"Weekly SIDS incidence per {int(rate_scale):,} live births", fontsize=12)
    axes[1].set_ylabel(f"Weekly SIDS incidence per {int(rate_scale):,} live births", fontsize=12)

    axes[0].legend(frameon=False, fontsize=10, loc="upper right")

    for ax, label in zip(axes, ["A", "B"]):
        ax.text(
            0.02, 0.98, label,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=14, fontweight="bold"
        )

    fig.savefig(f"output_figures/ncfrp/Figure2_{title}.tiff", dpi=300, format="tiff")
    fig.savefig(f"output_figures/ncfrp/Figure2_{title}.png", dpi=300)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Figure 4: forest plot of IRRs from early and late NB models
# -----------------------------------------------------------------------------
def extract_nb_irr_table(nb_res, window_label):
    """
    Build a tidy IRR table for forest plotting from a fitted NB model.
    """
    params = nb_res.params.copy()
    conf = nb_res.conf_int().copy()
    conf.columns = ["coef_low", "coef_high"]
    pvals = nb_res.pvalues.copy()

    keep_terms = ["pca", "bed_share", "smoke", "preterm"]

    label_map = {
        "pca": "PCA (per week)",
        "bed_share": "Bed sharing",
        "smoke": "Maternal smoking",
        "preterm": "Preterm birth",
    }

    group_map = {
        "pca": "Developmental factor",
        "bed_share": "Sleep environment",
        "smoke": "Perinatal / exposure factors",
        "preterm": "Perinatal / exposure factors",
    }

    rows = []
    for term in keep_terms:
        if term not in params.index:
            continue

        rows.append({
            "term": term,
            "variable": label_map[term],
            "group": group_map[term],
            "coef": params.loc[term],
            "ci_low": conf.loc[term, "coef_low"],
            "ci_high": conf.loc[term, "coef_high"],
            "pvalue": pvals.loc[term],
            "rate_ratio": np.exp(params.loc[term]),
            "rr_ci_low": np.exp(conf.loc[term, "coef_low"]),
            "rr_ci_high": np.exp(conf.loc[term, "coef_high"]),
            "window": window_label,
        })

    df = pd.DataFrame(rows)

    var_order = [
        "PCA (per week)",
        "Bed sharing",
        "Maternal smoking",
        "Preterm birth",
    ]
    df["variable"] = pd.Categorical(df["variable"], categories=var_order, ordered=True)
    df = df.sort_values("variable").reset_index(drop=True)

    return df


def draw_nb_forest_panel(model, ax, title, xmin=0.6, xmax=25):
    """
    Draw one polished forest-plot panel for a single NB model.
    """
    df = extract_nb_irr_table(model, title)

    fp.forestplot(
        dataframe=df,
        estimate="rate_ratio",
        ll="rr_ci_low",
        hl="rr_ci_high",
        varlabel="variable",
        groupvar="group",
        group_order=[
            "Developmental factor",
            "Sleep environment",
            "Perinatal / exposure factors",
        ],
        pval="pvalue",
        color_alt_rows=True,
        xlineval=1,
        ax=ax,
        **{
            "ylabel": "Variables",
            "xlabel": "Adjusted incidence rate ratio",
            "title": title,
        }
    )

    ax.axvline(1, color="gray", linestyle="--", alpha=0.8, linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlim([xmin, xmax])

    # ax.xaxis.set_major_locator(mticker.FixedLocator([0.5, 1, 2, 5, 10, 20]))
    # ax.xaxis.set_major_formatter(mticker.FixedFormatter(["0.5", "1", "2", "5", "10", "20"]))
    # ax.xaxis.set_minor_locator(mticker.NullLocator())

    return ax


def plot_nb_forest(model_early, title, model_late, filename_prefix="Figure4"):
    """
    Paper-style two-panel forest plot.
    """
    os.makedirs("output_figures/ncfrp", exist_ok=True)

    fig, axs = plt.subplots(2, 1, figsize=(11.5, 12))
    fig.subplots_adjust(left=0.42, right=0.94, top=0.96, bottom=0.07, hspace=0.40)

    draw_nb_forest_panel(
        model_early,
        axs[0],
        "Early developmental window",
        xmin=0.5,
        xmax=25,
    )

    draw_nb_forest_panel(
        model_late,
        axs[1],
        "Late developmental window",
        xmin=0.5,
        xmax=25,
    )

    # Panel labels aligned to figure margin
    x_left = 0.01
    for ax, label in zip(axs, ["A", "B"]):
        bbox = ax.get_position()
        fig.text(
            x_left,
            bbox.y1,
            label,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.savefig(f"output_figures/ncfrp/{filename_prefix}_{title}.tiff", dpi=300, format="tiff")
    fig.savefig(f"output_figures/ncfrp/{filename_prefix}_{title}.png", dpi=300)
    plt.close(fig)
# -----------------------------------------------------------------------------
# Piecewise NB model
# -----------------------------------------------------------------------------

def prepare_piecewise_data(
    input1: pd.DataFrame,
    var1: str,
    peak1: int,
    peak2: int,
    peak3: int,
):
    input11 = input1.loc[input1[var1] < peak1].copy()
    input12 = input1.loc[(input1[var1] >= peak2) & (input1[var1] < peak3)].copy()

    offset1 = input11["log_exposure"].copy()
    offset2 = input12["log_exposure"].copy()

    x1 = sm.add_constant(input11[[var1, "bed_share", "smoke", "preterm"]], has_constant="add")
    x2 = sm.add_constant(input12[[var1, "bed_share", "smoke", "preterm"]], has_constant="add")

    y1 = input11["count"].to_numpy(dtype=float)
    y2 = input12["count"].to_numpy(dtype=float)

    return x1, x2, y1, y2, offset1, offset2, input11, input12


def fit_piecewise_nb(
    input1: pd.DataFrame,
    var1: str = "pca",
    peak1: int = 43,
    peak2: int = 47,
    peak3: int = 100,
):
    x1, x2, y1, y2, offset1, offset2, input11, input12 = prepare_piecewise_data(
        input1, var1, peak1, peak2, peak3
    )

    check_overdispersion(input1)
    check_overdispersion(input1, "bed_share")

    model1 = NegativeBinomial(y1, x1, offset=offset1).fit(maxiter=200, disp=False)
    model2 = NegativeBinomial(y2, x2, offset=offset2).fit(maxiter=200, disp=False)

    return {
        "early_model": model1,
        "late_model": model2,
        "early_data": input11,
        "late_data": input12,
        "early_x": x1,
        "late_x": x2,
        "early_y": y1,
        "late_y": y2,
        "early_offset": offset1,
        "late_offset": offset2,
    }


# -----------------------------------------------------------------------------
# Full-curve NB GAM
# -----------------------------------------------------------------------------

def make_knots_from_data(
    x: np.ndarray,
    step: int = 6,
) -> np.ndarray:
    """
    Build sparse interior knots constrained to the observed x-range.
    Simpler knot placement to discourage artificial multi-peak fits.
    """
    x = np.asarray(x, dtype=float).ravel()
    x_min = float(np.min(x))
    x_max = float(np.max(x))

    knots = np.arange(np.ceil(x_min) + step, np.floor(x_max), step, dtype=float)
    knots = knots[(knots > x_min) & (knots < x_max)]

    return knots


def prepare_curve_data_ncfrp(input1: pd.DataFrame, var1: str):
    """
    Full-curve model:
        count ~ linear(bed_share + smoke + preterm)
              + smooth(PCA_main)
              + smooth(PCA_bedshare_deviation)
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


def fit_nb_gam(
    x_linear: pd.DataFrame,
    x_smooth: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    step: int = 6,
):
    degree = 3

    pca_main_vals = np.asarray(x_smooth[:, 0], dtype=float)
    pca_beddev_vals = np.asarray(x_smooth[:, 1], dtype=float)

    knots_main = make_knots_from_data(pca_main_vals, step=step)
    knots_beddev = make_knots_from_data(pca_beddev_vals, step=step)

    if len(knots_main) < 2:
        raise ValueError(
            f"Too few interior knots for main smooth. "
            f"x_min={pca_main_vals.min()}, x_max={pca_main_vals.max()}, knots={knots_main}"
        )

    if len(knots_beddev) < 2:
        raise ValueError(
            f"Too few interior knots for bed-share deviation smooth. "
            f"x_min={pca_beddev_vals.min()}, x_max={pca_beddev_vals.max()}, knots={knots_beddev}"
        )

    df_main = len(knots_main) + degree + 1
    df_beddev = len(knots_beddev) + degree + 1

    smoother = BSplines(
        x_smooth,
        df=[df_main, df_beddev],
        degree=[degree, degree],
        knot_kwds=[{"knots": knots_main}, {"knots": knots_beddev}],
        include_intercept=False,
    )

    family = NB(alpha=0.10)

    model = GLMGam(
        y,
        exog=x_linear,
        smoother=smoother,
        family=family,
        offset=offset,
    )

    res = model.fit()
    return res, {"knots_main": knots_main, "knots_beddev": knots_beddev}

# -----------------------------------------------------------------------------
# Prediction and plotting
# -----------------------------------------------------------------------------

def visualize_curve(x_smooth: np.ndarray, y: np.ndarray, var1: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x_smooth[:, 0], y, s=25)
    ax.set_xlabel(var1)
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(GAM_FIG_DIR / "full_curve_input.png", dpi=300)
    plt.close(fig)


def make_prediction_design(
    pca_grid: np.ndarray,
    bed_share: int,
    smoke: int,
    preterm: int,
    offset_value: float = 0.0,
    pca_anchor: float = 46.0,
):
    exog = pd.DataFrame({
        "Intercept": np.ones_like(pca_grid, dtype=float),
        "bed_share": np.full_like(pca_grid, bed_share, dtype=float),
        "smoke": np.full_like(pca_grid, smoke, dtype=float),
        "preterm": np.full_like(pca_grid, preterm, dtype=float),
    })

    pca_main = pca_grid.astype(float)
    pca_beddev = np.where(
        np.full_like(pca_grid, bed_share, dtype=int) == 1,
        pca_grid.astype(float),
        np.full_like(pca_grid, pca_anchor, dtype=float),
    )

    exog_smooth = np.column_stack([pca_main, pca_beddev])
    offset = np.full_like(pca_grid, offset_value, dtype=float)

    return exog, exog_smooth, offset

def clip_exog_smooth_to_training_range(res, exog_smooth: np.ndarray) -> np.ndarray:
    x_new = np.asarray(exog_smooth, dtype=float).copy()
    x_train = np.asarray(res.model.smoother.x, dtype=float)

    for j in range(x_new.shape[1]):
        lower = np.min(x_train[:, j])
        upper = np.max(x_train[:, j])
        x_new[:, j] = np.clip(x_new[:, j], lower, upper)

    return x_new

def pred_with_ci(res, exog: pd.DataFrame, exog_smooth: np.ndarray, offset: np.ndarray):
    exog_smooth_safe = clip_exog_smooth_to_training_range(res, exog_smooth)

    try:
        pr = res.get_prediction(exog=exog, exog_smooth=exog_smooth_safe, offset=offset)
        sf = pr.summary_frame(alpha=0.05)
        return (
            sf["mean"].to_numpy(),
            sf["mean_ci_lower"].to_numpy(),
            sf["mean_ci_upper"].to_numpy(),
        )
    except Exception as e:
        print("CI not available via get_prediction in this setup:", repr(e))
        mu = res.predict(exog=exog, exog_smooth=exog_smooth_safe, offset=offset)
        return np.asarray(mu), None, None

def plot_gam_curve(
    input1: pd.DataFrame,
    title: str,
    res,
    var1: str = "pca",
    smoke_: int = 0,
    preterm_: int = 0,
):
    """
    Plot Figure 3 using stratum-specific offsets from the modeled data.

    This fixes the vertical overshoot bug by making predictions with the same
    log_exposure scale used during fitting, instead of offset=0.
    """
    pca_grid = np.arange(PCA_MIN, PCA_MAX + 1)

    # Pull the actual stratum-specific log offsets from the modeled input
    offset_ns_val = get_stratum_offset_from_input(
        input1,
        bed_share=0,
        smoke=smoke_,
        preterm=preterm_,
    )
    offset_bs_val = get_stratum_offset_from_input(
        input1,
        bed_share=1,
        smoke=smoke_,
        preterm=preterm_,
    )

    exog_ns, exog_smooth_ns, offset_ns = make_prediction_design(
        pca_grid,
        bed_share=0,
        smoke=smoke_,
        preterm=preterm_,
        offset_value=offset_ns_val,
    )
    exog_bs, exog_smooth_bs, offset_bs = make_prediction_design(
        pca_grid,
        bed_share=1,
        smoke=smoke_,
        preterm=preterm_,
        offset_value=offset_bs_val,
    )

    mu_ns, lo_ns, hi_ns = pred_with_ci(res, exog_ns, exog_smooth_ns, offset_ns)
    mu_bs, lo_bs, hi_bs = pred_with_ci(res, exog_bs, exog_smooth_bs, offset_bs)

    # Observed rows from the same modeled dataset and same covariate stratum
    obs_ns = input1.loc[
        (input1["bed_share"] == 0) &
        (input1["smoke"] == smoke_) &
        (input1["preterm"] == preterm_)
    ].sort_values(var1)

    obs_bs = input1.loc[
        (input1["bed_share"] == 1) &
        (input1["smoke"] == smoke_) &
        (input1["preterm"] == preterm_)
    ].sort_values(var1)

    # obs_ns = (
    #     input1.loc[input1["bed_share"] == 0]
    #     .groupby(var1, as_index=False)['count']
    #     .sum()
    #     .sort_values(var1)
    # )

    # obs_bs = (
    #     input1.loc[input1["bed_share"] == 1]
    #     .groupby(var1, as_index=False)['count']
    #     .sum()
    #     .sort_values(var1)
    # )

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), constrained_layout=True)

    # ------------------------------------------------------------------
    # Panel A: observed vs fitted trajectories
    # ------------------------------------------------------------------
    ax = axes[0]

    # ax.scatter(
    #     obs_ns[var1], obs_ns["count"],
    #     marker="o", s=35, label="Observed non–bed sharing"
    # )
    # ax.scatter(
    #     obs_bs[var1], obs_bs["count"],
    #     marker="^", s=40, label="Observed bed sharing"
    # )
    ax.plot(
        obs_ns[var1], obs_ns["count"],
        marker="v", linewidth=.5, label="Observed non–bed sharing", color='blue'
    )
    ax.plot(
        obs_bs[var1], obs_bs["count"],
        marker="^", linewidth=.5, label="Observed bed sharing", color='orange'
    )

    ax.plot(
        pca_grid, mu_ns,
        linewidth=3, linestyle="-.",label="Predicted mean non–bed sharing", color='blue'
    )
    ax.plot(
        pca_grid, mu_bs,
        linewidth=3, linestyle="--", label="Predicted mean bed sharing", color='orange'
    )

    if lo_ns is not None:
        ax.fill_between(pca_grid, lo_ns, hi_ns, alpha=0.15)
    if lo_bs is not None:
        ax.fill_between(pca_grid, lo_bs, hi_bs, alpha=0.15)

    ax.set_xlabel("PCA (weeks)")
    ax.set_ylabel("SUID case count")
    ax.set_title("A. GAM-estimated SIDS incidence trajectories by sleep environment")
    ax.set_xlim(PCA_MIN - 1, PCA_MAX)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    # ------------------------------------------------------------------
    # Panel B: bed-sharing contrast
    # ------------------------------------------------------------------
    ax2 = axes[1]

    # Predicted contrast
    diff_pred = mu_bs - mu_ns
    ax2.plot(
        pca_grid,
        diff_pred,
        linewidth=3,
        linestyle="--",c='k',
        label="Predicted difference"
    )

    ## Confidence Intervals
    diff_lo = lo_bs - hi_ns
    diff_hi = hi_bs - lo_ns
    if lo_ns is not None and lo_bs is not None:
        ax2.fill_between(
            pca_grid,
            diff_lo,
            diff_hi,
            alpha=0.15,color='k',
            # label="Predicted 95% CI"
        )
    # Observed contrast on overlapping support only
    obs_ns_diff = obs_ns[[var1, "count"]].rename(columns={"count": "count_ns"})
    obs_bs_diff = obs_bs[[var1, "count"]].rename(columns={"count": "count_bs"})

    obs_diff = pd.merge(obs_bs_diff, obs_ns_diff, on=var1, how="inner").sort_values(var1)
    obs_diff["count_diff"] = obs_diff["count_bs"] - obs_diff["count_ns"]

    if not obs_diff.empty:
        # ax2.plot(
        #     obs_diff[var1],
        #     obs_diff["count_diff"],
        #     s=35,
        #     marker="o",
        #     label="Observed difference"
        # )

        # optional light connecting line
        ax2.plot(
            obs_diff[var1],
            obs_diff["count_diff"],
            linewidth=0.5,
            alpha=1,
            color='k',
            label="Observed difference"
        )

    ax2.axhline(0, linestyle="--", linewidth=1.2, alpha=0.8)

    ax2.set_xlabel("PCA (weeks)")
    ax2.set_ylabel("Bed-sharing minus non-sharing count")
    ax2.set_title("B. Bed-sharing minus non-sharing trajectory")
    ax2.set_xlim(PCA_MIN - 1, PCA_MAX)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False)

    # Panel labels
    for ax_i, label in zip(axes, ["A", "B"]):
        ax_i.text(
            0.02, 0.98, label,
            transform=ax_i.transAxes,
            va="top", ha="left",
            fontsize=14, fontweight="bold"
        )

    fig.savefig(GAM_FIG_DIR / f"Figure3_{title}.png", dpi=300)
    fig.savefig(GAM_FIG_DIR / f"Figure3_{title}.tiff", dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Wrapper runners
# -----------------------------------------------------------------------------

def run_piecewise_and_gam(df_run, label):
    print("\n" + "=" * 80)
    print(f"RUN: {label}")
    print("=" * 80)

    var1 = "pca"

    input_df = make_ncfrp_input(var1, df_run)

    print("\nInput head:")
    print(input_df.head())

    # overdispersion summary
    check_overdispersion(input_df)
    check_overdispersion(input_df, "bed_share")

    input_df = smooth_stratified_counts(
        input_df,
        var1=var1,
        window=3,
        min_periods=1,
        center=True,
        round_counts=False
    )

    input_df = apply_min_count_threshold(input_df, min_count=MIN_COUNT)

    # piecewise NB
    x1, x2, y1, y2, offset1, offset2, input11, input12 = prepare_piecewise_data(
        input_df,
        var1=var1,
        peak1=43,
        peak2=47,
        peak3=100,
    )

    model_early = NegativeBinomial(y1, x1, offset=offset1).fit(maxiter=100, disp=False)
    model_late = NegativeBinomial(y2, x2, offset=offset2).fit(maxiter=100, disp=False)

    print(f"\n{label} - early window model")
    print(model_early.summary())

    print(f"\n{label} - late window model")
    print(model_late.summary())

    # full-curve GAM
    curve_df = input_df.copy()
    x_linear, x_smooth, y, offset, curve_df_used = prepare_curve_data_ncfrp(curve_df, var1=var1)
    gam_res, knots = fit_nb_gam(x_linear, x_smooth, y, offset)

    print(f"\n{label} - full curve NB GAM")
    print(gam_res.summary())
    print(f"Knots used: {knots}")

    # manuscript figures only from ALL run
    # if label == "ALL":
    plot_incidence_curves_pna_pca(df_run, label,total_births=None, rate_scale=100000.0)
    plot_nb_forest(model_early, label, model_late, filename_prefix="Figure4")
    plot_gam_curve(curve_df_used, label, gam_res, var1="pca", smoke_=0, preterm_=0)

    out_dir = "output_data/ncfrp"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_slug = label.lower()

    save_nb_model_outputs(model_early, f"{out_dir}/{label_slug}_early")
    save_nb_model_outputs(model_late, f"{out_dir}/{label_slug}_late")
    save_gam_model_outputs(gam_res, f"{out_dir}/{label_slug}_gam")

    return {
        "input_df": input_df,
        "model_early": model_early,
        "model_late": model_late,
        "gam_res": gam_res,
        "knots": knots,
        "early_data": input11,
        "late_data": input12,
        "curve_data": curve_df_used,
    }

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ensure_dirs()
    df = load_ncfrp_data()

    print(f"Loaded cleaned NCFRP dataset: {df.shape}")
    smoke_bedshare_case_association(df)

    # All
    results_all = run_piecewise_and_gam(df, "ALL")

    # # Females
    # results_f = run_piecewise_and_gam(df.loc[df["sex"] == "F"].copy(), "FEMALES")

    # # Males
    # results_m = run_piecewise_and_gam(df.loc[df["sex"] == "M"].copy(), "MALES")

    # df_clean, audit = load_ncfrp_data_with_audit(
    #     DATA_PATH,
    #     pca_min=PCA_MIN,
    #     pca_max=PCA_MAX,
    #     save_report_path="output_data/ncfrp_flow_audit.csv",
    # )

    # print(audit)
    # print("\nFinal analytic shape:", df_clean.shape)

if __name__ == "__main__":
    main()
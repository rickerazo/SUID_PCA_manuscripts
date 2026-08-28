#!/usr/bin/env python3
"""
run_windowed_nb_models.py

Standalone port of the notebook's confirmatory windowed negative binomial
regression (early window PCA<peak1, late window PCA>=peak2), corrected to
use the fixed case-selection pipeline (audit_subject_breakdown.py) and the
fixed exposure pickles (risk_exposure_fixed.py).

Ports, faithfully: structure_data(), count_data(), make_input(),
prepare_data(), and the NegativeBinomial().fit() calls from
SIDS_Collaboration_manuscript_pipeline_v4.ipynb.

WHAT THIS FIXES vs. the notebook
---------------------------------
1. Feeds the model the CORRECTED case series (first-week exclusion actually
   applied; PCA window widened per pca_boundary_report's findings) instead
   of the buggy one that produced the published Figure 4 / Supplemental
   Table 1.

2. `run_binomial_pipeline()` in the notebook calls `prepare_data(..., peak3)`
   using its OWN default `peak3=85`, never overridden by main() (main()
   doesn't pass peak3 at all). That default silently did nothing while
   pca_max was 75 (data never reached 85 anyway), but would silently
   re-truncate the late window back down to <85 now that pca_max is 90.
   This script takes peak3 as an explicit parameter and defaults it to
   pca_max, so widening the window can't be quietly undone downstream.

3. The notebook computed exposure with a per-row `for i, row in
   input1.iterrows(): ...` loop; this uses a vectorized `.map()` instead.
   Same result, much faster, and avoids the SettingWithCopy-prone
   `input1.loc[i, 'exposure'] *= ...` pattern.

VALIDATION CAVEAT
-----------------
Unlike the Figure 1 reconciliation (audit_subject_breakdown.py), this
script CANNOT do an exact digit-for-digit reproduction check against the
published Supplemental Table 1, because the exposure pickles were
overwritten by the risk_exposure fix before this script was written (see
conversation history) — we no longer have the exact exposure values that
generated the published IRRs. The "reproduce old" run below uses the
CORRECTED exposure values with the OLD (buggy) case selection, so a
mismatch against Supplemental Table 1 could come from either source. It's
still useful: large, qualitative agreement (same sign/ballpark IRRs) is a
sanity check that the modeling mechanics (aggregation, offset, design
matrix) are ported correctly; if it looks wildly different in kind (not
just magnitude), stop and investigate before trusting the corrected run.

USAGE
-----
    python run_windowed_nb_models.py \
        --input /path/to/matched_cases_cdc.xlsx \
        --exposure-dir output_data/risk_exposure \
        --outdir output_data/nb_models \
        --compare-buggy
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

# Reuse the already-validated case-selection pipeline instead of duplicating it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_subject_breakdown import (  # noqa: E402
    AuditTrail,
    clean_case_series,
    read_any,
)

LOG = logging.getLogger("run_windowed_nb_models")


# ---------------------------------------------------------------------------
# Ported from the notebook: structure_data / count_data / make_input / prepare_data
# ---------------------------------------------------------------------------

def structure_data(c0: pd.Series, preterm: int, smoke: int, sex: int, var1: str) -> pd.DataFrame:
    return pd.DataFrame({
        var1: c0.index,
        "count": c0.values,
        "preterm": preterm,
        "smoke": smoke,
        "sex": sex,
    })


def count_data(
    df_: pd.DataFrame,
    var1: str,
    *,
    combgest_col: str = "combgest",
    smoking_col: str = "cig_rec",
    sex_col: str = "sex",
    preterm_cutoff: int = 36,
) -> pd.DataFrame:
    """
    Aggregate case-level rows into (var1 value) x (preterm, smoke, sex)
    strata counts. Rows whose smoking_col isn't exactly 'Y' or 'N' are
    implicitly dropped here (they never match d0/d1's masks) — this is
    where the manuscript's "excluded: missing/invalid cig_rec" step
    actually happens, matching count_data() in the notebook exactly.
    """
    pret = df_.loc[df_[combgest_col] <= preterm_cutoff]
    term = df_.loc[df_[combgest_col] > preterm_cutoff]

    d0 = pret.loc[pret[smoking_col] == "N"]
    d1 = pret.loc[pret[smoking_col] == "Y"]
    m10 = d0.loc[d0[sex_col] == "M"]
    f10 = d0.loc[d0[sex_col] == "F"]
    m11 = d1.loc[d1[sex_col] == "M"]
    f11 = d1.loc[d1[sex_col] == "F"]

    d3 = term.loc[term[smoking_col] == "N"]
    d4 = term.loc[term[smoking_col] == "Y"]
    m00 = d3.loc[d3[sex_col] == "M"]
    f00 = d3.loc[d3[sex_col] == "F"]
    m01 = d4.loc[d4[sex_col] == "M"]
    f01 = d4.loc[d4[sex_col] == "F"]

    c01 = m00[var1].value_counts().sort_index()
    c02 = f00[var1].value_counts().sort_index()
    c03 = m01[var1].value_counts().sort_index()
    c04 = f01[var1].value_counts().sort_index()
    c05 = m10[var1].value_counts().sort_index()
    c06 = f10[var1].value_counts().sort_index()
    c07 = m11[var1].value_counts().sort_index()
    c08 = f11[var1].value_counts().sort_index()

    parts = [
        structure_data(c05, 1, 0, 1, var1), structure_data(c06, 1, 0, 0, var1),
        structure_data(c07, 1, 1, 1, var1), structure_data(c08, 1, 1, 0, var1),
        structure_data(c01, 0, 0, 1, var1), structure_data(c02, 0, 0, 0, var1),
        structure_data(c03, 0, 1, 1, var1), structure_data(c04, 0, 1, 0, var1),
    ]
    return pd.concat(parts, ignore_index=True)


def load_exposure(exposure_dir: Path) -> tuple[dict, dict, dict, dict]:
    def _load(name: str) -> dict:
        with open(Path(exposure_dir) / f"{name}.pkl", "rb") as f:
            return pickle.load(f)

    return (
        _load("smoke_exposure"),
        _load("sex_exposure"),
        _load("term_exposure"),
        _load("bedshare_exposure"),
    )


def make_input(
    var1: str,
    case_df: pd.DataFrame,
    exposure_dir: Path,
    *,
    combgest_col: str = "combgest",
    smoking_col: str = "cig_rec",
    sex_col: str = "sex",
) -> pd.DataFrame:
    smoking_exposure, sex_exposure, premature_exposure, _ = load_exposure(exposure_dir)
    input1 = count_data(
        case_df, var1, combgest_col=combgest_col, smoking_col=smoking_col, sex_col=sex_col
    )
    input1 = input1.dropna(subset=[var1]).copy()

    input1["exposure"] = (
        input1["smoke"].map(smoking_exposure)
        * input1["sex"].map(sex_exposure)
        * input1["preterm"].map(premature_exposure)
    )
    if input1["exposure"].isna().any():
        n_bad = int(input1["exposure"].isna().sum())
        raise ValueError(
            f"{n_bad} strata got a NaN exposure — one of the exposure dicts is "
            f"missing a 0/1 key. This is exactly the failure mode the "
            f"risk_exposure.py bug produced; re-check the exposure pickles."
        )
    input1["log_exposure"] = np.log(input1["exposure"])
    return input1


def prepare_data(
    input1: pd.DataFrame,
    var1: str,
    peak1: int,
    peak2: int,
    peak3: int,
    wildcard: str = "preterm",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    input1 = input1.copy()
    input1["sex*preterm"] = input1["sex"] * input1[wildcard]
    input1["smoke*preterm"] = input1["smoke"] * input1[wildcard]

    input11 = input1.loc[input1[var1] < peak1].copy()
    # <=, not <: matches the inclusive upper-bound convention used to build
    # the analytic cohort itself (clean_case_series() keeps PCA <= pca_max).
    # Using '<' here silently dropped infants at exactly PCA == peak3 from
    # the late-window model even though they're part of the reported cohort.
    input12 = input1.loc[(input1[var1] >= peak2) & (input1[var1] <= peak3)].copy()

    offset1 = input11["log_exposure"].reset_index(drop=True)
    offset2 = input12["log_exposure"].reset_index(drop=True)

    cols = [var1, "sex", "smoke", wildcard, "smoke*preterm", "sex*preterm"]
    x1 = sm.add_constant(input11[cols]).reset_index(drop=True)
    x2 = sm.add_constant(input12[cols]).reset_index(drop=True)
    y1 = input11["count"].values
    y2 = input12["count"].values

    return x1, x2, y1, y2, offset1, offset2


PREDICTOR_LABELS = {
    "sex": "Male (vs female)",
    "smoke": "Maternal smoking",
    "preterm": "Preterm birth",
    "smoke*preterm": "Smoking x Preterm",
    "sex*preterm": "Sex x Preterm",
}


def fit_windowed_models(
    case_df: pd.DataFrame,
    *,
    var1: str = "pca",
    peak1: int,
    peak2: int,
    peak3: int,
    exposure_dir: Path,
    combgest_col: str = "combgest",
    smoking_col: str = "cig_rec",
    sex_col: str = "sex",
):
    """
    Reproduces run_binomial_pipeline()'s modeling steps: aggregate, attach
    exposure, split into early/late windows, fit two NegativeBinomial
    models. Returns (input1, x1, x2, y1, y2, model1, model2).
    """
    input1 = make_input(
        var1, case_df, exposure_dir,
        combgest_col=combgest_col, smoking_col=smoking_col, sex_col=sex_col,
    )
    x1, x2, y1, y2, offset1, offset2 = prepare_data(input1, var1, peak1, peak2, peak3)

    LOG.info("Early window: %d strata rows, %d cases", len(x1), int(y1.sum()))
    LOG.info("Late  window: %d strata rows, %d cases", len(x2), int(y2.sum()))

    model1 = NegativeBinomial(y1, x1, offset=offset1).fit(maxiter=200, disp=False)
    model2 = NegativeBinomial(y2, x2, offset=offset2).fit(maxiter=200, disp=False)

    return input1, x1, x2, y1, y2, model1, model2


def irr_table(model, var1: str, window_n_cases: int) -> pd.DataFrame:
    """Rate-ratio table in the same shape as the manuscript's Supplemental Table 1."""
    params = model.params
    ci = model.conf_int()
    pvals = model.pvalues

    rows = []
    for name in params.index:
        if name in ("const",):
            continue
        if name == "alpha":
            continue
        label = PREDICTOR_LABELS.get(name, "PCA (per week)" if name == var1 else name)
        rows.append({
            "Predictor": label,
            "IRR": np.exp(params[name]),
            "CI_low": np.exp(ci.loc[name, 0]),
            "CI_high": np.exp(ci.loc[name, 1]),
            "p": pvals[name],
        })

    alpha_val = params["alpha"] if "alpha" in params.index else np.nan
    rows.append({"Predictor": "Alpha (overdispersion)", "IRR": alpha_val,
                 "CI_low": np.nan, "CI_high": np.nan, "p": np.nan})
    rows.append({"Predictor": "Observations (strata rows)", "IRR": model.nobs,
                 "CI_low": np.nan, "CI_high": np.nan, "p": np.nan})
    rows.append({"Predictor": "Total cases in window", "IRR": window_n_cases,
                 "CI_low": np.nan, "CI_high": np.nan, "p": np.nan})
    rows.append({"Predictor": "Log-likelihood", "IRR": model.llf,
                 "CI_low": np.nan, "CI_high": np.nan, "p": np.nan})
    pr2 = getattr(model, "prsquared", np.nan)
    rows.append({"Predictor": "Pseudo-R2", "IRR": pr2,
                 "CI_low": np.nan, "CI_high": np.nan, "p": np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit the confirmatory windowed negative binomial models on the "
                    "corrected case series.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                    help="Path to the raw case file, e.g. matched_cases_cdc.xlsx")
    p.add_argument("--exposure-dir", type=Path, default=Path("output_data/risk_exposure"),
                    help="Directory containing smoke/sex/term/bedshare_exposure.pkl")
    p.add_argument("--outdir", type=Path, default=Path("./nb_model_output"))
    p.add_argument("--compare-buggy", action="store_true",
                    help="Also fit on the original (bug-active, pca_max=75 inclusive, "
                         "stale peak3=85) case series for comparison, using the SAME "
                         "(corrected) exposure pickles — see the module docstring's "
                         "validation caveat before over-interpreting any exact match.")

    p.add_argument("--infage-col", default="infage")
    p.add_argument("--combgest-col", default="combgest")
    p.add_argument("--sex-col", default="sex")
    p.add_argument("--smoking-col", default="cig_rec")
    p.add_argument("--pca-min", type=int, default=36)
    p.add_argument("--pca-max", type=int, default=90)
    p.add_argument("--peak1", type=int, default=43)
    p.add_argument("--peak2", type=int, default=47)
    p.add_argument("--peak3", type=int, default=None,
                    help="Upper truncation for the late window before fitting. Defaults "
                         "to --pca-max. The notebook's run_binomial_pipeline() left this "
                         "at a stale default of 85, which would silently re-truncate the "
                         "late window if left unset while pca_max=90.")
    return p


def _run_one(raw_df, args, *, outdir: Path, label: str, reproduce_original_bug: bool,
             pca_max: int, pca_max_inclusive: bool, peak3: int):
    outdir.mkdir(parents=True, exist_ok=True)
    case_df, _, trail = clean_case_series(
        raw_df,
        infage_col=args.infage_col,
        combgest_col=args.combgest_col,
        pca_min=args.pca_min,
        pca_max=pca_max,
        pca_max_inclusive=pca_max_inclusive,
        reproduce_original_bug=reproduce_original_bug,
    )
    trail.to_frame().to_csv(outdir / f"audit_trail_{label}.csv", index=False)

    input1, x1, x2, y1, y2, model1, model2 = fit_windowed_models(
        case_df,
        peak1=args.peak1, peak2=args.peak2, peak3=peak3,
        exposure_dir=args.exposure_dir,
        combgest_col=args.combgest_col, smoking_col=args.smoking_col, sex_col=args.sex_col,
    )

    t1 = irr_table(model1, "pca", int(y1.sum()))
    t2 = irr_table(model2, "pca", int(y2.sum()))
    t1.to_csv(outdir / f"irr_early_window_{label}.csv", index=False)
    t2.to_csv(outdir / f"irr_late_window_{label}.csv", index=False)

    with open(outdir / f"model_summary_early_{label}.txt", "w") as f:
        f.write(str(model1.summary()))
    with open(outdir / f"model_summary_late_{label}.txt", "w") as f:
        f.write(str(model2.summary()))

    LOG.info("[%s] Early window IRR table:\n%s", label, t1.to_string(index=False))
    LOG.info("[%s] Late  window IRR table:\n%s", label, t2.to_string(index=False))
    return t1, t2


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    peak3 = args.peak3 if args.peak3 is not None else args.pca_max

    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.outdir / "nb_model_run.log")],
    )

    if not args.input.exists():
        LOG.error("Input file not found: %s", args.input)
        return 1
    for name in ("smoke_exposure", "sex_exposure", "term_exposure", "bedshare_exposure"):
        p = args.exposure_dir / f"{name}.pkl"
        if not p.exists():
            LOG.error("Missing exposure pickle: %s", p)
            return 1

    LOG.info("Loading %s", args.input)
    raw_df = read_any(args.input)
    LOG.info("Raw shape: %s", raw_df.shape)
    LOG.info("Using peak3=%d for the late-window truncation (pca_max=%d)", peak3, args.pca_max)

    if args.compare_buggy:
        LOG.info("=" * 78)
        LOG.info("Fitting on the ORIGINAL (bug-active) case series, corrected exposure "
                 "pickles. See module docstring's validation caveat.")
        _run_one(raw_df, args, outdir=args.outdir / "original_buggy", label="original_buggy",
                  reproduce_original_bug=True, pca_max=75, pca_max_inclusive=True, peak3=85)

    LOG.info("=" * 78)
    LOG.info("Fitting on the CORRECTED case series (pca_max=%d, peak3=%d)",
              args.pca_max, peak3)
    _run_one(raw_df, args, outdir=args.outdir / "corrected", label="corrected",
              reproduce_original_bug=False, pca_max=args.pca_max, pca_max_inclusive=False,
              peak3=peak3)

    LOG.info("=" * 78)
    LOG.info("DONE. See %s", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

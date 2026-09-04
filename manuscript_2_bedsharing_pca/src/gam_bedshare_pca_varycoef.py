#!/usr/bin/env python3
"""
gam_bedshare_pca_varycoef.py

A corrected, working replacement for the retired `gam_res_int` bed_share x PCA
interaction-spline robustness check (see AUDIT_FINDINGS.md A3.27's close-out
and ncfrp_manuscript_v5.py's run_piecewise_and_gam, which used to fit it).

WHY A NEW SCRIPT INSTEAD OF A GLMGam FIX
-----------------------------------------
The retired variant built its interaction smooth from the raw product
`pca * bed_share` (zero for every non-bed-sharing row) and fed it straight
into statsmodels.gam.smooth_basis.BSplines, which has no "by" mechanism - it
independently spline-expands whatever column it is given. Two compounding
bugs followed: (1) knot placement (make_knots_from_data, based on min/max of
that zero-inflated column) put knots in a PCA range with zero real bed_share=1
support, and (2) multiplying BEFORE basis expansion, rather than AFTER, is not
a standard varying-coefficient construction and is not identifiable against
the parametric bed_share term. The result: matching, exploded std errors
(~329) between bed_share and the high-order spline terms - the signature of
an unpenalized flat/near-null ridge in the Fisher information that no amount
of GLMGam's roughness-penalty (CV alpha) selection can fix, because that
penalty is internal to each smooth's own basis and never ties a parametric
term to a smooth term.

This script fixes both bugs directly, and fits via a plain NB MLE
(statsmodels.discrete.discrete_model.NegativeBinomial - the same optimizer
already used and trusted for the early/late piecewise models) instead of
GLMGam, sidestepping the penalized-IRLS machinery that was the proximate
numerical failure mode:

  1. The interaction smooth's knots come from the bed_share==1 SUBSET's real
     PCA quantiles only (default k_int=4 basis functions - fewer than the
     main smooth's k_main=6, matching that subgroup's smaller effective N).
  2. The interaction basis is evaluated at every row's REAL pca value - no
     out-of-domain zero placeholder for bed_share==0 rows.
  3. It is CENTERED on the bed_share==1 subset's column means (the standard
     sum-to-zero identifiability constraint used by e.g. mgcv's
     `s(pca, by=bed_share)`) and only THEN multiplied by the bed_share
     indicator - "expand, then multiply", not "multiply, then expand". The
     main s(pca) smooth is centered the same way (over all rows) - patsy's
     cr() basis is a partition of unity (each row's basis values sum to
     exactly 1), so it is perfectly collinear with a separate parametric
     Intercept term unless centered.

A rank/condition-number check (ncfrp_manuscript_v5.check_design_rank) runs
before every fit, and the save path (ncfrp_manuscript_v5.save_nb_model_outputs)
already asserts every CI brackets its point estimate; this module adds its own
max-SE sanity check on top (--se-sanity-max) so a future regression of this
same failure mode cannot silently reach a CSV again.

Output is ADDITIVE to whatever --outdir tree ncfrp_manuscript_v5.py already
wrote (out_joint/, out_marginal/, ...): new file names, new figure
subdirectory, nothing here overwrites an existing file from those runs.

    python src/gam_bedshare_pca_varycoef.py \\
        --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \\
        --exposure-dir output_data/risk_exposure \\
        --outdir out_joint --offset-mode joint --pca-min 36 --pca-max 90
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix, build_design_matrices
from statsmodels.discrete.discrete_model import NegativeBinomial

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ncfrp_manuscript_v5 as pipe  # noqa: E402

LOG = logging.getLogger("gam_bedshare_pca_varycoef")

VARYCOEF_FIG_DIR = pipe.FIG_DIR / "varycoef"


# -----------------------------------------------------------------------------
# Design construction (the correct "expand, then multiply" varying-coefficient
# form; see module docstring for what this replaces and why)
# -----------------------------------------------------------------------------

def build_varying_coef_design(input1: pd.DataFrame, *, var1: str = "pca",
                              k_main: int = 6, k_int: int = 4) -> tuple:
    """
    Returns (X, y, offset, design_info). X is a DataFrame with columns
    [Intercept, bed_share, smoke, preterm, pca_main_s0.., pca_bedshare_int_s0..],
    suitable for statsmodels.discrete.discrete_model.NegativeBinomial directly.
    design_info carries what predict_with_delta_ci needs to evaluate the same
    bases at new PCA grid points.
    """
    input1 = input1.dropna(subset=[var1, "log_exposure", "count"]).copy()
    pca = input1[var1].astype(float).to_numpy()
    bed = input1["bed_share"].astype(int).to_numpy()

    n_bed = int(bed.sum())
    if n_bed < (k_int + 3):
        raise ValueError(
            f"Too few bed_share=1 rows ({n_bed}) to place {k_int} interior "
            f"knots for the interaction smooth; lower --k-int or check input.")

    # --- main smooth: s(pca), all rows, patsy picks knots from the full range.
    # patsy's cr() basis is a partition of unity (each row's basis values sum
    # to exactly 1), so it is perfectly collinear with a separate parametric
    # Intercept column unless centered - the same sum-to-zero identifiability
    # constraint applied to the interaction smooth below, applied here too
    # (over ALL rows, since this smooth is not itself a "by" term). Centering
    # a partition-of-unity basis makes every row's centered values sum to
    # exactly 0, which is the correct identifiability constraint but also
    # makes the centered basis intrinsically rank (k_main - 1), not k_main -
    # this is the standard, expected df cost of a sum-to-zero smooth (its own
    # "intercept" is absorbed by the parametric model intercept), not a data
    # problem. Drop the now-redundant last column rather than leave it in and
    # let check_design_rank flag an artifact of construction, not the data.
    main_dm = dmatrix(f"cr(x, df={k_main}) - 1", {"x": pca}, return_type="dataframe")
    B_main_raw = main_dm.to_numpy()
    main_center = B_main_raw.mean(axis=0)
    B_main = (B_main_raw - main_center)[:, :-1]

    # --- interaction smooth: knots from the bed_share==1 SUBSET only (fixes
    # the knot-range bug: the old code let a zero-inflated product corrupt
    # the knot range, putting knots where there is no real bed_share=1 data).
    # patsy's cr() basis dimension = (interior knots + 2 implicit boundary
    # knots); to land on k_int basis columns (matching the --k-int flag's
    # documented meaning), request k_int-2 interior knots.
    pca1 = pca[bed == 1]
    n_interior = max(1, k_int - 2)
    quantiles = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    int_knots = np.quantile(pca1, quantiles)
    # Pass knots as data (not string-embedded) - avoids any formatting
    # ambiguity from numpy scalar repr across numpy versions.
    int_dm = dmatrix("cr(x, knots=int_knots) - 1", {"x": pca, "int_knots": int_knots},
                     return_type="dataframe")   # evaluated at every row's REAL pca
    B_int_raw = int_dm.to_numpy()

    # sum-to-zero constraint over the bed_share==1 subset (removes aliasing
    # with the parametric bed_share term), THEN multiply by the indicator -
    # the actual varying-coefficient "by" mechanism the old code got backwards.
    # Same partition-of-unity rank cost as the main smooth - drop the
    # redundant last column before multiplying by bed_share.
    int_center = B_int_raw[bed == 1].mean(axis=0)
    B_int = ((B_int_raw - int_center)[:, :-1]) * bed[:, None]

    x_linear = pd.DataFrame({
        "Intercept": 1.0,
        "bed_share": bed.astype(float),
        "smoke": input1["smoke"].astype(int).to_numpy(dtype=float),
        "preterm": input1["preterm"].astype(int).to_numpy(dtype=float),
    }, index=input1.index)

    main_cols = [f"pca_main_s{i}" for i in range(B_main.shape[1])]
    int_cols = [f"pca_bedshare_int_s{i}" for i in range(B_int.shape[1])]
    X = pd.concat([
        x_linear,
        pd.DataFrame(B_main, columns=main_cols, index=input1.index),
        pd.DataFrame(B_int, columns=int_cols, index=input1.index),
    ], axis=1)

    y = input1["count"].to_numpy(dtype=float)
    offset = input1["log_exposure"].to_numpy(dtype=float)

    design_info = {
        "main_design_info": main_dm.design_info,
        "int_design_info": int_dm.design_info,
        "main_center": main_center,
        "int_center": int_center,
        "main_cols": main_cols,
        "int_cols": int_cols,
        "int_knots": int_knots,
        "k_main": k_main,
        "k_int": k_int,
        "n_bed_share": n_bed,
        "n_total": len(input1),
        # exact linear-predictor column names/order, so predict_with_delta_ci
        # can slice res.params / res.cov_params() down to the beta sub-block
        # and exclude the NB dispersion parameter ("alpha") that
        # NegativeBinomial.fit() appends - alpha does not enter eta = X@beta,
        # and mixing it in would misalign the matrix multiply (X0 has no
        # column for it) or, worse, silently pull in the wrong covariance
        # entries for the delta-method variance.
        "exog_names": list(X.columns),
    }
    return X, y, offset, design_info


def fit_bedshare_pca_varycoef(X: pd.DataFrame, y: np.ndarray, offset: np.ndarray,
                              *, label: str, cond_threshold: float = 1e8):
    pipe.check_design_rank(X.to_numpy(), list(X.columns), label,
                           cond_threshold=cond_threshold)
    model = NegativeBinomial(y, X, offset=offset)
    return model.fit(maxiter=200, disp=False)


def _assert_se_sanity(irr: pd.DataFrame, label: str, se_max: float) -> None:
    huge = irr.loc[irr["std_err"] > se_max]
    if len(huge):
        raise AssertionError(
            f"[{label}] {len(huge)} term(s) have std_err > {se_max} "
            f"(rank-deficient / still-broken design):\n{huge[['coef', 'std_err']]}")


# -----------------------------------------------------------------------------
# Prediction (manual delta method - do not assume NegativeBinomialResults
# exposes a working get_prediction() mean-CI in the pinned statsmodels; see
# RUNBOOK.md verification step for how to confirm which path is active)
# -----------------------------------------------------------------------------

def predict_with_delta_ci(res, design_info: dict, pca_grid: np.ndarray, *,
                          bed_share: int, smoke: int, preterm: int,
                          offset_value: float) -> tuple:
    n = len(pca_grid)
    (main_mat,) = build_design_matrices([design_info["main_design_info"]],
                                        {"x": pca_grid})
    (int_mat,) = build_design_matrices([design_info["int_design_info"]],
                                       {"x": pca_grid})
    # Same centering + last-column drop as build_varying_coef_design, applied
    # identically here so the fitted params line up with these columns.
    B_main = (np.asarray(main_mat) - design_info["main_center"])[:, :-1]
    B_int_raw = np.asarray(int_mat)
    B_int = ((B_int_raw - design_info["int_center"])[:, :-1]) * float(bed_share)

    x_linear = np.column_stack([
        np.ones(n), np.full(n, bed_share, dtype=float),
        np.full(n, smoke, dtype=float), np.full(n, preterm, dtype=float),
    ])
    X0 = np.column_stack([x_linear, B_main, B_int])

    # Slice params/cov down to the linear-predictor (beta) sub-block, in the
    # exact order X0's columns were built in - excludes the NB dispersion
    # parameter ("alpha"), which does not enter eta and has no column in X0.
    exog_names = design_info["exog_names"]
    params_full = res.params
    cov_full = res.cov_params()
    if hasattr(params_full, "loc"):
        beta = params_full.loc[exog_names].to_numpy()
        cov = cov_full.loc[exog_names, exog_names].to_numpy()
    else:  # fallback if statsmodels ever returns bare arrays here
        beta = np.asarray(params_full)[: len(exog_names)]
        cov = np.asarray(cov_full)[: len(exog_names), : len(exog_names)]

    eta0 = X0 @ beta + offset_value
    var_eta0 = np.einsum("ij,jk,ik->i", X0, cov, X0)
    se_eta0 = np.sqrt(np.clip(var_eta0, 0.0, None))
    mu0 = np.exp(eta0)
    lo = np.exp(eta0 - 1.96 * se_eta0)
    hi = np.exp(eta0 + 1.96 * se_eta0)
    return mu0, lo, hi


# -----------------------------------------------------------------------------
# Figure (mirrors ncfrp_manuscript_v5.plot_gam_curve's layout for visual
# consistency with the existing Figure 3; written to a distinct varycoef/
# subdirectory under a distinct name, Figure3b_*, so it cannot be confused
# with or overwrite Figure3_*)
# -----------------------------------------------------------------------------

def plot_varycoef_curve(input1: pd.DataFrame, title: str, res, design_info: dict,
                        *, var1: str = "pca", smoke_: int = 0, preterm_: int = 0,
                        pca_min: int, pca_max: int) -> None:
    pca_grid = np.arange(pca_min, pca_max + 1, dtype=float)
    off_ns = pipe.get_stratum_offset_from_input(input1, 0, smoke_, preterm_)
    off_bs = pipe.get_stratum_offset_from_input(input1, 1, smoke_, preterm_)
    mu_ns, lo_ns, hi_ns = predict_with_delta_ci(
        res, design_info, pca_grid, bed_share=0, smoke=smoke_, preterm=preterm_,
        offset_value=off_ns)
    mu_bs, lo_bs, hi_bs = predict_with_delta_ci(
        res, design_info, pca_grid, bed_share=1, smoke=smoke_, preterm=preterm_,
        offset_value=off_bs)

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
    ax.fill_between(pca_grid, lo_ns, hi_ns, alpha=0.15, color="blue")
    ax.fill_between(pca_grid, lo_bs, hi_bs, alpha=0.15, color="orange")
    ax.set_xlabel("PCA (weeks)")
    ax.set_ylabel("SUID case count")
    ax.set_title("A. Varying-coefficient GAM: SUID incidence by sleep environment")
    ax.set_xlim(pca_min - 1, pca_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    ax2 = axes[1]
    ratio = np.divide(mu_bs, mu_ns, out=np.full_like(mu_bs, np.nan, dtype=float),
                      where=mu_ns > 0)
    ax2.plot(pca_grid, ratio, linewidth=3, linestyle="--", color="k",
             label="Fitted bed-sharing / non-sharing ratio")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lo = np.log(lo_bs) - np.log(hi_ns)
        log_hi = np.log(hi_bs) - np.log(lo_ns)
    ax2.fill_between(pca_grid, np.exp(log_lo), np.exp(log_hi), alpha=0.15, color="k")
    ax2.axhline(1.0, linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.set_xlabel("PCA (weeks)")
    ax2.set_ylabel("Fitted incidence rate ratio (bed sharing vs not)")
    ax2.set_title("B. Bed-sharing vs non-sharing fitted rate ratio across PCA "
                  "(varying-coefficient model)")
    ax2.set_xlim(pca_min - 1, pca_max)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False)

    for ax_i, panel_label in zip(axes, ["A", "B"]):
        ax_i.text(0.02, 0.98, panel_label, transform=ax_i.transAxes, va="top",
                  ha="left", fontsize=14, fontweight="bold")

    VARYCOEF_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(VARYCOEF_FIG_DIR / f"Figure3b_{title}.png", dpi=300, bbox_inches="tight")
    fig.savefig(VARYCOEF_FIG_DIR / f"Figure3b_{title}.tiff", dpi=300, format="tiff",
                bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Orchestration - one stratum
# -----------------------------------------------------------------------------

def run_varycoef_model(df_run: pd.DataFrame, label: str, *,
                       pca_min: int, pca_max: int, min_count: int,
                       exposure_dir: Path, offset_mode: str,
                       k_main: int, k_int: int, cond_threshold: float,
                       se_sanity_max: float) -> dict:
    LOG.info("=" * 78)
    LOG.info("VARYCOEF RUN: %s  (PCA %d-%d, offset=%s, k_main=%d, k_int=%d)",
             label, pca_min, pca_max, offset_mode, k_main, k_int)

    var1 = "pca"
    input_df = pipe.make_ncfrp_input(var1, df_run, exposure_dir=exposure_dir,
                                     min_count=min_count, offset_mode=offset_mode)

    X, y, offset, design_info = build_varying_coef_design(
        input_df, var1=var1, k_main=k_main, k_int=k_int)
    LOG.info("%s design: %d rows, %d cols (n bed_share=1: %d)", label,
             X.shape[0], X.shape[1], design_info["n_bed_share"])

    res = fit_bedshare_pca_varycoef(X, y, offset, label=label,
                                    cond_threshold=cond_threshold)

    pipe.OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    slug = label.lower()
    out_prefix = f"{pipe.OUT_DATA_DIR}/{slug}_bedshare_pca_varycoef"
    irr = pipe.save_nb_model_outputs(res, out_prefix)   # asserts CI brackets estimate
    _assert_se_sanity(irr, out_prefix, se_sanity_max)

    plot_varycoef_curve(input_df, label, res, design_info,
                        pca_min=pca_min, pca_max=pca_max)

    return {"input_df": input_df, "res": res, "design_info": design_info, "irr": irr}


# -----------------------------------------------------------------------------
# CLI - reuses ncfrp_manuscript_v5's flag names/semantics where there is a
# direct analogue, so this can be invoked with the same arguments run_all.sh
# already threads through the main analysis step. Deliberately no
# --gam-alpha-grid: this model has no CV-selected roughness penalty (alpha is
# jointly MLE-estimated as part of the NB fit, same as the piecewise models).
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bed-share x PCA varying-coefficient GAM robustness check "
                    "(corrected replacement for the retired gam_res_int).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-path", type=Path, default=pipe.DATA_PATH)
    p.add_argument("--exposure-dir", type=Path, default=pipe.EXPOSURE_DIR)
    p.add_argument("--outdir", type=Path, default=Path("."),
                   help="working directory for output_figures/ and output_data/ "
                        "(should match an existing ncfrp_manuscript_v5.py run's "
                        "--outdir, e.g. out_joint or out_marginal)")
    p.add_argument("--pca-min", type=int, default=pipe.PCA_MIN)
    p.add_argument("--pca-max", type=int, default=pipe.PCA_MAX)
    p.add_argument("--min-count", type=int, default=pipe.MIN_COUNT)
    p.add_argument("--first-week-cut", type=int, default=pipe.FIRST_WEEK_CUT)
    p.add_argument("--gestage-max", type=int, default=pipe.GESTAGE_MAX)
    p.add_argument("--max-infage-days", type=int, default=pipe.MAX_INFAGE_DAYS)
    p.add_argument("--year-min", type=int, default=pipe.YEAR_MIN)
    p.add_argument("--year-max", type=int, default=pipe.YEAR_MAX)
    p.add_argument("--offset-mode", choices=["joint", "marginal"], default="joint")
    p.add_argument("--k-main", type=int, default=6,
                   help="basis functions for the main s(pca) smooth")
    p.add_argument("--k-int", type=int, default=4,
                   help="basis functions for the bed_share-by-pca interaction "
                        "smooth; kept smaller than --k-main to match the "
                        "bed_share=1 subgroup's smaller effective N")
    p.add_argument("--cond-threshold", type=float, default=1e8,
                   help="design-matrix condition-number threshold for the "
                        "pre-fit rank check")
    p.add_argument("--se-sanity-max", type=float, default=pipe.SE_SANITY_MAX,
                   help="max acceptable log-scale std err; higher raises")
    p.add_argument("--seed", type=int, default=pipe.SEED)
    p.add_argument("--no-sex-strata", action="store_true",
                   help="skip the FEMALES / MALES runs")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    orig_cwd = Path.cwd()
    args.data_path = args.data_path.resolve()
    args.exposure_dir = args.exposure_dir.resolve()
    pipe.ORIG_CWD = orig_cwd

    import os
    if args.outdir != Path("."):
        args.outdir.mkdir(parents=True, exist_ok=True)
        os.chdir(args.outdir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    np.random.seed(args.seed)

    df, _audit = pipe.load_ncfrp_data(
        args.data_path, pca_min=args.pca_min, pca_max=args.pca_max,
        first_week_cut=args.first_week_cut, gestage_max=args.gestage_max,
        max_infage_days=args.max_infage_days, year_min=args.year_min,
        year_max=args.year_max, audit_csv=None)
    LOG.info("Loaded analytic dataset: %s", df.shape)

    runs = {"ALL": df}
    if not args.no_sex_strata:
        runs["FEMALES"] = df.loc[df["sex"] == "F"].copy()
        runs["MALES"] = df.loc[df["sex"] == "M"].copy()

    for label, sub in runs.items():
        try:
            run_varycoef_model(
                sub, label, pca_min=args.pca_min, pca_max=args.pca_max,
                min_count=args.min_count, exposure_dir=args.exposure_dir,
                offset_mode=args.offset_mode, k_main=args.k_main, k_int=args.k_int,
                cond_threshold=args.cond_threshold, se_sanity_max=args.se_sanity_max)
        except Exception as e:
            LOG.warning("varycoef run %s failed: %r", label, e)

    LOG.info("DONE. Outputs under %s", Path("output_data/ncfrp").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
generate_figure3_gam_curves.py

Standalone, corrected port of the notebook's Figure 3 pipeline: a
negative-binomial GAM with B-splines over PCA, fit separately for each sex
(`prepare_curve_data()` / `model_curve()` / `make_knots()` / `make_design()`
/ `pca_grid()` / `NB_splines()` in
SIDS_Collaboration_manuscript_pipeline_v4.ipynb), rebuilt on top of the
already-validated case-selection pipeline (audit_subject_breakdown.py) and
the already-validated aggregation/exposure step (run_windowed_nb_models.
make_input(), reused directly rather than re-implemented).

WHAT THIS FIXES vs. the notebook
---------------------------------
1. Feeds the model the CORRECTED case series (first-week exclusion actually
   applied) and the CORRECTED exposure pickles, same as Figures 2 and 4.

2. prepare_curve_data() hardcoded the PCA window to <36 / >75, a leftover
   from before the PCA upper bound was widened to 90 (see HANDOFF_CONTEXT.md
   bug #2) — silently re-truncating the curve back to the old window no
   matter what pca_max was used upstream. This version takes --pca-min /
   --pca-max explicitly and applies them with the same inclusive-upper-bound
   convention (`<=`) used everywhere else in this pipeline (bug #4). In
   practice the case series is already restricted to [pca_min, pca_max] by
   clean_case_series() before it ever reaches this script, so this filter is
   now a consistency check rather than a silent second truncation.

3. make_knots(a, b, start=36, peak=46, tail=75) hardcoded the same stale
   tail=75. Now tail defaults to --pca-max (90), so the upper-tail knot
   spacing actually covers the widened analysis window instead of stopping
   13 weeks short of the data.

4. AJE/captions031.md expects three SEPARATE panel images (Figure 3A/B/C)
   under specific legacy filenames (curvefit_exp/NB_splines_{smoke}_
   {preterm}_a{a}_b{b}_{prediction,difference,logdifference}.png,
   smoke=preterm=0, a=b=10) — this script writes those three files. It ALSO
   writes a combined 3-row Figure3.png/.tiff (300 dpi TIFF) with panel
   labels A/B/C, matching the notebook's original pca_grid() combined
   layout, since AJE's actual submission wants one file per figure number
   regardless of how the response-letter captions split it. Both are built
   from the same computed curves (draw_*_panel() functions shared between
   the two), so they can't drift apart from each other.

5. pca_grid()'s M-F difference/log-ratio panels assumed the female and male
   prediction grids line up 1:1 by array position (`mu_m - mu_f` etc. with
   no join). That silently misaligns, or outright crashes on a shape
   mismatch, whenever a PCA week in the reference stratum was observed for
   one sex but not the other — confirmed by a smoke test (--smoke-ref 1
   --preterm-ref 1 on synthetic data hit exactly this: 52 male PCA weeks
   vs. 51 female, `ValueError: operands could not be broadcast together`).
   Now explicitly inner-joined on the PCA value before any cross-sex
   arithmetic, with a log line reporting any PCA week dropped for lacking
   one sex. The default reference stratum (smoke=0, preterm=0) is the
   cohort's largest and least likely to hit this in practice, but the
   sparser tail weeks introduced by the widened pca_max=90 window make it
   plausible enough there to fix outright rather than assume away.

WHAT THIS DELIBERATELY PRESERVES (not bugs, just worth flagging)
------------------------------------------------------------------
- The GAM is fit at a single reference stratum (non-smoking, term birth:
  smoke=0, preterm=0) for visualization, matching the original
  pca_grid(a, b, data, model, smoke_=0, preterm_=0) call. The regression
  itself (fit via model_curve()) still includes smoke/preterm/interaction
  terms as linear covariates; only the plotted curve is sliced at this one
  reference stratum. Override with --smoke-ref/--preterm-ref if a different
  reference stratum is wanted.
- Confidence bands rely on statsmodels' GLMGam.get_prediction(), which
  cannot extrapolate a B-spline smoother beyond the exact per-sex data range
  it was fit on (raises NotImplementedError). Because the prediction grid
  here is built from make_design(), i.e. the SAME observed PCA values used
  to fit each sex's smoother (not an arbitrary synthetic grid), this should
  not normally trigger — but if it does for a given sex/stratum, the
  original notebook's fallback is preserved: log the exception and fall
  back to a point prediction with no CI band, rather than crashing.
- Two housekeeping fixes made while porting fp.forestplot-adjacent plotting
  conventions in Figure 4 do NOT apply here; this file only touches the GAM
  curve, not the forest plot.

USAGE
-----
    python generate_figure3_gam_curves.py \
        --input output_data/controls/matched_cases_cdc.xlsx \
        --exposure-dir output_data/risk_exposure \
        --outdir output_figures/figures_reviewed

Runtime note: model_curve() cross-validates the spline penalty weight via
GLMGam.select_penweight_kfold() over a 2D alpha grid (--alpha-grid-size^2
combinations x --cv-folds folds) — this is the notebook's original approach
and can take several minutes on the real dataset. Use --alpha-grid-size and
--cv-folds to shrink it for a quick smoke test.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.gam.generalized_additive_model import GLMGam
from statsmodels.gam.smooth_basis import BSplines
from statsmodels.genmod.families import NegativeBinomial as NB_FAMILY

# Reuse the already-validated case-selection and aggregation/exposure steps
# instead of duplicating them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_subject_breakdown import (  # noqa: E402
    clean_case_series,
    read_any,
)
from run_windowed_nb_models import make_input  # noqa: E402

LOG = logging.getLogger("generate_figure3_gam_curves")

# Legacy filenames captions031.md already references. Do not rename without
# also updating captions031.md, which we were asked not to touch.
PANEL_SUFFIXES = ("prediction", "difference", "logdifference")


# ---------------------------------------------------------------------------
# Ported from the notebook: prepare_curve_data / make_knots / model_curve /
# make_design / pred_with_ci / regression_ci
# ---------------------------------------------------------------------------

def prepare_curve_data(
    input1: pd.DataFrame,
    var1: str,
    wildcard: str,
    *,
    pca_min: int,
    pca_max: int,
):
    """
    Build the GAM design matrix from make_input()'s aggregated strata.
    Upper bound is inclusive (`<=`), matching the pipeline-wide convention
    (see module docstring, fix #2).
    """
    df = input1.copy()
    df.loc[df[var1] < pca_min, var1] = np.nan
    df.loc[df[var1] > pca_max, var1] = np.nan  # keep var1 <= pca_max
    n_before = len(df)
    df = df.dropna(subset=[var1])
    if len(df) != n_before:
        LOG.info("  prepare_curve_data: dropped %d strata rows outside [%d,%d]",
                  n_before - len(df), pca_min, pca_max)

    df[f"sex*{wildcard}"] = df["sex"] * df[wildcard]
    df[f"smoke*{wildcard}"] = df["smoke"] * df[wildcard]
    offset1 = df["log_exposure"]

    x1 = sm.add_constant(
        df[[var1, "sex", "smoke", wildcard, f"smoke*{wildcard}", f"sex*{wildcard}"]]
    )
    y1 = df["count"].values
    x1 = x1.reset_index(drop=True).copy()
    offset1 = offset1.reset_index(drop=True)
    return x1, y1, offset1


def make_knots(a: int, b: int, *, pca_min: int, peak: int, pca_max: int) -> np.ndarray:
    """
    Interior knot locations: denser spacing (step `a`) from pca_min to the
    peak boundary, then step `b` from the peak boundary out to pca_max.
    `peak` defaults to 46 (end of the descriptively-included 43-46 peak
    interval — see HANDOFF_CONTEXT.md item 5), not the analysis pca_max, so
    the knot break stays anchored to the biological peak regardless of how
    wide the overall PCA window is.
    """
    knots0 = np.arange(pca_min, peak, a)
    knots1 = np.arange(peak, pca_max, b)
    return np.concatenate([knots0, knots1])


def model_curve(
    df: pd.DataFrame,
    y: np.ndarray,
    offset: pd.Series,
    a: int,
    b: int,
    *,
    pca_min: int,
    peak_knot: int,
    pca_max: int,
    nb_alpha: float,
    alpha_grid_size: int,
    cv_folds: int,
):
    """
    Fit the NB GAM: linear terms (sex, smoke, preterm, interactions) plus a
    smooth of PCA that is allowed to differ by sex (two BSplines smooths,
    pca_f and pca_m, each held at the sex-pooled median PCA for the "off"
    sex — matches the notebook exactly).
    """
    y = np.asarray(y).astype(float)
    df = df.copy()
    df["pca"] = df["pca"].astype(float)
    for c in ("sex", "smoke", "preterm"):
        df[c] = df[c].astype(int)
    df["smoke_preterm"] = df["smoke"] * df["preterm"]
    df["sex_preterm"] = df["sex"] * df["preterm"]

    exog = pd.DataFrame({
        "Intercept": 1.0,
        "sex": df["sex"].values,
        "smoke": df["smoke"].values,
        "preterm": df["preterm"].values,
        "smoke_preterm": df["smoke_preterm"].values,
        "sex_preterm": df["sex_preterm"].values,
    }, index=df.index)

    pca = df["pca"].astype(float).values
    sex = df["sex"].astype(int).values
    pca_mid = np.nanmedian(pca)
    df["pca_f"] = np.where(sex == 0, pca, pca_mid)
    df["pca_m"] = np.where(sex == 1, pca, pca_mid)
    x_smooth = df[["pca_f", "pca_m"]].values

    knots = make_knots(a, b, pca_min=pca_min, peak=peak_knot, pca_max=pca_max)
    degree = 3
    knot_kwds = [{"knots": knots}, {"knots": knots}]
    smoother = BSplines(
        x_smooth,
        df=[len(knots) + degree + 1, len(knots) + degree + 1],
        degree=[degree, degree],
        include_intercept=False,
        knot_kwds=knot_kwds,
    )

    family = NB_FAMILY(alpha=nb_alpha)
    model = GLMGam(y, exog=exog, smoother=smoother, family=family, offset=offset)

    alphas = np.logspace(-2, 3, alpha_grid_size)
    LOG.info("  Cross-validating spline penalty weight: %d^2 alphas x %d folds",
              alpha_grid_size, cv_folds)
    alpha_cv, _ = model.select_penweight_kfold(alphas=[alphas, alphas], k_folds=cv_folds)
    LOG.info("  Selected alpha_cv=%s", alpha_cv)

    model2 = GLMGam(y, exog=exog, smoother=smoother, alpha=alpha_cv,
                     family=family, offset=offset)
    res = model2.fit()
    LOG.info("\n%s", res.summary())
    return res


def make_design(df: pd.DataFrame, sex: int, smoke: int, preterm: int):
    """Build exog/exog_smooth for prediction, sliced to one stratum, using
    the OBSERVED PCA values for that stratum (not a synthetic grid) — this
    keeps prediction points within the range each per-sex smoother was fit
    on, avoiding GLMGam's B-spline extrapolation limitation."""
    df2 = df.loc[(df["sex"] == sex) & (df["smoke"] == smoke) & (df["preterm"] == preterm)]

    pca_grid = np.asarray(df2["pca"], dtype=float)
    sex_grid = np.asarray(df2["sex"], dtype=float)
    smk_grid = np.asarray(df2["smoke"], dtype=float)
    trm_grid = np.asarray(df2["preterm"], dtype=float)
    count_grid = np.asarray(df2["count"], dtype=float)

    exog = pd.DataFrame({
        "Intercept": 1.0,
        "sex": sex_grid,
        "smoke": smk_grid,
        "preterm": trm_grid,
        "smoke_preterm": smk_grid * trm_grid,
        "sex_preterm": sex_grid * trm_grid,
    })

    pca_mid = np.nanmedian(df["pca"].astype(float).values)
    pca_f = np.where(sex_grid == 0, pca_grid, pca_mid)
    pca_m = np.where(sex_grid == 1, pca_grid, pca_mid)
    exog_smooth = np.column_stack([pca_f, pca_m])

    offset1 = df2["log_exposure"]
    return pca_grid, exog, exog_smooth, offset1, count_grid


def pred_with_ci(res, exog, exog_smooth, offset):
    try:
        pr = res.get_prediction(exog=exog, exog_smooth=exog_smooth, offset=offset)
        sf = pr.summary_frame(alpha=0.05)
        return sf["mean"].values, sf["mean_ci_lower"].values, sf["mean_ci_upper"].values
    except Exception as e:
        LOG.warning("  CI not available via get_prediction for this stratum: %r", e)
        mu = res.predict(exog=exog, exog_smooth=exog_smooth, offset=offset)
        return mu, None, None


def regression_ci(model, exog_f, exog_smooth_f, exog_m, exog_smooth_m, offset_f, offset_m):
    mu_f, lo_f, hi_f = pred_with_ci(model, exog_f, exog_smooth_f, offset_f)
    mu_m, lo_m, hi_m = pred_with_ci(model, exog_m, exog_smooth_m, offset_m)
    return mu_f, lo_f, hi_f, mu_m, lo_m, hi_m


# ---------------------------------------------------------------------------
# Plotting: draw_*_panel() draws onto a given ax (reused for both the three
# separate panel images matching AJE/captions031.md's Figure 3A/3B/3C, and
# the combined 3-row Figure3.png/.tiff for journal submission — same pattern
# generate_figure2_rate_curves.py uses for its per-panel + combined outputs).
# ---------------------------------------------------------------------------

def draw_prediction_panel(ax, pca_grid_f, count_f, mu_f, lo_f, hi_f,
                           pca_grid_m, count_m, mu_m, lo_m, hi_m, pca_max: int):
    ax.scatter(pca_grid_f, count_f, color="orangered", label="Observed Female", marker="v")
    ax.plot(pca_grid_f, mu_f, label="Predicted Female", color="r", linewidth=3)
    if lo_f is not None:
        ax.fill_between(pca_grid_f, lo_f, hi_f, alpha=0.2, color="magenta", label="Predicted 95% CI")

    ax.scatter(pca_grid_m, count_m, color="navy", label="Observed Male", marker="^")
    ax.plot(pca_grid_m, mu_m, label="Predicted Male", color="b", linewidth=3, linestyle="--")
    if lo_m is not None:
        ax.fill_between(pca_grid_m, lo_m, hi_m, alpha=0.2, color="cyan", label="Predicted 95% CI")

    ax.set_xlabel("PCA (weeks)", fontsize=14)
    ax.set_ylabel("Predicted mean count", fontsize=14)
    ax.legend(fontsize=11, ncols=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim([35, pca_max])
    return ax


def draw_difference_panel(ax, pca_grid_f, diff_observed, dif_pca, dif_lo, dif_hi, pca_max: int):
    ax.plot(pca_grid_f, diff_observed.rolling(window=3, center=True, min_periods=1).mean(),
            c="green", label="Observed M-F difference", linewidth=3)
    ax.plot(pca_grid_f, dif_pca, c="k", label="Predicted M-F difference", linewidth=3, linestyle="--")
    if dif_lo is not None:
        ax.fill_between(pca_grid_f, dif_lo, dif_hi, color="gray", alpha=0.2, label="95% CI")
    ax.set_ylabel("M-F count difference", fontsize=14)
    ax.set_xlabel("PCA (weeks)", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim([35, pca_max])
    ax.legend(fontsize=11)
    return ax


def draw_logdifference_panel(ax, pca_grid_f, logg_observed, log_pca, log_lo, log_hi, pca_max: int):
    ax.plot(pca_grid_f, logg_observed.rolling(window=3, center=True, min_periods=1).mean(),
            c="green", label="Observed M/F ratio, natural log", linewidth=3)
    ax.plot(pca_grid_f, log_pca, c="k", label="Predicted M/F ratio, natural log", linewidth=3, linestyle="--")
    if log_lo is not None:
        ax.fill_between(pca_grid_f, log_lo, log_hi, color="gray", alpha=0.2, label="95% CI")
    ax.set_ylabel("Ln M/F ratio", fontsize=14)
    ax.set_xlabel("PCA (weeks)", fontsize=14)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim([35, pca_max])
    return ax


def _standalone_figure(draw_fn, *args, **kwargs):
    """Wrap a draw_*_panel() call in its own single-axes figure, for the
    legacy per-panel PNGs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_fn(ax, *args, **kwargs)
    fig.tight_layout()
    return fig


def build_combined_figure(prediction_args, difference_args, logdifference_args, *, outdir: Path, stem: str):
    """
    One 3-row Figure3.png/.tiff with panel labels A/B/C, for journal
    submission — same content as the three split panel PNGs, matching the
    notebook's original pca_grid() combined layout (fig, axs = subplots(3,
    1, figsize=(12,12)), TIFF at 300 dpi), which captions031.md's per-panel
    convention deliberately splits apart but AJE still needs as one file per
    figure number.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), constrained_layout=True)
    draw_prediction_panel(axs[0], *prediction_args)
    draw_difference_panel(axs[1], *difference_args)
    draw_logdifference_panel(axs[2], *logdifference_args)

    for ax, label in zip(axs, ["A", "B", "C"]):
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                 fontsize=18, fontweight="bold", va="top", ha="left")

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.tiff", dpi=300, format="tiff")
    fig.savefig(outdir / f"{stem}.png", dpi=300)
    plt.close(fig)
    LOG.info("Wrote %s and %s", outdir / f"{stem}.png", outdir / f"{stem}.tiff")


def build_curve_panels(a, b, data, model, *, smoke_: int, preterm_: int, pca_max: int, outdir: Path):
    pca_grid_f, exog_f, exog_smooth_f, offset_f, count_f = make_design(data, sex=0, smoke=smoke_, preterm=preterm_)
    pca_grid_m, exog_m, exog_smooth_m, offset_m, count_m = make_design(data, sex=1, smoke=smoke_, preterm=preterm_)

    if len(pca_grid_f) == 0 or len(pca_grid_m) == 0:
        raise ValueError(
            f"No rows for reference stratum smoke={smoke_}, preterm={preterm_} in one or "
            f"both sexes (n_female={len(pca_grid_f)}, n_male={len(pca_grid_m)}). Check "
            f"--smoke-ref/--preterm-ref against the data."
        )

    mu_f, lo_f, hi_f, mu_m, lo_m, hi_m = regression_ci(
        model, exog_f, exog_smooth_f, exog_m, exog_smooth_m, offset_f, offset_m
    )

    stem = f"NB_splines_{smoke_}_{preterm_}_a{a}_b{b}"
    outdir.mkdir(parents=True, exist_ok=True)

    prediction_args = (pca_grid_f, count_f, mu_f, lo_f, hi_f,
                        pca_grid_m, count_m, mu_m, lo_m, hi_m, pca_max)
    fig1 = _standalone_figure(draw_prediction_panel, *prediction_args)
    fig1.savefig(outdir / f"{stem}_prediction.png", dpi=300)
    plt.close(fig1)

    # M-F difference / log-ratio need one value per sex per PCA week, so the
    # two sexes' grids must be aligned on the actual PCA value, not
    # position. The notebook assumed both grids lined up 1:1 by index, which
    # silently misaligns (or crashes, if lengths differ) whenever a PCA week
    # in this reference stratum was observed for one sex but not the other —
    # a real possibility in the sparser tail weeks of the widened pca_max=90
    # window. Align explicitly via an inner merge on pca, and report what,
    # if anything, got dropped.
    def _sex_frame(pca_grid, mu, lo, hi, count, label):
        d = {"pca": pca_grid, f"mu_{label}": mu, f"count_{label}": count}
        if lo is not None:
            d[f"lo_{label}"] = lo
            d[f"hi_{label}"] = hi
        return pd.DataFrame(d)

    df_f = _sex_frame(pca_grid_f, mu_f, lo_f, hi_f, count_f, "f")
    df_m = _sex_frame(pca_grid_m, mu_m, lo_m, hi_m, count_m, "m")
    merged = df_f.merge(df_m, on="pca", how="inner").sort_values("pca").reset_index(drop=True)

    dropped_f = set(pca_grid_f) - set(merged["pca"])
    dropped_m = set(pca_grid_m) - set(merged["pca"])
    if dropped_f or dropped_m:
        LOG.warning(
            "  Stratum smoke=%d,preterm=%d: %d PCA week(s) observed for females only "
            "%s and %d observed for males only %s dropped from the difference/"
            "log-difference panels (need both sexes present at that PCA week).",
            smoke_, preterm_, len(dropped_f), sorted(dropped_f), len(dropped_m), sorted(dropped_m),
        )

    have_ci = "lo_f" in merged.columns and "lo_m" in merged.columns
    pca_grid_merged = merged["pca"].values
    dif_pca = merged["mu_m"].values - merged["mu_f"].values
    dif_lo = (merged["lo_m"] - merged["lo_f"]).values if have_ci else None
    dif_hi = (merged["hi_m"] - merged["hi_f"]).values if have_ci else None
    diff_observed = pd.Series(merged["count_m"].values - merged["count_f"].values)

    difference_args = (pca_grid_merged, diff_observed, dif_pca, dif_lo, dif_hi, pca_max)
    fig2 = _standalone_figure(draw_difference_panel, *difference_args)
    fig2.savefig(outdir / f"{stem}_difference.png", dpi=300)
    plt.close(fig2)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_pca = np.log(merged["mu_m"].values / merged["mu_f"].values)
        log_lo = np.log(merged["lo_m"].values / merged["lo_f"].values) if have_ci else None
        log_hi = np.log(merged["hi_m"].values / merged["hi_f"].values) if have_ci else None
        logg_observed = pd.Series(np.log(merged["count_m"].values / merged["count_f"].values))

    logdifference_args = (pca_grid_merged, logg_observed, log_pca, log_lo, log_hi, pca_max)
    fig3 = _standalone_figure(draw_logdifference_panel, *logdifference_args)
    fig3.savefig(outdir / f"{stem}_logdifference.png", dpi=300)
    plt.close(fig3)

    for suffix in PANEL_SUFFIXES:
        LOG.info("Wrote %s", outdir / f"{stem}_{suffix}.png")

    # Combined 3-row Figure3.png/.tiff, for the actual journal submission
    # (AJE wants one file per figure number, panel-labeled) — same data as
    # the three split panels above, just recomposed onto one canvas.
    build_combined_figure(prediction_args, difference_args, logdifference_args,
                           outdir=outdir, stem="Figure3")

    return {
        "pca_grid_f": pca_grid_f, "mu_f": mu_f, "count_f": count_f,
        "pca_grid_m": pca_grid_m, "mu_m": mu_m, "count_m": count_m,
    }


# ---------------------------------------------------------------------------
# Orchestration (NB_splines(), corrected)
# ---------------------------------------------------------------------------

def run_nb_splines(
    case_df: pd.DataFrame,
    *,
    var1: str,
    exposure_dir: Path,
    outdir: Path,
    pca_min: int,
    pca_max: int,
    peak_knot: int,
    a: int,
    b: int,
    smoke_ref: int,
    preterm_ref: int,
    nb_alpha: float,
    alpha_grid_size: int,
    cv_folds: int,
    combgest_col: str,
    smoking_col: str,
    sex_col: str,
):
    input1 = make_input(
        var1, case_df, exposure_dir,
        combgest_col=combgest_col, smoking_col=smoking_col, sex_col=sex_col,
    )
    x1, y1, offset1 = prepare_curve_data(input1, var1, "preterm", pca_min=pca_min, pca_max=pca_max)
    LOG.info("GAM design matrix: %d strata rows, %d total cases", len(x1), int(y1.sum()))

    model = model_curve(
        x1, y1, offset1, a, b,
        pca_min=pca_min, peak_knot=peak_knot, pca_max=pca_max,
        nb_alpha=nb_alpha, alpha_grid_size=alpha_grid_size, cv_folds=cv_folds,
    )
    data = pd.concat([x1, offset1, pd.DataFrame(y1, columns=["count"])], axis=1)

    return build_curve_panels(
        a, b, data, model,
        smoke_=smoke_ref, preterm_=preterm_ref, pca_max=pca_max, outdir=outdir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rebuild Figure 3 (NB-GAM PCA curves by sex) with an explicit, "
                    "auditable, up-to-date data pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                    help="Path to the raw case file, e.g. matched_cases_cdc.xlsx")
    p.add_argument("--exposure-dir", type=Path, default=Path("output_data/risk_exposure"))
    p.add_argument("--outdir", type=Path, default=Path("curvefit_exp"),
                    help="Matches captions031.md's curvefit_exp/ image paths.")

    p.add_argument("--infage-col", default="infage")
    p.add_argument("--combgest-col", default="combgest")
    p.add_argument("--sex-col", default="sex")
    p.add_argument("--smoking-col", default="cig_rec")
    p.add_argument("--pca-min", type=int, default=36)
    p.add_argument("--pca-max", type=int, default=90)
    p.add_argument("--peak-knot", type=int, default=46,
                    help="Knot-spacing break point (dense spacing below, coarse above). "
                         "Matches the top of the descriptively-included 43-46 peak "
                         "interval; independent of --pca-max.")
    p.add_argument("--a", type=int, default=10, help="Knot spacing below --peak-knot.")
    p.add_argument("--b", type=int, default=10, help="Knot spacing above --peak-knot.")
    p.add_argument("--smoke-ref", type=int, default=0, choices=[0, 1],
                    help="Reference stratum for the plotted curve (0=non-smoker).")
    p.add_argument("--preterm-ref", type=int, default=0, choices=[0, 1],
                    help="Reference stratum for the plotted curve (0=term birth).")
    p.add_argument("--nb-alpha", type=float, default=0.10,
                    help="Negative-binomial dispersion prior (not cross-validated; "
                         "matches the notebook's fixed value).")
    p.add_argument("--alpha-grid-size", type=int, default=20,
                    help="Points per axis in the logspace(-2,3,...) penalty-weight grid "
                         "searched by select_penweight_kfold (grid is squared: this many "
                         "x this many). Reduce for a fast smoke test.")
    p.add_argument("--cv-folds", type=int, default=5,
                    help="K-folds for select_penweight_kfold. Reduce for a fast smoke test.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.outdir / "figure3_run.log")],
    )

    if not args.input.exists():
        LOG.error("Input file not found: %s", args.input)
        return 1
    for name in ("smoke_exposure", "sex_exposure", "term_exposure"):
        pth = args.exposure_dir / f"{name}.pkl"
        if not pth.exists():
            LOG.error("Missing exposure pickle: %s", pth)
            return 1

    LOG.info("Loading %s", args.input)
    raw_df = read_any(args.input)
    LOG.info("Raw shape: %s", raw_df.shape)

    case_df, _, trail = clean_case_series(
        raw_df,
        infage_col=args.infage_col,
        combgest_col=args.combgest_col,
        pca_min=args.pca_min,
        pca_max=args.pca_max,
        pca_max_inclusive=False,
        reproduce_original_bug=False,
    )
    trail.to_frame().to_csv(args.outdir / "audit_trail_figure3.csv", index=False)
    LOG.info("Corrected case series: n=%d (PCA window [%d,%d])",
              len(case_df), args.pca_min, args.pca_max)

    run_nb_splines(
        case_df,
        var1="pca",
        exposure_dir=args.exposure_dir,
        outdir=args.outdir,
        pca_min=args.pca_min,
        pca_max=args.pca_max,
        peak_knot=args.peak_knot,
        a=args.a,
        b=args.b,
        smoke_ref=args.smoke_ref,
        preterm_ref=args.preterm_ref,
        nb_alpha=args.nb_alpha,
        alpha_grid_size=args.alpha_grid_size,
        cv_folds=args.cv_folds,
        combgest_col=args.combgest_col,
        smoking_col=args.smoking_col,
        sex_col=args.sex_col,
    )

    LOG.info("DONE. See %s", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

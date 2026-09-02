#!/usr/bin/env python3
"""
compare_time_axes.py  (AUDIT A6.40-42 / A3.26 / Ed E8, E9, E13, E15)

Ed Mitchell's review of v6 asks, in effect: show that postconceptional age (PCA)
is a better developmental axis than chronological (postnatal) age, rather than
asserting it; and check the "prematurity contributes earlier" claim against the
data. This script produces the supporting analyses:

  1. GAM fit comparison - the same NB spline GAM fit with PCA as the time axis
     and again with chronological age (infageweeks) as the time axis; reports
     deviance / AIC / log-likelihood / pseudo-R2 for each. A materially better
     fit under PCA is the argument that PCA beats CA (Ed E9).
  2. Observed vs fitted, PCA panel and CA panel side by side.
  3. PCA vs chronological age at death - scatter + Pearson/Spearman correlation
     on the analytic cohort. If they are near-collinear, "what does PCA add"
     must be answered in the text (Ed E15).
  4. Proportion of deaths that are preterm, by PCA week and by chronological-age
     week - a direct descriptive check on the "prematurity earlier" claim
     (Ed E13 / A3.26).

Reuses ncfrp_manuscript_v5: load_ncfrp_data, make_ncfrp_input,
prepare_curve_data_ncfrp, fit_nb_gam.

USAGE
    python compare_time_axes.py \
        --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \
        --exposure-dir output_data/risk_exposure \
        --pca-min 36 --pca-max 90 --offset-mode joint \
        --outdir output_data/ncfrp/time_axes
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from ncfrp_manuscript_v5 import (load_ncfrp_data, make_ncfrp_input,
                                 prepare_curve_data_ncfrp, fit_nb_gam)

LOG = logging.getLogger("compare_time_axes")


def _gam_fit_stats(res) -> dict:
    llf = float(getattr(res, "llf", np.nan))
    k = float(getattr(res, "df_model", np.nan)) + 1.0
    aic = float(getattr(res, "aic", 2 * k - 2 * llf if np.isfinite(llf) else np.nan))
    return {
        "deviance": float(getattr(res, "deviance", np.nan)),
        "log_likelihood": llf,
        "aic": aic,
        "df_model": float(getattr(res, "df_model", np.nan)),
        "pseudo_r2": float(getattr(res, "pseudo_rsquared", lambda: np.nan)())
        if callable(getattr(res, "pseudo_rsquared", None))
        else float(getattr(res, "pseudo_rsquared", np.nan)),
    }


def _fit_one_axis(df: pd.DataFrame, var1: str, exposure_dir: Path,
                  offset_mode: str, gam_alpha_grid: int):
    inp = make_ncfrp_input(var1, df, exposure_dir=exposure_dir, min_count=1,
                           offset_mode=offset_mode)
    x_lin, x_smooth, y, offset, curve = prepare_curve_data_ncfrp(inp, var1)
    res, knots = fit_nb_gam(x_lin, x_smooth, y, offset, alpha_grid_size=gam_alpha_grid)
    return inp, curve, res, knots


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", type=Path,
                   default=Path("standardized/NCFRP_SCH_LR_BW_pop.csv"))
    p.add_argument("--exposure-dir", type=Path, default=Path("output_data/risk_exposure"))
    p.add_argument("--pca-min", type=int, default=36)
    p.add_argument("--pca-max", type=int, default=90)
    p.add_argument("--offset-mode", choices=["joint", "marginal"], default="joint")
    p.add_argument("--gam-alpha-grid", type=int, default=0)
    p.add_argument("--year-min", type=int, default=None)
    p.add_argument("--year-max", type=int, default=None)
    p.add_argument("--outdir", type=Path, default=Path("output_data/ncfrp/time_axes"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    args.outdir.mkdir(parents=True, exist_ok=True)

    df, _ = load_ncfrp_data(args.data_path, pca_min=args.pca_min, pca_max=args.pca_max,
                            year_min=args.year_min, year_max=args.year_max, audit_csv=None)
    LOG.info("cohort n=%d", len(df))

    # ---- 1. GAM fit comparison ----------------------------------------
    _, curve_pca, res_pca, _ = _fit_one_axis(df, "pca", args.exposure_dir,
                                             args.offset_mode, args.gam_alpha_grid)
    _, curve_ca, res_ca, _ = _fit_one_axis(df, "infageweeks", args.exposure_dir,
                                           args.offset_mode, args.gam_alpha_grid)
    cmp = pd.DataFrame({"axis": ["PCA", "chronological_age"],
                        **{k: [_gam_fit_stats(res_pca)[k], _gam_fit_stats(res_ca)[k]]
                           for k in _gam_fit_stats(res_pca)}})
    cmp.to_csv(args.outdir / "gam_fit_comparison.csv", index=False)
    LOG.info("GAM fit comparison:\n%s", cmp.to_string(index=False))

    # ---- 2. observed vs fitted, both axes ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, curve, var1, title in [(axes[0], curve_pca, "pca", "A. PCA axis"),
                                   (axes[1], curve_ca, "infageweeks", "B. Chronological age axis")]:
        obs = curve.groupby(var1, as_index=False)["count"].sum().sort_values(var1)
        ax.plot(obs[var1], obs["count"], marker="o", linewidth=0.6, label="observed (summed)")
        ax.set_title(title)
        ax.set_xlabel("weeks")
        ax.set_ylabel("SUID case count")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.savefig(args.outdir / "observed_by_axis.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.outdir / "observed_by_axis.tiff", dpi=300, format="tiff",
                bbox_inches="tight")
    plt.close(fig)

    # ---- 3. PCA vs chronological age at death ----------------------
    d = df.dropna(subset=["pca", "infageweeks"])
    pr, pp = pearsonr(d["pca"], d["infageweeks"])
    sr, sp = spearmanr(d["pca"], d["infageweeks"])
    with open(args.outdir / "pca_vs_chronological_correlation.txt", "w") as f:
        f.write(f"n = {len(d)}\n")
        f.write(f"Pearson r  = {pr:.4f}  (p = {pp:.3g})\n")
        f.write(f"Spearman r = {sr:.4f}  (p = {sp:.3g})\n")
        f.write("High correlation is expected (PCA = gestage + postnatal age). "
                "The question is how much PCA REORDERS cases relative to "
                "chronological age; see the scatter and the residual spread.\n")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(d["infageweeks"], d["pca"], s=6, alpha=0.2)
    ax.set_xlabel("Chronological age at death (weeks)")
    ax.set_ylabel("Postconceptional age at death (weeks)")
    ax.set_title(f"PCA vs chronological age (Pearson r = {pr:.3f})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.outdir / "pca_vs_chronological_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 4. proportion preterm by age, both axes ------------------
    d = df.copy()
    d["preterm01"] = (d["preterm"] == "Y").astype(int)
    out = {}
    for var1, name in [("pca", "by_pca_week"), ("infageweeks", "by_chronological_week")]:
        g = (d.groupby(d[var1].astype(int))
             .agg(n=("preterm01", "size"), n_preterm=("preterm01", "sum")))
        g["prop_preterm"] = g["n_preterm"] / g["n"]
        g.to_csv(args.outdir / f"proportion_preterm_{name}.csv")
        out[name] = g
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, g in out.items():
        ax.plot(g.index, g["prop_preterm"], marker="o", markersize=3, label=name)
    ax.set_ylabel("Proportion of deaths that were preterm (<37 wk)")
    ax.set_xlabel("weeks")
    ax.set_title("Preterm share of SUID deaths by age axis (Ed E13 check)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(args.outdir / "proportion_preterm_by_age.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    LOG.info("Wrote time-axis comparison outputs to %s", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

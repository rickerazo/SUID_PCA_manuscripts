#!/usr/bin/env python3
"""
generate_figure4_forest_plots.py

Standalone, corrected port of the notebook's Figure 4 pipeline: forest plots
of the confirmatory windowed negative-binomial regression IRRs
(`extract_meaning_from_model()` in
SIDS_Collaboration_manuscript_pipeline_v4.ipynb), rebuilt on top of the
already-validated model fitting in run_windowed_nb_models.py.

By default this does NOT refit anything — it reads the IRR tables
run_windowed_nb_models.py already wrote (nb_models/corrected/irr_early_
window_corrected.csv, irr_late_window_corrected.csv), the same numbers
Supplemental Table 1 and Supplemental Table 4 report, and only builds the
plot. This guarantees Figure 4 and the supplemental tables can never quietly
disagree because one was regenerated and the other wasn't. Pass --refit-from
to fit fresh in one step instead (e.g. for a clean end-to-end run).

WHAT THIS FIXES / CHANGES vs. the notebook
---------------------------------------------
1. Feeds the model the CORRECTED case series and exposure pickles (via
   run_windowed_nb_models.py, already validated — see HANDOFF_CONTEXT.md).

2. The notebook's `xlineval=2` argument passed to fp.forestplot() is not a
   parameter this version of the forestplot library (0.4.1) recognizes —
   it's silently swallowed by **kwargs and does nothing (confirmed by
   inspecting the installed package and by a smoke test: the rendered plot
   is byte-identical with or without it). The actual null-effect reference
   line comes from the separate `axes.axvline(x=1, ...)` call right after,
   which IS real and is kept. Dropped the dead `xlineval=2` rather than
   silently carry forward a parameter that never did anything.

3. Likewise, the notebook passed `"title": "..."` inside the kwargs dict
   forwarded to fp.forestplot(); this library does not set an axes title
   from that kwarg either (confirmed the same way — ax.get_title() stays
   empty). This version sets the title explicitly via ax.set_title() after
   the call, so titles the manuscript's figure legend leans on for
   distinguishing panel A/B actually render (the notebook's combined
   Figure4.png didn't need this because the two panels were adjacent with
   external "A"/"B" labels; captions031.md now treats them as two fully
   separate images, so an in-image cue matters more).

4. captions031.md expects two SEPARATE images (Figure 4A early window, 4B
   late window) under specific legacy filenames (data/review_013/
   manuscript/forest_plot_m1.png, forest_plot_m2.png) — this script writes
   those two files. It ALSO writes a combined 2-row Figure4.png/.tiff (300
   dpi TIFF) with panel labels A/B, matching the notebook's original
   run_binomial_pipeline() combined layout, since AJE's actual submission
   wants one file per figure number regardless of how the response-letter
   captions split it. Both are built from the same draw_forest_plot() calls,
   so they can't drift apart from each other.

5. Predictor labels and the Intrinsic/Extrinsic/Perinatal/Interactions
   grouping are unchanged from the notebook, reusing
   run_windowed_nb_models.PREDICTOR_LABELS and generate_supplemental_
   table4.py's PREDICTOR_ORDER so the row set/order can't drift between
   this figure, Supplemental Table 1, and Supplemental Table 4.

USAGE
-----
    # Default: plot the already-fit, already-validated corrected models.
    python generate_figure4_forest_plots.py \
        --irr-early output_data/nb_models/corrected/irr_early_window_corrected.csv \
        --irr-late  output_data/nb_models/corrected/irr_late_window_corrected.csv \
        --outdir output_figures/figures_reviewed

    # One-shot: fit fresh from the raw case file, then plot.
    python generate_figure4_forest_plots.py \
        --refit-from output_data/controls/matched_cases_cdc.xlsx \
        --exposure-dir output_data/risk_exposure \
        --outdir data/review_013/manuscript
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
import forestplot as fp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_subject_breakdown import clean_case_series, read_any  # noqa: E402
from run_windowed_nb_models import fit_windowed_models, irr_table  # noqa: E402

LOG = logging.getLogger("generate_figure4_forest_plots")

# Same 6 coefficient rows extract_meaning_from_model() plotted, in the same
# order prepare_data() builds the design matrix (const, pca, sex, smoke,
# preterm, smoke*preterm, sex*preterm) minus const/alpha. Kept in sync with
# generate_supplemental_table4.PREDICTOR_ORDER (imported, not duplicated) so
# the figure and the supplemental table can't silently diverge in row set.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_supplemental_table4 import PREDICTOR_ORDER  # noqa: E402

GROUP_LABEL = {
    "PCA (per week)": "Intrinsic Factors",
    "Male (vs female)": "Intrinsic Factors",
    "Maternal smoking": "Extrinsic Factors",
    "Preterm birth": "Perinatal Factors",
    "Smoking x Preterm": "Interactions",
    "Sex x Preterm": "Interactions",
}
GROUP_ORDER = ["Intrinsic Factors", "Extrinsic Factors", "Perinatal Factors", "Interactions"]
DISPLAY_LABEL = {
    "Smoking x Preterm": "Smoking × Preterm",
    "Sex x Preterm": "Sex × Preterm",
}


# ---------------------------------------------------------------------------
# irr_table() (from run_windowed_nb_models.py) -> forestplot-ready DataFrame
# ---------------------------------------------------------------------------

def to_forest_df(irr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict an irr_table()-shaped DataFrame (Predictor, IRR, CI_low,
    CI_high, p, plus trailing model-fit-statistic rows) down to the 6
    coefficient rows the forest plot shows, in PREDICTOR_ORDER, with group
    labels attached.
    """
    idx = irr_df.set_index("Predictor")
    missing = [p for p in PREDICTOR_ORDER if p not in idx.index]
    if missing:
        raise KeyError(f"irr table is missing expected predictor rows: {missing}")

    rows = []
    for name in PREDICTOR_ORDER:
        r = idx.loc[name]
        rows.append({
            "variable": DISPLAY_LABEL.get(name, name),
            "rate_ratio": r["IRR"],
            "rr_ci_low": r["CI_low"],
            "rr_ci_high": r["CI_high"],
            "pvalue": r["p"],
            "group": GROUP_LABEL[name],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ported from the notebook: extract_meaning_from_model(), generalized to take
# an already-restricted/labeled DataFrame instead of a live model object, so
# it works identically whether the IRRs came from a fresh fit or a saved CSV.
# ---------------------------------------------------------------------------

def draw_forest_plot(df: pd.DataFrame, xmin: float, title: str, ax):
    fp.forestplot(
        dataframe=df,
        estimate="rate_ratio",
        ll="rr_ci_low",
        hl="rr_ci_high",
        varlabel="variable",
        groupvar="group",
        group_order=GROUP_ORDER,
        color_alt_rows=True,
        pval="pvalue",
        ax=ax,
        ylabel="Variables",
        xlabel="Incidence Rate Ratio",
    )
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlim([xmin, None])
    ax.set_title(title, fontsize=14, pad=12)  # forestplot doesn't honor a title kwarg (see #3 above)
    return ax


def build_forest_figure(df: pd.DataFrame, xmin: float, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(left=0.4, right=0.97, top=0.90, bottom=0.1)
    draw_forest_plot(df, xmin, title, ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # forestplot's row/CI-report text is drawn in axes-relative coordinates
    # that can extend left of x=0 of the figure (long labels like "Maternal
    # smoking" or "Sex x Preterm" plus their IRR(CI) text). A fixed
    # subplots_adjust margin isn't enough to guarantee no clipping across
    # every predictor-label length, so save with bbox_inches="tight" to let
    # matplotlib measure the actual text extents and pad accordingly —
    # confirmed necessary by a smoke test where "left" margin alone clipped
    # "Maternal smoking" down to "ng".
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def build_combined_forest_figure(df_early: pd.DataFrame, df_late: pd.DataFrame,
                                  xmin_early: float, xmin_late: float, outdir: Path) -> None:
    """
    One 2-row Figure4.png/.tiff (300 dpi TIFF) with panel labels A/B, for
    journal submission — same two panels as forest_plot_m1.png/forest_plot_
    m2.png, recomposed onto one canvas, matching the notebook's original
    run_binomial_pipeline() combined layout (fig, axs = subplots(2, 1,
    figsize=(10, 14)), "A"/"B" placed via fig.text at each axes' figure-
    fraction top, in the blank margin left of the row-label column rather
    than inside the axes).
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 14))
    fig.subplots_adjust(left=0.4, right=0.97, top=0.95, bottom=0.05, hspace=0.3)
    draw_forest_plot(df_early, xmin_early, "Early Window (<43 post-conceptional weeks)", axs[0])
    draw_forest_plot(df_late, xmin_late, "Late Window (≥47 post-conceptional weeks)", axs[1])

    # y nudged above bbox.y1 (not exactly on it) so the bold panel letter
    # clears forestplot's "Intrinsic Factors" group header text underneath
    # instead of sitting flush against it.
    x_left = 0.01
    for ax, label in zip(axs, ["A", "B"]):
        bbox = ax.get_position()  # figure coords
        fig.text(x_left, bbox.y1 + 0.015, label, fontsize=18, fontweight="bold", va="top", ha="left")

    outdir.mkdir(parents=True, exist_ok=True)
    # Same bbox_inches="tight" rationale as build_forest_figure() above —
    # forestplot's row-label text can extend past a fixed margin.
    fig.savefig(outdir / "Figure4.tiff", dpi=300, format="tiff", bbox_inches="tight")
    fig.savefig(outdir / "Figure4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote %s and %s", outdir / "Figure4.png", outdir / "Figure4.tiff")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rebuild Figure 4 (forest plots of windowed NB regression IRRs).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--irr-early", type=Path,
                        help="Path to an existing irr_early_window_*.csv (from "
                             "run_windowed_nb_models.py). Use with --irr-late.")
    mode.add_argument("--refit-from", type=Path,
                        help="Raw case file (e.g. matched_cases_cdc.xlsx) to fit fresh "
                             "instead of reading saved IRR tables.")
    p.add_argument("--irr-late", type=Path,
                    help="Path to an existing irr_late_window_*.csv. Required with --irr-early.")
    p.add_argument("--outdir", type=Path, default=Path("data/review_013/manuscript"),
                    help="Matches captions031.md's data/review_013/manuscript/ image paths.")

    p.add_argument("--xmin-early", type=float, default=0.4,
                    help="X-axis lower bound for the early-window panel (matches the "
                         "notebook's run_binomial_pipeline() call).")
    p.add_argument("--xmin-late", type=float, default=0.65,
                    help="X-axis lower bound for the late-window panel.")

    # --refit-from options
    p.add_argument("--exposure-dir", type=Path, default=Path("output_data/risk_exposure"))
    p.add_argument("--infage-col", default="infage")
    p.add_argument("--combgest-col", default="combgest")
    p.add_argument("--sex-col", default="sex")
    p.add_argument("--smoking-col", default="cig_rec")
    p.add_argument("--pca-min", type=int, default=36)
    p.add_argument("--pca-max", type=int, default=90)
    p.add_argument("--peak1", type=int, default=43)
    p.add_argument("--peak2", type=int, default=47)
    p.add_argument("--peak3", type=int, default=None,
                    help="Defaults to --pca-max (see run_windowed_nb_models.py).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.irr_early and not args.irr_late:
        LOG.error("--irr-early requires --irr-late")
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.outdir / "figure4_run.log")],
    )

    if args.refit_from:
        if not args.refit_from.exists():
            LOG.error("Input file not found: %s", args.refit_from)
            return 1
        peak3 = args.peak3 if args.peak3 is not None else args.pca_max
        LOG.info("Refitting from %s (peak3=%d)", args.refit_from, peak3)
        raw_df = read_any(args.refit_from)
        case_df, _, trail = clean_case_series(
            raw_df,
            infage_col=args.infage_col, combgest_col=args.combgest_col,
            pca_min=args.pca_min, pca_max=args.pca_max,
            pca_max_inclusive=False, reproduce_original_bug=False,
        )
        trail.to_frame().to_csv(args.outdir / "audit_trail_figure4.csv", index=False)

        _, x1, x2, y1, y2, model1, model2 = fit_windowed_models(
            case_df, peak1=args.peak1, peak2=args.peak2, peak3=peak3,
            exposure_dir=args.exposure_dir,
            combgest_col=args.combgest_col, smoking_col=args.smoking_col, sex_col=args.sex_col,
        )
        irr_early = irr_table(model1, "pca", int(y1.sum()))
        irr_late = irr_table(model2, "pca", int(y2.sum()))
        irr_early.to_csv(args.outdir / "irr_early_window_figure4.csv", index=False)
        irr_late.to_csv(args.outdir / "irr_late_window_figure4.csv", index=False)
    else:
        for pth in (args.irr_early, args.irr_late):
            if not pth.exists():
                LOG.error("IRR table not found: %s", pth)
                return 1
        LOG.info("Reading pre-fit IRR tables: %s, %s", args.irr_early, args.irr_late)
        irr_early = pd.read_csv(args.irr_early)
        irr_late = pd.read_csv(args.irr_late)

    df_early = to_forest_df(irr_early)
    df_late = to_forest_df(irr_late)

    build_forest_figure(
        df_early, args.xmin_early, "Early Window (<43 post-conceptional weeks)",
        args.outdir / "forest_plot_m1.png",
    )
    build_forest_figure(
        df_late, args.xmin_late, "Late Window (≥47 post-conceptional weeks)",
        args.outdir / "forest_plot_m2.png",
    )

    # Combined 2-row Figure4.png/.tiff, for the actual journal submission
    # (AJE wants one file per figure number, panel-labeled) — same data as
    # the two split panels above.
    build_combined_forest_figure(df_early, df_late, args.xmin_early, args.xmin_late, args.outdir)

    LOG.info("DONE. See %s", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

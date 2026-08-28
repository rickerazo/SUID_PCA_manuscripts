#!/usr/bin/env python3
"""
generate_figure2_rate_curves.py

Standalone, corrected port of the notebook's Figure 2 pipeline (SUID rates
by postnatal age and postconceptional age): normalized_rates() ->
combine_sexes() -> visualize_death_rates() -> death_rates().

WHAT THIS FIXES / SURFACES vs. the notebook
---------------------------------------------
1. The code that produces Figure 2 was commented out in main() until this
   session, disconnected from every other fix made in this pipeline audit
   (first-week exclusion, PCA window widening, cig_rec validity, exposure
   computation). This script rebuilds it independently so it can be
   validated on its own terms rather than assumed correct.

2. normalized_rates(var1, peak3=75) never had its peak3 default updated
   when the rest of the analysis moved from 75 to 90. Here, --pca-max
   defaults to 90 and there is no silent fallback.

3. normalized_rates() reads its numerator directly from R95_num.parquet /
   W75_num.parquet / R99_num.parquet (produced by
   consolidate_icd_datasets.py), NOT from matched_cases_cdc.xlsx, which is
   what load_case_control() uses for Table 1, Figure 3, and Figure 4. These
   may or may not be the same underlying case population. This script
   supports BOTH sources (--numerator-source) and, with --compare-sources,
   reports whether they actually agree in total N (figure2_source_
   comparison.txt) instead of silently assuming they match.

4. normalized_rates() additionally drops combgest>44 (a truncation not
   present anywhere else in the pipeline) and never filters for a valid
   cig_rec (unlike count_data(), used for Table 1/Figure 3/Figure 4). Both
   are now explicit, documented, and OFF by default here (i.e. the default
   behavior matches the rest of the corrected pipeline, not the original
   Figure 2 code) — pass --max-combgest 44 and --no-require-valid-smoking
   to reproduce the original quirks for comparison.

5. normalized_rates() never explicitly drops the infage>=99 "unknown"
   sentinel; it happened to get swept up incidentally by the pca>peak3
   truncation. This script drops it explicitly, matching load_case_
   control()'s step 2b, so it doesn't depend on peak3 being small enough
   to catch it by accident.

USAGE
-----
    # Default: consistent with the rest of the corrected pipeline
    # (matched_cases_cdc.xlsx, pca_max=90, valid cig_rec required, no
    # extra combgest truncation).
    python generate_figure2_rate_curves.py \
        --denominator standardized/consolidated/denominator.parquet \
        --matched-cases-xlsx output_data/controls/matched_cases_cdc.xlsx \
        --outdir output_figures/manuscript

    # Also load the original per-ICD-code source under the original
    # (uncorrected) cleaning rules, and report whether it agrees with the
    # matched_cases_cdc.xlsx-based population in total N:
    python generate_figure2_rate_curves.py \
        --denominator standardized/consolidated/denominator.parquet \
        --matched-cases-xlsx output_data/controls/matched_cases_cdc.xlsx \
        --r95 standardized/consolidated/R95_num.parquet \
        --w75 standardized/consolidated/W75_num.parquet \
        --r99 standardized/consolidated/R99_num.parquet \
        --outdir output_figures/manuscript_reviewd \
        --compare-sources
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

LOG = logging.getLogger("generate_figure2_rate_curves")


# ---------------------------------------------------------------------------
# Numerator loaders
# ---------------------------------------------------------------------------

def load_numerator_matched_cases(
    path: Path,
    *,
    infage_col: str = "infage",
    combgest_col: str = "combgest",
    sex_col: str = "sex",
) -> pd.DataFrame:
    """Same source as load_case_control() / Table 1 / Figure 3 / Figure 4."""
    df = pd.read_excel(path)
    for col in (infage_col, combgest_col, sex_col):
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in {path}. "
                            f"Available: {list(df.columns)}")
    return df


def load_numerator_per_icd(r95: Path, w75: Path, r99: Path) -> pd.DataFrame:
    """Original notebook source: the three per-ICD-code numerator parquets."""
    parts = []
    for label, path in (("R95", r95), ("W75", w75), ("R99", r99)):
        if not path.exists():
            raise FileNotFoundError(f"{label} numerator file not found: {path}")
        parts.append(pd.read_parquet(path))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Cleaning + PCA computation (ported from normalized_rates(), corrected)
# ---------------------------------------------------------------------------

def clean_and_compute_pca(
    df: pd.DataFrame,
    *,
    infage_col: str = "infage",
    combgest_col: str = "combgest",
    smoking_col: str = "cig_rec",
    pca_min: int = 36,
    pca_max: int = 90,
    max_combgest: float | None = None,
    require_valid_smoking: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    n0 = len(df)

    if max_combgest is not None:
        df.loc[df[combgest_col] > max_combgest, combgest_col] = np.nan
    else:
        # Match load_case_control() step 2: drop only the unknown sentinel.
        df.loc[df[combgest_col] >= 99, combgest_col] = np.nan

    df.loc[df[infage_col] <= 1, infage_col] = np.nan          # first-week exclusion
    df.loc[df[infage_col] >= 99, infage_col] = np.nan         # unknown infant age
    df.dropna(subset=[combgest_col, infage_col], inplace=True)

    df["pca"] = df[combgest_col] + df[infage_col]
    df.loc[df["pca"] < pca_min, "pca"] = np.nan
    df.loc[df["pca"] > pca_max, "pca"] = np.nan
    df.dropna(subset=["pca"], inplace=True)

    if require_valid_smoking:
        df = df.loc[df[smoking_col].isin(["Y", "N"])].copy()

    LOG.info("  cleaned: n=%d -> n=%d (max_combgest=%s, require_valid_smoking=%s, "
              "pca window [%d,%d])",
              n0, len(df), max_combgest, require_valid_smoking, pca_min, pca_max)
    return df


# ---------------------------------------------------------------------------
# Rate computation + plotting (ported from normalized_rates() / combine_sexes()
# / visualize_death_rates() / death_rates())
# ---------------------------------------------------------------------------

def compute_sex_specific_series(df: pd.DataFrame, var1: str, sex_col: str = "sex"):
    males = df.loc[df[sex_col] == "M"]
    females = df.loc[df[sex_col] == "F"]
    x1 = males[var1].value_counts().sort_index().index.to_numpy()
    y1 = males[var1].value_counts().sort_index().values
    x0 = females[var1].value_counts().sort_index().index.to_numpy()
    y0 = females[var1].value_counts().sort_index().values
    return x0, y0, x1, y1


def combine_sexes(x0, x1, y0, y1):
    xmin = min(x0.min(), x1.min())
    xmax = max(x0.max(), x1.max())
    x = np.arange(xmin, xmax + 1)
    yout = np.zeros_like(x, dtype=float)
    yout[(x0 - xmin).astype(int)] += y0
    yout[(x1 - xmin).astype(int)] += y1
    return x, yout


def visualize_death_rates(x0, y0, x1, y1, boys, gals, colors, factor, var1, ax, pca_max: int):
    x, y = combine_sexes(x0, x1, y0, y1)
    ax.plot(x, y / (boys + gals) * factor, color="k", linewidth=5, alpha=0.7, label="Total")
    ax.plot(x0, y0 / gals * factor, color=colors[0], linewidth=3, alpha=0.9, label="F", linestyle="--")
    ax.plot(x1, y1 / boys * factor, color=colors[1], linewidth=3, alpha=0.9, label="M", linestyle=":")

    ax.set_ylabel("Death Rate per 100,000 Live Births", fontsize=14)
    if var1 == "infage":
        ax.set_xlabel("Postnatal age (weeks)", fontsize=14)
        ax.set_xticks(np.arange(2, 66, 6))
    if var1 == "pca":
        ax.set_xlabel("Post-conception age (weeks)", fontsize=14)
        ax.set_xticks(np.arange(36, pca_max + 4, 6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(markerscale=2, fontsize=14)

    peak_f_age = x0[np.argmax(y0)]
    peak_m_age = x1[np.argmax(y1)]
    LOG.info("  %s: peak female rate at %s=%s (n=%d) | peak male rate at %s=%s (n=%d)",
              var1, var1, peak_f_age, int(np.max(y0)), var1, peak_m_age, int(np.max(y1)))
    return ax, x, y


# Legacy per-panel filenames, matching what AJE/captions031.md already
# references (<img src="data/review_013/manuscript/death_rates_infage.png">
# / death_rates_pca.png). Kept as-is intentionally — do not rename without
# also updating captions031.md, which we were asked not to touch.
LEGACY_PANEL_FILENAMES = {
    "infage": "death_rates_infage.png",
    "pca": "death_rates_pca.png",
}


def write_panel_csv(var1: str, x, y, x0, y0, x1, y1, boys: int, gals: int, outdir: Path):
    out = pd.DataFrame({var1: x, "total_n": y, "total_rate_per_100k": y / (boys + gals) * 100000})
    out_f = pd.DataFrame({var1: x0, "female_n": y0, "female_rate_per_100k": y0 / gals * 100000})
    out_m = pd.DataFrame({var1: x1, "male_n": y1, "male_rate_per_100k": y1 / boys * 100000})
    merged = out.merge(out_f, on=var1, how="outer").merge(out_m, on=var1, how="outer").sort_values(var1)
    merged.to_csv(outdir / f"figure2_rates_{var1}.csv", index=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_figure2(
    denominator_path: Path,
    numerator_df: pd.DataFrame,
    outdir: Path,
    *,
    infage_col: str = "infage",
    combgest_col: str = "combgest",
    sex_col: str = "sex",
    smoking_col: str = "cig_rec",
    pca_min: int = 36,
    pca_max: int = 90,
    max_combgest: float | None = None,
    require_valid_smoking: bool = True,
) -> int:
    """Returns the final n feeding the PCA panel (post all cleaning)."""
    den = pd.read_parquet(denominator_path)
    boys = int((den[sex_col] == "M").sum())
    gals = int((den[sex_col] == "F").sum())
    LOG.info("Denominator: %d male, %d female live births", boys, gals)

    df = clean_and_compute_pca(
        numerator_df,
        infage_col=infage_col, combgest_col=combgest_col, smoking_col=smoking_col,
        pca_min=pca_min, pca_max=pca_max,
        max_combgest=max_combgest, require_valid_smoking=require_valid_smoking,
    )
    outdir.mkdir(parents=True, exist_ok=True)

    # Compute each panel's series once, reuse for both the individually-named
    # legacy files and the combined manuscript figure.
    series = {var1: compute_sex_specific_series(df, var1) for var1 in ("infage", "pca")}

    # 1. Individual panels under the filenames captions031.md already expects.
    for var1, (x0, y0, x1, y1) in series.items():
        fig_single, ax_single = plt.subplots(figsize=(8, 5))
        _, x, y = visualize_death_rates(x0, y0, x1, y1, boys, gals,
                                          ["orangered", "blue"], 100000, var1, ax_single, pca_max)
        fig_single.tight_layout()
        fig_single.savefig(outdir / LEGACY_PANEL_FILENAMES[var1], dpi=300)
        plt.close(fig_single)
        write_panel_csv(var1, x, y, x0, y0, x1, y1, boys, gals, outdir)
        LOG.info("Wrote %s", outdir / LEGACY_PANEL_FILENAMES[var1])

    # 2. Combined 2-panel Figure2.png/tiff, for the manuscript docx legend.
    fig, axs = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    for ax, var1 in zip(axs, ("infage", "pca")):
        x0, y0, x1, y1 = series[var1]
        visualize_death_rates(x0, y0, x1, y1, boys, gals,
                                ["orangered", "blue"], 100000, var1, ax, pca_max)
    for ax, label in zip(axs, ["A", "B"]):
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                 fontsize=18, fontweight="bold", va="top", ha="left")

    fig.savefig(outdir / "Figure2.tiff", dpi=300, format="tiff")
    fig.savefig(outdir / "Figure2.png")
    plt.close(fig)
    LOG.info("Wrote %s and %s", outdir / "Figure2.png", outdir / "Figure2.tiff")

    return len(df)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rebuild Figure 2 (SUID rates by age scale) with an explicit, "
                    "auditable, up-to-date data pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--denominator", type=Path,
                    default=Path("standardized/consolidated/denominator.parquet"))
    p.add_argument("--numerator-source", choices=["matched_cases_xlsx", "per_icd_parquet"],
                    default="matched_cases_xlsx",
                    help="Which case source to use for the figure. matched_cases_xlsx "
                         "matches Table 1/Figure 3/Figure 4 for a consistent cohort "
                         "across the paper; per_icd_parquet reproduces the notebook's "
                         "original (different) source.")
    p.add_argument("--matched-cases-xlsx", type=Path,
                    default=Path("output_data/controls/matched_cases_cdc.xlsx"))
    p.add_argument("--r95", type=Path, default=Path("standardized/consolidated/R95_num.parquet"))
    p.add_argument("--w75", type=Path, default=Path("standardized/consolidated/W75_num.parquet"))
    p.add_argument("--r99", type=Path, default=Path("standardized/consolidated/R99_num.parquet"))
    p.add_argument("--outdir", type=Path, default=Path("output_figures/manuscript"))

    p.add_argument("--infage-col", default="infage")
    p.add_argument("--combgest-col", default="combgest")
    p.add_argument("--sex-col", default="sex")
    p.add_argument("--smoking-col", default="cig_rec")
    p.add_argument("--pca-min", type=int, default=36)
    p.add_argument("--pca-max", type=int, default=90)
    p.add_argument("--max-combgest", type=float, default=None,
                    help="If set, additionally drops combgest above this value "
                         "(the notebook's original, undocumented behavior used 44). "
                         "Default (unset) matches the rest of the pipeline: only the "
                         "unknown-sentinel (>=99) is dropped.")
    p.add_argument("--require-valid-smoking", dest="require_valid_smoking",
                    action="store_true", default=True,
                    help="Restrict to cig_rec in {Y,N}, matching count_data() used for "
                         "Table 1/Figure 3/Figure 4. Default True.")
    p.add_argument("--no-require-valid-smoking", dest="require_valid_smoking",
                    action="store_false",
                    help="Reproduce the notebook's original Figure 2 (no smoking-"
                         "validity filter at all).")
    p.add_argument("--compare-sources", action="store_true",
                    help="Also load the other numerator source (with the same cleaning "
                         "settings) and report whether total N agrees, without using it "
                         "for the actual figure.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.outdir / "figure2_run.log")],
    )

    if not args.denominator.exists():
        LOG.error("Denominator file not found: %s", args.denominator)
        return 1

    def _load(source: str) -> pd.DataFrame:
        if source == "matched_cases_xlsx":
            return load_numerator_matched_cases(
                args.matched_cases_xlsx,
                infage_col=args.infage_col, combgest_col=args.combgest_col, sex_col=args.sex_col,
            )
        return load_numerator_per_icd(args.r95, args.w75, args.r99)

    LOG.info("Primary numerator source: %s", args.numerator_source)
    primary_df = _load(args.numerator_source)
    LOG.info("Loaded %d raw rows", len(primary_df))

    n_final = build_figure2(
        args.denominator, primary_df, args.outdir,
        infage_col=args.infage_col, combgest_col=args.combgest_col,
        sex_col=args.sex_col, smoking_col=args.smoking_col,
        pca_min=args.pca_min, pca_max=args.pca_max,
        max_combgest=args.max_combgest, require_valid_smoking=args.require_valid_smoking,
    )
    LOG.info("Figure 2 built from n=%d cases (source=%s)", n_final, args.numerator_source)

    if args.compare_sources:
        other_source = "per_icd_parquet" if args.numerator_source == "matched_cases_xlsx" else "matched_cases_xlsx"
        LOG.info("Loading comparison source: %s", other_source)
        try:
            other_raw = _load(other_source)
            other_clean = clean_and_compute_pca(
                other_raw,
                infage_col=args.infage_col, combgest_col=args.combgest_col,
                smoking_col=args.smoking_col, pca_min=args.pca_min, pca_max=args.pca_max,
                max_combgest=args.max_combgest, require_valid_smoking=args.require_valid_smoking,
            )
            n_other = len(other_clean)
            report_path = args.outdir / "figure2_source_comparison.txt"
            with open(report_path, "w") as f:
                f.write("Figure 2 numerator source comparison\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"{args.numerator_source}: raw n={len(primary_df)}, cleaned n={n_final}\n")
                f.write(f"{other_source}: raw n={len(other_raw)}, cleaned n={n_other}\n\n")
                delta = n_final - n_other
                pct = 100 * abs(delta) / max(n_final, n_other)
                f.write(f"Difference in cleaned n: {delta} ({pct:.1f}% of the larger)\n")
                if pct < 1:
                    f.write("These two sources agree closely — likely the same underlying "
                            "case population reached by two different paths.\n")
                else:
                    f.write("These two sources disagree meaningfully. Do not assume Figure 2 "
                            "represents the same cohort as Table 1/Figure 3/Figure 4 without "
                            "reconciling this difference first.\n")
            LOG.info("Wrote source comparison to %s (%s n=%d vs %s n=%d)",
                      report_path, args.numerator_source, n_final, other_source, n_other)
        except FileNotFoundError as e:
            LOG.warning("Could not load comparison source: %s", e)

    LOG.info("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

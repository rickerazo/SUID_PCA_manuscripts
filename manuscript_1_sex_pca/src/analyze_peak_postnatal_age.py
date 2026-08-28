#!/usr/bin/env python3
"""
analyze_peak_postnatal_age.py

Reviewer 2 asked us to clarify that the classic "2 to 4 month" SIDS peak
described in the literature refers to chronologic (postnatal) age, not PCA
(post-conceptional age) — the axis this manuscript's own "peak" (43-46 weeks
PCA, excluded from the windowed NB models, summarized only descriptively) is
expressed on. Rather than asserting agreement with the literature figure by
hand, this script checks it directly: it isolates the exact set of cases in
matched_cases_cdc.xlsx that fall in this manuscript's peak PCA band and
reports their POSTNATAL age (infage, in weeks) — the chronologic-age axis
the reviewer is asking about — broken down by every characteristic that
plausibly shifts that mapping.

No re-derivation of case selection: this reuses clean_case_series(),
exclude_invalid_smoking(), and split_analytic_windows() from
audit_subject_breakdown.py exactly as-is, so `peak_df` here is byte-for-byte
the same set of n=4,438 (default 36/90/43/47 settings) cases already
described in HANDOFF_CONTEXT.md as "peak phase [43,47) (descriptive)" —
individual-level rows, not the weekly-aggregated counts run_windowed_nb_
models.py builds for the regression (that aggregation is PCA-indexed and
would have discarded the individual infage values this analysis needs).

CHARACTERISTICS (each gets its own table + figure, all from one run)
--------------------------------------------------------------------------
PCA = combgest + infage. Two cases can share the same PCA while having very
different postnatal ages if their gestational age at birth differs — a term
infant (combgest~40) reaches PCA 45 at infage~5 weeks, while an infant born
at combgest~30 reaches PCA 45 at infage~15 weeks. That's the whole reason
the manuscript uses PCA rather than chronologic age in the first place, and
it's why "preterm" is the primary breakdown below. Smoking and sex are not
part of the PCA formula and have no mechanistic reason to shift it the same
way; they're included as a check for confounding/interaction in this peak
band, in the same spirit as the manuscript's own smoking x preterm and
sex x preterm interaction terms.

- preterm  (combgest<37 vs >=37)         Supplemental Table 5 (primary)
- smoking  (cig_rec=Y vs N)              Supplemental Table 6
- sex      (male vs female)              Supplemental Table 7

Table numbers above assume no other new supplemental table has been added
elsewhere in the meantime — recheck before using them in the manuscript.

Adding a fourth characteristic later means adding one GroupSpec to
build_group_specs() below; compute_histogram_table(), write_histogram_table(),
and make_histogram() are all written generically against GroupSpec and don't
need to change.

OUTPUTS (written to --outdir; {stem} below is each spec's filename_stem)
--------------------------------------------------------------------------
- peak_band_cases.csv              individual-level rows for the peak-band
                                    cases (pca, infage, combgest, cig_rec,
                                    sex — whatever clean_case_series
                                    retained), for further inspection
- peak_postnatal_age_summary.md    n / mean / median / IQR / min / max of
                                    infage, in weeks AND months, overall and
                                    by each characteristic above; % of
                                    peak-band cases whose postnatal age
                                    falls inside the literature's 2-4 month
                                    window — one file, one section per
                                    characteristic
- {stem}_histogram_data.csv/.md    the exact per-bin counts behind each
                                    figure below, as a labeled table
- {stem}_hist.png                  histogram of infage (weeks, with a
                                    months axis on top) for the peak-band
                                    cases, split by that characteristic,
                                    with the 2-4 month literature window
                                    shaded for visual comparison. Both bars
                                    are always drawn at equal (full bin)
                                    width; distinguished by color/alpha and
                                    draw order (opaque group down first, so
                                    the semi-transparent group blends over
                                    it in the overlap rather than hiding it)

USAGE
-----
    python analyze_peak_postnatal_age.py \
        --input output_data/controls/matched_cases_cdc.xlsx \
        --outdir ./peak_postnatal_age
"""

from __future__ import annotations  # required: tuple[...]/list[...]/X|None below need this on py3.8

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_subject_breakdown import (  # noqa: E402
    clean_case_series,
    exclude_invalid_smoking,
    read_any,
    split_analytic_windows,
)

LOG = logging.getLogger("analyze_peak_postnatal_age")

# Average month length used to convert weeks -> months for comparison against
# the reviewer's "2 to 4 month" phrasing. 30.44 days/month (365.25/12) is the
# standard astronomical-month convention; not a clinical definition, just a
# consistent unit conversion.
DAYS_PER_MONTH = 30.44
WEEKS_PER_MONTH = DAYS_PER_MONTH / 7  # ~4.349
LIT_PEAK_LOW_MONTHS = 2.0
LIT_PEAK_HIGH_MONTHS = 4.0


@dataclass(frozen=True)
class GroupSpec:
    """One binary characteristic to split the peak-band cases by.

    pos_mask_fn(df, column) returns True for the "pos" group — drawn
    opaque/foreground, plotted and listed first. pos is always this
    pipeline's own "index"/exposed convention (preterm, smoking=Y, male —
    the same categories that are the RR=exposed rows in Supplemental Tables
    2/3), so the three plots read consistently rather than an arbitrary
    per-characteristic choice of which side is which.
    """
    key: str
    column: str
    pos_label: str
    neg_label: str
    pos_mask_fn: Callable[[pd.DataFrame, str], pd.Series]
    table_number: int
    filename_stem: str


def build_group_specs(combgest_col: str, smoking_col: str, sex_col: str) -> list[GroupSpec]:
    return [
        GroupSpec(
            key="preterm", column=combgest_col,
            pos_label="Preterm (combgest<37 wk)", neg_label="Term (combgest>=37 wk)",
            pos_mask_fn=lambda df, col: df[col].astype(float) < 37,
            table_number=5,
            filename_stem="peak_postnatal_age",  # legacy name, predates the other two specs
        ),
        GroupSpec(
            key="smoking", column=smoking_col,
            pos_label="Smoking (cig_rec=Y)", neg_label="Non-smoking (cig_rec=N)",
            pos_mask_fn=lambda df, col: df[col].astype(str).str.upper() == "Y",
            table_number=6,
            filename_stem="peak_postnatal_age_by_smoking",
        ),
        GroupSpec(
            key="sex", column=sex_col,
            pos_label="Male", neg_label="Female",
            pos_mask_fn=lambda df, col: df[col].astype(str).str.upper() == "M",
            table_number=7,
            filename_stem="peak_postnatal_age_by_sex",
        ),
    ]


def build_peak_band(
    raw_df: pd.DataFrame,
    *,
    infage_col: str,
    combgest_col: str,
    smoking_col: str,
    pca_min: int,
    pca_max: int,
    peak1: int,
    peak2: int,
):
    """Steps 0-8 of the established pipeline, returning (early, late, peak,
    combined, trail). Identical case-selection logic to every other script
    in this directory — see audit_subject_breakdown.clean_case_series /
    split_analytic_windows docstrings for the step-by-step rationale."""
    case_df, _, trail = clean_case_series(
        raw_df,
        infage_col=infage_col,
        combgest_col=combgest_col,
        pca_min=pca_min,
        pca_max=pca_max,
        pca_max_inclusive=False,
        reproduce_original_bug=False,
    )
    case_df = exclude_invalid_smoking(case_df, trail, smoking_col=smoking_col)
    early, late, peak, combined = split_analytic_windows(
        case_df, trail, peak1=peak1, peak2=peak2
    )
    return early, late, peak, combined, trail


def summarize_infage(df: pd.DataFrame, infage_col: str, label: str) -> dict:
    weeks = df[infage_col].astype(float)
    months = weeks / WEEKS_PER_MONTH
    lit_lo_wk = LIT_PEAK_LOW_MONTHS * WEEKS_PER_MONTH
    lit_hi_wk = LIT_PEAK_HIGH_MONTHS * WEEKS_PER_MONTH
    in_lit_window = weeks.between(lit_lo_wk, lit_hi_wk, inclusive="both")
    return {
        "label": label,
        "n": int(len(weeks)),
        "mean_wk": weeks.mean(),
        "median_wk": weeks.median(),
        "q1_wk": weeks.quantile(0.25),
        "q3_wk": weeks.quantile(0.75),
        "min_wk": weeks.min(),
        "max_wk": weeks.max(),
        "mean_mo": months.mean(),
        "median_mo": months.median(),
        "q1_mo": months.quantile(0.25),
        "q3_mo": months.quantile(0.75),
        "pct_in_lit_2to4mo": 100.0 * in_lit_window.mean() if len(weeks) else float("nan"),
    }


def _to_markdown_table(df: pd.DataFrame) -> str:
    # Matches generate_table1.py's hand-rolled writer: df.to_markdown() needs
    # the `tabulate` package, which isn't reliably installed (confirmed
    # absent from this project's HPC py38 env and this local checkout).
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False))
    return "\n".join([header, sep, body])


def _rows_to_markdown(rows: list[dict]) -> str:
    header = ("| Group | n | Mean (wk) | Median (wk) | IQR (wk) | Mean (mo) | "
              "Median (mo) | IQR (mo) | % within lit. 2-4mo window |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['n']} | {r['mean_wk']:.2f} | {r['median_wk']:.2f} | "
            f"{r['q1_wk']:.1f}-{r['q3_wk']:.1f} | {r['mean_mo']:.2f} | "
            f"{r['median_mo']:.2f} | {r['q1_mo']:.1f}-{r['q3_mo']:.1f} | "
            f"{r['pct_in_lit_2to4mo']:.1f}% |"
        )
    return "\n".join(lines)


def write_summary_md(
    sections: list[dict],
    peak1: int,
    peak2: int,
    n_peak_total: int,
    outpath: Path,
) -> None:
    """sections: one dict per characteristic, each
    {"heading": str, "rows": [summarize_infage(...), ...], "interpretation": str}."""
    lit_lo_wk = LIT_PEAK_LOW_MONTHS * WEEKS_PER_MONTH
    lit_hi_wk = LIT_PEAK_HIGH_MONTHS * WEEKS_PER_MONTH
    with open(outpath, "w") as f:
        f.write(
            f"# Postnatal (chronologic) age of the PCA peak-band cases "
            f"[{peak1},{peak2}) weeks PCA\n\n"
        )
        f.write(f"n = {n_peak_total} total peak-band cases.\n\n")
        f.write(
            f"Literature \"2 to 4 month\" chronologic-age window, converted to weeks "
            f"using {WEEKS_PER_MONTH:.3f} wk/month "
            f"({DAYS_PER_MONTH:.2f} days/month): "
            f"[{lit_lo_wk:.2f}, {lit_hi_wk:.2f}] weeks postnatal.\n\n"
        )
        for sec in sections:
            f.write(f"## {sec['heading']}\n\n")
            f.write(_rows_to_markdown(sec["rows"]))
            f.write("\n\n")
            f.write(sec["interpretation"])
            f.write("\n\n")


def compute_histogram_bins(weeks: pd.Series, bin_width: int = 1) -> np.ndarray:
    """Single source of truth for the bin edges used by both the plot and
    the supplemental-table export below, so the two can never drift apart."""
    lo = int(np.floor(weeks.min()))
    hi = int(np.ceil(weeks.max())) + 1
    return np.arange(lo, hi + bin_width, bin_width)


def compute_histogram_table(peak_df: pd.DataFrame, infage_col: str,
                             spec: GroupSpec, bin_width: int = 1) -> pd.DataFrame:
    """The exact per-bin counts rendered in {spec.filename_stem}_hist.png, as
    a table — for the supplement, since a reader can't extract data from a
    PNG. Same bins as the figure by construction (compute_histogram_bins is
    shared, not re-derived)."""
    weeks = peak_df[infage_col].astype(float)
    pos_mask = spec.pos_mask_fn(peak_df, spec.column)
    bins = compute_histogram_bins(weeks, bin_width=bin_width)

    pos_counts, _ = np.histogram(weeks[pos_mask], bins=bins)
    neg_counts, _ = np.histogram(weeks[~pos_mask], bins=bins)

    table = pd.DataFrame({
        "Postnatal age, bin start (weeks)": bins[:-1],
        "Postnatal age, bin end (weeks, exclusive)": bins[1:],
        f"{spec.neg_label}, n": neg_counts,
        f"{spec.pos_label}, n": pos_counts,
        "Total, n": neg_counts + pos_counts,
    })
    assert table["Total, n"].sum() == len(peak_df), (
        f"histogram table total ({table['Total, n'].sum()}) does not match "
        f"peak-band n ({len(peak_df)}) — bin edges must have dropped a case"
    )
    return table


def write_histogram_table(peak_df: pd.DataFrame, infage_col: str, spec: GroupSpec,
                           peak1: int, peak2: int, outdir: Path) -> pd.DataFrame:
    table = compute_histogram_table(peak_df, infage_col, spec)
    n_total = len(peak_df)
    pos_mask = spec.pos_mask_fn(peak_df, spec.column)
    n_pos = int(pos_mask.sum())
    n_neg = n_total - n_pos

    csv_path = outdir / f"{spec.filename_stem}_histogram_data.csv"
    table.to_csv(csv_path, index=False)

    caption = (
        f"Supplemental Table {spec.table_number}. Postnatal (chronologic) age at "
        f"death, in 1-week bins, for SUID cases within this manuscript's own "
        f"post-conceptional age (PCA) peak band ({peak1}-{peak2 - 1} weeks PCA; "
        f"descriptive only, excluded from the windowed negative binomial "
        f"models), stratified by {spec.key} ({spec.neg_label} vs. {spec.pos_label}). "
        f"n={n_total} total ({n_neg} {spec.neg_label.split(' (')[0].lower()}, "
        f"{n_pos} {spec.pos_label.split(' (')[0].lower()}). Underlies the "
        f"postnatal-age distribution described in the Discussion in response "
        f"to Reviewer 2's comment on PCA vs. chronologic age."
    )
    md_path = outdir / f"{spec.filename_stem}_histogram_data.md"
    with open(md_path, "w") as f:
        f.write(caption + "\n\n")
        f.write(_to_markdown_table(table))
        f.write("\n")

    LOG.info("Wrote %s, %s (%d bins, sums to n=%d)",
              csv_path, md_path, len(table), table["Total, n"].sum())
    return table


def make_histogram(peak_df: pd.DataFrame, infage_col: str, spec: GroupSpec,
                    peak1: int, peak2: int, outpath: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weeks = peak_df[infage_col].astype(float)
    pos_mask = spec.pos_mask_fn(peak_df, spec.column)
    lit_lo_wk = LIT_PEAK_LOW_MONTHS * WEEKS_PER_MONTH
    lit_hi_wk = LIT_PEAK_HIGH_MONTHS * WEEKS_PER_MONTH

    bins = compute_histogram_bins(weeks, bin_width=1)
    # Overlapping, not stacked or side-by-side: both series drawn at the
    # same x position and the same (always equal) width, each on its own
    # y=0 baseline. The opaque ("pos") group goes down first, the
    # semi-transparent ("neg") group is drawn on top — draw order matters
    # here, since an opaque bar on top would fully hide whatever's under it
    # wherever it's taller; a semi-transparent bar on top instead lets both
    # colors blend through the overlap. Same np.histogram/bins approach as
    # compute_histogram_table, so these still match that table exactly
    # (just composited differently on the page).
    pos_counts, _ = np.histogram(weeks[pos_mask], bins=bins)
    neg_counts, _ = np.histogram(weeks[~pos_mask], bins=bins)
    bin_lefts = bins[:-1]
    bin_width = bins[1] - bins[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(lit_lo_wk, lit_hi_wk, color="gold", alpha=0.25,
               label=f"Literature peak ({LIT_PEAK_LOW_MONTHS:.0f}-"
                     f"{LIT_PEAK_HIGH_MONTHS:.0f} mo chronologic age)")
    ax.bar(bin_lefts, pos_counts, width=bin_width, align="edge",
           color="#DD8452", alpha=1.0, edgecolor="white", linewidth=0.3,
           label=spec.pos_label)
    ax.bar(bin_lefts, neg_counts, width=bin_width, align="edge",
           color="#4C72B0", alpha=0.5, edgecolor="white", linewidth=0.3,
           label=spec.neg_label)
    ax.set_xlabel("Postnatal (chronologic) age at death, infage (weeks)")
    ax.set_ylabel("Number of SUID cases")
    ax.set_title(
        f"Postnatal age of cases in the {peak1}-{peak2 - 1} week PCA peak band, "
        f"by {spec.key} (n={len(peak_df)})"
    )

    # Secondary top axis in months, sharing the same data range, for direct
    # visual comparison against the reviewer's "2 to 4 month" phrasing.
    def wk2mo(x):
        return x / WEEKS_PER_MONTH

    def mo2wk(x):
        return x * WEEKS_PER_MONTH

    secax = ax.secondary_xaxis("top", functions=(wk2mo, mo2wk))
    secax.set_xlabel("Postnatal age (months)")

    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


PRETERM_INTERPRETATION = (
    "Interpretation: PCA = gestational age at birth (combgest) + postnatal "
    "age at death (infage). A fixed PCA band therefore maps to a range of "
    "postnatal ages, not a single one, and the range shifts earlier for "
    "term infants and later for preterm infants at the same PCA — that "
    "predictable direction of shift is what this split is checking for. "
    "This is the primary breakdown for Reviewer 2's comment."
)
SMOKING_INTERPRETATION = (
    "Interpretation: smoking status is not part of the PCA formula and has "
    "no mechanistic reason to shift the PCA-to-postnatal-age mapping the "
    "way gestational age does. This breakdown checks whether smoking-"
    "associated peak-band cases nonetheless cluster at a different "
    "postnatal age, independent of the preterm effect above."
)
SEX_INTERPRETATION = (
    "Interpretation: sex is not part of the PCA formula either. This "
    "breakdown checks whether male and female peak-band cases differ in "
    "postnatal-age timing, as a secondary check alongside the manuscript's "
    "main sex-difference analysis."
)
INTERPRETATION_BY_KEY = {
    "preterm": PRETERM_INTERPRETATION,
    "smoking": SMOKING_INTERPRETATION,
    "sex": SEX_INTERPRETATION,
}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                    help="Path to the raw case file, e.g. matched_cases_cdc.xlsx")
    p.add_argument("--outdir", type=Path, default=Path("./peak_postnatal_age"))

    p.add_argument("--infage-col", default="infage")
    p.add_argument("--combgest-col", default="combgest")
    p.add_argument("--smoking-col", default="cig_rec")
    p.add_argument("--sex-col", default="sex")
    p.add_argument("--pca-min", type=int, default=36)
    p.add_argument("--pca-max", type=int, default=90)
    p.add_argument("--peak1", type=int, default=43)
    p.add_argument("--peak2", type=int, default=47)
    p.add_argument("--characteristics", default="preterm,smoking,sex",
                    help="Comma-separated subset of preterm,smoking,sex to run "
                         "(default: all three).")
    p.add_argument("--skip-plot", action="store_true",
                    help="Skip the matplotlib histograms (summary/table CSV/MD still written).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.outdir / "peak_postnatal_age_run.log")],
    )

    if not args.input.exists():
        LOG.error("Input file not found: %s", args.input)
        return 1

    LOG.info("Loading %s", args.input)
    raw_df = read_any(args.input)
    LOG.info("Raw shape: %s", raw_df.shape)

    early, late, peak, combined, trail = build_peak_band(
        raw_df,
        infage_col=args.infage_col, combgest_col=args.combgest_col,
        smoking_col=args.smoking_col,
        pca_min=args.pca_min, pca_max=args.pca_max,
        peak1=args.peak1, peak2=args.peak2,
    )
    trail.to_frame().to_csv(args.outdir / "audit_trail_peak_postnatal_age.csv", index=False)
    LOG.info(
        "Peak band [%d,%d) weeks PCA: n=%d  (early n=%d, late n=%d, for context)",
        args.peak1, args.peak2, len(peak), len(early), len(late),
    )

    if len(peak) == 0:
        LOG.error("No cases fell in the peak band — check --peak1/--peak2 against "
                   "--pca-min/--pca-max, or the input file.")
        return 1

    peak.to_csv(args.outdir / "peak_band_cases.csv", index=False)

    all_specs = build_group_specs(args.combgest_col, args.smoking_col, args.sex_col)
    wanted = {k.strip() for k in args.characteristics.split(",") if k.strip()}
    specs = [s for s in all_specs if s.key in wanted]
    unknown = wanted - {s.key for s in all_specs}
    if unknown:
        LOG.error("Unknown --characteristics value(s): %s (valid: %s)",
                   sorted(unknown), [s.key for s in all_specs])
        return 1

    sections = []
    written_files = []
    for spec in specs:
        pos_mask = spec.pos_mask_fn(peak, spec.column)
        peak_pos = peak.loc[pos_mask]
        peak_neg = peak.loc[~pos_mask]

        rows = [
            summarize_infage(peak, args.infage_col, "All peak-band cases"),
            summarize_infage(peak_neg, args.infage_col, f"  {spec.neg_label}"),
            summarize_infage(peak_pos, args.infage_col, f"  {spec.pos_label}"),
        ]
        for r in rows:
            LOG.info(
                "[%s] %-30s n=%-5d median=%.1f wk (%.1f mo), IQR=%.1f-%.1f wk, "
                "%.1f%% within lit. 2-4mo window",
                spec.key, r["label"], r["n"], r["median_wk"], r["median_mo"],
                r["q1_wk"], r["q3_wk"], r["pct_in_lit_2to4mo"],
            )
        sections.append({
            "heading": f"By {spec.key} ({spec.neg_label} vs. {spec.pos_label})",
            "rows": rows,
            "interpretation": INTERPRETATION_BY_KEY[spec.key],
        })

        write_histogram_table(peak, args.infage_col, spec, args.peak1, args.peak2, args.outdir)
        written_files.append(f"{spec.filename_stem}_histogram_data.{{csv,md}}")

        if not args.skip_plot:
            hist_path = args.outdir / f"{spec.filename_stem}_hist.png"
            make_histogram(peak, args.infage_col, spec, args.peak1, args.peak2, hist_path)
            LOG.info("Wrote %s", hist_path)
            written_files.append(str(hist_path))

    summary_path = args.outdir / "peak_postnatal_age_summary.md"
    write_summary_md(sections, args.peak1, args.peak2, len(peak), summary_path)

    print(f"\nWrote {args.outdir}/peak_band_cases.csv, {summary_path}, "
          + ", ".join(written_files))
    LOG.info("DONE. See %s", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

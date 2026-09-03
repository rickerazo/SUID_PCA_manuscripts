#!/usr/bin/env python3
"""
pca_boundary_report_ncfrp.py

AUDIT A3.17. Diagnose the PCA analytic-window boundaries directly from the NCFRP
case data, so the manuscript can justify PCA_MIN / PCA_MAX by a prespecified
count-stability rule instead of the fixed 36-75 that reviewer 1 objected to for
manuscript 1 (R1.3: "either include more weeks PCA than 75, or give some other
justification ... the top and bottom boundaries of exclusion are not similar").

Ported from manuscript_1_sex_pca/src/audit_subject_breakdown.py::pca_boundary_report,
adapted to the NCFRP columns and to this study's stratification (bed sharing is
the primary exposure; sex-stratified analyses are a sensitivity check, so
sex-specific weekly counts matter too).

Runs the audited case loader with apply_window=False (everything except the PCA
window), then writes:

  pca_weekly_counts.csv        n_total / n by bed-share / n by sex, per PCA week,
                                with a flag for weeks where any tracked subgroup
                                count is below --min-cell, on BOTH tails.
  pca_low_tail_excluded.csv    gestational-age x postnatal-age breakdown of the
                                cases a PCA < --pca-min rule would exclude.
  pca_high_tail_excluded.csv   same, for PCA > --pca-max.
  pca_boundary_diagnostics.txt narrative: where a symmetric percentile trim would
                                cut for comparison, the '>' vs '>=' sensitivity at
                                each bound, and how many weeks on each side fail
                                the stability threshold.

USAGE
    python pca_boundary_report_ncfrp.py \
        --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \
        --outdir output_data/ncfrp/pca_boundary \
        --min-cell 5 --pca-min 36 --pca-max 90
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ncfrp_manuscript_v5 import load_ncfrp_data

LOG = logging.getLogger("pca_boundary_report_ncfrp")


def pca_boundary_report(pre_window_df: pd.DataFrame, outdir: Path, *,
                        pca_min: int, pca_max: int, min_cell: int = 5,
                        tail_pct: float = 1.5) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pre_window_df.copy()
    df["pca"] = pd.to_numeric(df["pca"], errors="coerce")
    df = df.dropna(subset=["pca"])
    df["pca"] = df["pca"].astype(int)
    n_total = len(df)
    if n_total == 0:
        LOG.warning("empty input, nothing to diagnose")
        return

    full_index = range(int(df["pca"].min()), int(df["pca"].max()) + 1)
    weekly = (df.groupby("pca").size().reindex(full_index, fill_value=0)
              .rename("n_total").to_frame())
    weekly["n_bedshare"] = (df.loc[df["bed_share"] == "Y"].groupby("pca").size()
                            .reindex(weekly.index, fill_value=0))
    weekly["n_non_bedshare"] = (df.loc[df["bed_share"] == "N"].groupby("pca").size()
                                .reindex(weekly.index, fill_value=0))
    weekly["n_male"] = (df.loc[df["sex"] == "M"].groupby("pca").size()
                        .reindex(weekly.index, fill_value=0))
    weekly["n_female"] = (df.loc[df["sex"] == "F"].groupby("pca").size()
                          .reindex(weekly.index, fill_value=0))
    tracked = ["n_bedshare", "n_non_bedshare", "n_male", "n_female"]
    weekly["sparse_subgroup"] = (weekly[tracked] < min_cell).any(axis=1)
    weekly["in_chosen_window"] = (weekly.index >= pca_min) & (weekly.index <= pca_max)
    weekly["cum_pct_at_or_below"] = 100 * weekly["n_total"].cumsum() / n_total
    weekly["cum_pct_at_or_above"] = 100 * weekly["n_total"][::-1].cumsum()[::-1] / n_total
    weekly.index.name = "pca_week"
    weekly.to_csv(outdir / "pca_weekly_counts.csv")

    s = df["pca"].sort_values()
    lo_cut = s.quantile(tail_pct / 100)
    hi_cut = s.quantile(1 - tail_pct / 100)

    def _tail_breakdown(mask: pd.Series, label: str) -> pd.DataFrame:
        sub = df.loc[mask]
        if sub.empty:
            return pd.DataFrame(columns=["tail", "gestational_age_bin",
                                         "postnatal_age_bin", "n"])
        ga_bin = pd.cut(sub["inf3gestage"], bins=[0, 27, 31, 33, 36, 200],
                        labels=["<28wk (extremely preterm)", "28-31wk (very preterm)",
                                "32-33wk (moderate preterm)", "34-36wk (late preterm)",
                                ">=37wk (term)"])
        pna_bin = pd.cut(sub["infageweeks"], bins=[-1, 1, 4, 8, 12, 1000],
                         labels=["<=1wk", "2-4wk", "5-8wk", "9-12wk", ">12wk"])
        out = (sub.assign(gestational_age_bin=ga_bin, postnatal_age_bin=pna_bin)
               .groupby(["gestational_age_bin", "postnatal_age_bin"], observed=True)
               .size().rename("n").reset_index())
        out = out.loc[out["n"] > 0].sort_values("n", ascending=False)
        out.insert(0, "tail", label)
        return out

    _tail_breakdown(df["pca"] < pca_min, f"PCA < {pca_min}").to_csv(
        outdir / "pca_low_tail_excluded.csv", index=False)
    _tail_breakdown(df["pca"] > pca_max, f"PCA > {pca_max}").to_csv(
        outdir / "pca_high_tail_excluded.csv", index=False)

    n_low = int((df["pca"] < pca_min).sum())
    n_high = int((df["pca"] > pca_max).sum())
    n_high_incl = int((df["pca"] >= pca_max).sum())
    n_at_max = int((df["pca"] == pca_max).sum())
    n_at_min = int((df["pca"] == pca_min).sum())
    sparse_low = int(weekly.loc[weekly.index < pca_min, "sparse_subgroup"].sum())
    sparse_in = int(weekly.loc[weekly["in_chosen_window"], "sparse_subgroup"].sum())
    sparse_high = int(weekly.loc[weekly.index > pca_max, "sparse_subgroup"].sum())

    with open(outdir / "pca_boundary_diagnostics.txt", "w") as f:
        f.write("PCA window boundary diagnostics (NCFRP; for the Methods section)\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Cases after all cleaning EXCEPT the PCA window: n = {n_total}\n\n")
        f.write(f"Candidate window: {pca_min} <= PCA <= {pca_max}\n")
        f.write(f"  Excluded PCA < {pca_min}: n = {n_low} "
                f"({100 * n_low / n_total:.2f}%)\n")
        f.write(f"  Excluded PCA > {pca_max}: n = {n_high} "
                f"({100 * n_high / n_total:.2f}%)\n\n")
        f.write("Boundary-operator sensitivity ('>' vs '>='):\n")
        f.write(f"  n at exactly PCA == {pca_max}: {n_at_max}. "
                f"'> {pca_max}' excludes {n_high}; '>= {pca_max}' would exclude "
                f"{n_high_incl}. Match whichever the current Figure 1 shows.\n")
        f.write(f"  n at exactly PCA == {pca_min}: {n_at_min} (INCLUDED; lower "
                f"bound is '< {pca_min}').\n\n")
        f.write(f"Symmetric {tail_pct}%-per-tail trim, for comparison:\n")
        f.write(f"  would cut at approximately PCA <= {lo_cut:.1f} and "
                f"PCA >= {hi_cut:.1f}. This will NOT match a count-stability "
                f"window because the case PCA distribution is right-skewed "
                f"(fast rise, long decline). State which rule the manuscript "
                f"uses and why.\n\n")
        f.write(f"Count-stability check (min_cell = {min_cell}, tracked subgroups: "
                f"bed-share / non-bed-share / male / female):\n")
        f.write(f"  weeks BELOW {pca_min} with a sparse subgroup: {sparse_low} "
                f"of {int((weekly.index < pca_min).sum())}\n")
        f.write(f"  weeks ABOVE {pca_max} with a sparse subgroup: {sparse_high} "
                f"of {int((weekly.index > pca_max).sum())}\n")
        f.write(f"  weeks INSIDE [{pca_min},{pca_max}] with a sparse subgroup: "
                f"{sparse_in} (target: 0)\n\n")
        f.write("Pick pca_min / pca_max as the widest contiguous block of weeks "
                "with no sparse subgroup, then pass those to "
                "ncfrp_manuscript_v5.py via --pca-min / --pca-max. See "
                "pca_weekly_counts.csv for the week-by-week table.\n")

    LOG.info("Wrote PCA boundary diagnostics to %s (low tail n=%d, high tail n=%d, "
             "sparse weeks inside window=%d)", outdir, n_low, n_high, sparse_in)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", type=Path,
                   default=Path("standardized/NCFRP_SCH_LR_BW_pop.csv"))
    p.add_argument("--outdir", type=Path,
                   default=Path("output_data/ncfrp/pca_boundary"))
    p.add_argument("--min-cell", type=int, default=5)
    p.add_argument("--pca-min", type=int, default=36)
    p.add_argument("--pca-max", type=int, default=90)
    p.add_argument("--tail-pct", type=float, default=1.5)
    p.add_argument("--year-min", type=int, default=None,
                   help="restrict to death years >= this before diagnosing the window")
    p.add_argument("--year-max", type=int, default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    # everything except the PCA window
    pre_window, _ = load_ncfrp_data(args.data_path, apply_window=False, audit_csv=None,
                                    year_min=args.year_min, year_max=args.year_max)
    pca_boundary_report(pre_window, args.outdir, pca_min=args.pca_min,
                        pca_max=args.pca_max, min_cell=args.min_cell,
                        tail_pct=args.tail_pct)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Supplementary table: sensitivity of the windowed and full-curve estimates to
(a) the age-at-death exclusion rule and (b) the offset mode.

Reads ONLY saved pipeline outputs (nothing re-fit, nothing hand-typed):

  --primary-dir      out_joint/output_data/ncfrp of the primary run
                     (run_all.sh 36 90 2008 2019 1  -> deaths at 7-13 d excluded)
  --sensitivity-dir  out_joint/output_data/ncfrp of the first-week-cut run
                     (run_all.sh 36 90 2008 2019 0  -> deaths at 7-13 d retained)
  --marginal-dir     out_marginal/output_data/ncfrp of the primary run
                     (same cohort as --primary-dir, marginal instead of joint offsets)

Each directory must hold all_early_irr.csv, all_late_irr.csv, all_gam_exp.csv and
two_population_Ns.csv. Writes <outdir>/<stem>.{csv,md,xlsx}.

Background (V7_REVISION_LOG.md, 2026-10-29): the NCFRP export contains no deaths
before 7 days of age, so FIRST_WEEK_CUT=0 removes nothing and the sensitivity run
is "primary cohort plus the deaths at 7-13 days". All of those fall below PCA 47,
so the late window is identical between the two joint runs; the script checks that
and refuses to write if it is not.

Example (laptop, against the frozen bundles):
  python src/generate_supp_firstweek_table.py \
      --primary-dir     AJE/v6/out_joint/output_data/ncfrp \
      --sensitivity-dir AJE/v7/out_joint/output_data/ncfrp \
      --marginal-dir    AJE/v6/out_marginal/output_data/ncfrp \
      --outdir . --stem supp_table_firstweek_sensitivity
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

TERM_LABELS = {
    "pca": "PCA, per week",
    "bed_share": "Bed sharing",
    "smoke": "Maternal smoking",
    "preterm": "Preterm birth",
    "bed_share:smoke": "Bed sharing x maternal smoking",
    "bed_share:preterm": "Bed sharing x preterm birth",
}
EARLY_TERMS = ["pca", "bed_share", "smoke", "preterm", "bed_share:smoke", "bed_share:preterm"]
LATE_TERMS = ["pca", "bed_share", "smoke", "preterm", "bed_share:smoke", "bed_share:preterm"]
GAM_TERMS = ["bed_share", "smoke", "preterm"]


def read_irr(path: Path) -> dict[str, tuple[float, float, float]]:
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["term"]] = (float(row["IRR"]), float(row["IRR_ci_low"]), float(row["IRR_ci_high"]))
    return out


def fmt(est: tuple[float, float, float], decimals: int) -> str:
    irr, lo, hi = est
    return f"{irr:.{decimals}f} ({lo:.{decimals}f}-{hi:.{decimals}f})"


def read_ns(path: Path) -> dict[str, int]:
    with open(path, newline="") as f:
        row = next(csv.DictReader(f))
    return {k: int(v) for k, v in row.items()}


def build(primary: Path, sensitivity: Path, marginal: Path) -> tuple[pd.DataFrame, str]:
    runs = {"primary": primary, "sensitivity": sensitivity, "marginal": marginal}
    early = {k: read_irr(d / "all_early_irr.csv") for k, d in runs.items()}
    late = {k: read_irr(d / "all_late_irr.csv") for k, d in runs.items()}
    gam = {k: read_irr(d / "all_gam_exp.csv") for k, d in runs.items()}
    ns = {k: read_ns(d / "two_population_Ns.csv") for k, d in runs.items()}

    # Guard: the two joint runs must agree on the late window (the retained
    # deaths all fall below PCA 47). A mismatch means the wrong bundle was passed.
    for term in LATE_TERMS:
        if late["primary"][term] != late["sensitivity"][term]:
            raise SystemExit(
                f"late-window '{term}' differs between --primary-dir and --sensitivity-dir; "
                "these should be identical for the first-week-cut comparison. Check the paths.")
    if ns["primary"]["late_n"] != ns["sensitivity"]["late_n"]:
        raise SystemExit("late_n differs between --primary-dir and --sensitivity-dir; check the paths.")
    if ns["primary"] != ns["marginal"]:
        raise SystemExit("--marginal-dir cohort Ns differ from --primary-dir; the marginal run must be "
                         "the same cohort as the primary run.")

    cols = ["Term",
            "Primary (joint offsets; deaths at 7-13 days excluded)",
            "Deaths at 7-13 days retained (joint offsets)",
            "Marginal offsets (deaths at 7-13 days excluded)"]
    rows: list[list[str]] = []

    def block(title: str, est: dict, terms: list[str]) -> None:
        rows.append([title, "", "", ""])
        for t in terms:
            d = 3 if t == "pca" else 2
            rows.append([TERM_LABELS[t], fmt(est["primary"][t], d), fmt(est["sensitivity"][t], d),
                         fmt(est["marginal"][t], d)])

    block("Early window (PCA <43 weeks)", early, EARLY_TERMS)
    block("Late window (PCA 47-90 weeks)", late, LATE_TERMS)
    block("Full-curve GAM (PCA 36-90 weeks)", gam, GAM_TERMS)
    rows.append(["Cohort", "", "", ""])
    for label, key in [("Analytic N", "full_analytic_N"), ("Early-window n", "early_n"),
                       ("Late-window n", "late_n")]:
        rows.append([label] + [f"{ns[k][key]:,}" for k in ("primary", "sensitivity", "marginal")])

    tab = pd.DataFrame(rows, columns=cols)
    added = ns["sensitivity"]["full_analytic_N"] - ns["primary"]["full_analytic_N"]
    note = ("Note: values are incidence rate ratios (95% CI). The NCFRP export contains no deaths "
            "before 7 days of age, so the sensitivity run retains only the deaths at 7-13 days "
            f"({added:,} enter the analytic cohort); all fall below PCA 47 weeks, so the late-window "
            "estimates are identical under both rules. Marginal offsets use the primary cohort with "
            "the smoking and preterm denominators taken from marginal rather than cross-classified "
            "live-birth distributions. Values trace to "
            f"{primary} (primary), {sensitivity} (age-at-death sensitivity) and {marginal} "
            "(offset sensitivity).")
    return tab, note


def write(tab: pd.DataFrame, note: str, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{stem}.csv"
    tab.to_csv(csv_path, index=False)
    with open(csv_path, "a", newline="") as f:
        f.write("\n" + note + "\n")
    with open(outdir / f"{stem}.md", "w") as f:
        f.write(tab.to_markdown(index=False) + "\n\n" + note + "\n")
    xlsx_path = outdir / f"{stem}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        tab.to_excel(xw, index=False, sheet_name="Sensitivity")
        ws = xw.sheets["Sensitivity"]
        for col in ("A", "B", "C", "D"):
            ws.column_dimensions[col].width = 34
        ws.cell(row=len(tab) + 3, column=1, value=note)
    for p in (csv_path, outdir / f"{stem}.md", xlsx_path):
        print(f"wrote {p}")
    print(tab.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--primary-dir", type=Path, required=True)
    p.add_argument("--sensitivity-dir", type=Path, required=True)
    p.add_argument("--marginal-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("output_data/ncfrp"))
    p.add_argument("--stem", default="supp_table_firstweek_sensitivity")
    args = p.parse_args(argv)
    tab, note = build(args.primary_dir, args.sensitivity_dir, args.marginal_dir)
    write(tab, note, args.outdir, args.stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())

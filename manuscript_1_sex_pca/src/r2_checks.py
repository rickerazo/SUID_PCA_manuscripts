#!/usr/bin/env python3
"""
r2_checks.py -- Verification checks run during the second round of peer review.
Log lines are labelled with the check IDs below.

Reuses the pipeline scripts in this folder (no new cleaning or modeling logic).

V-01  Supplemental Table 1: early-window NB model with the lower bound at
      PCA 36 (main analysis) and at PCA 38 (sensitivity column). The 36 column
      must reproduce the main early-window estimate (Male IRR 1.14, 0.85-1.53);
      the script checks this and logs PASS/FAIL.
V-02  Case counts by PCA week x preterm (combgest<=36) x sex for PCA 36-46,
      the minimum PCA among term cases, and an infage-units check. Confirms
      that PCA weeks 36-38 contain preterm infants only (Methods).
V-04  Whether the natality denominator has a BFED column, and its Y/N/U
      distribution overall and by sex and preterm status (input to
      bfed_sensitivity.py).
A-11  Total live births in the denominator, by sex and with valid maternal
      smoking (the n quoted in the Results).

USAGE
    python r2_checks.py \
        --input output_data/controls/matched_cases_cdc.xlsx \
        --exposure-dir output_data/risk_exposure \
        --denominator /path/to/denominator.parquet \
        --outdir r2_checks_out
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("r2_checks")

# Published early-window estimate (run_windowed_nb_models.py, PCA 36-42).
EXPECTED_EARLY_36 = {"Male (vs female)": (1.139275, 0.846135, 1.533973)}


def fmt_irr(r) -> str:
    return f"{r.IRR:.2f} ({r.CI_low:.2f}–{r.CI_high:.2f})"


def fmt_p(p) -> str:
    if pd.isna(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def v01_supp_table1(raw_df, args, P) -> None:
    cols = {}
    for pca_min in (36, 38):
        case_df, _, trail = P.clean_case_series(
            raw_df, infage_col=args.infage_col, combgest_col=args.combgest_col,
            pca_min=pca_min, pca_max=args.pca_max, pca_max_inclusive=False,
            reproduce_original_bug=False,
        )
        trail.to_frame().to_csv(args.outdir / f"v01_audit_trail_pcamin{pca_min}.csv", index=False)
        _, x1, _, y1, _, m1, _ = P.fit_windowed_models(
            case_df, peak1=args.peak1, peak2=args.peak2, peak3=args.pca_max,
            exposure_dir=args.exposure_dir, combgest_col=args.combgest_col,
            smoking_col=args.smoking_col, sex_col=args.sex_col,
        )
        conv = bool(m1.mle_retvals.get("converged", False))
        LOG.info("V-01 pca_min=%d: early window %d cases, %d strata rows, converged=%s",
                 pca_min, int(y1.sum()), len(x1), conv)
        t = P.irr_table(m1, "pca", int(y1.sum()))
        t.to_csv(args.outdir / f"v01_irr_early_pcamin{pca_min}.csv", index=False)
        cols[pca_min] = t.set_index("Predictor")

    # Reproduction check against the corrected R1 run.
    for name, (irr, lo, hi) in EXPECTED_EARLY_36.items():
        r = cols[36].loc[name]
        ok = np.allclose([r.IRR, r.CI_low, r.CI_high], [irr, lo, hi], atol=5e-4)
        LOG.info("V-01 reproduction check (%s, pca_min=36): got %.4f (%.4f-%.4f), "
                 "expected %.4f (%.4f-%.4f) -> %s",
                 name, r.IRR, r.CI_low, r.CI_high, irr, lo, hi, "PASS" if ok else "FAIL")

    stat_rows = {"Alpha (overdispersion)", "Observations (strata rows)",
                 "Total cases in window", "Log-likelihood", "Pseudo-R2"}
    out = []
    for pred in cols[36].index:
        row = {"Predictor": pred}
        for k in (36, 38):
            r = cols[k].loc[pred] if pred in cols[k].index else None
            if r is None:
                row[f"PCA{k}_IRR (95% CI)"], row[f"PCA{k}_p"] = "", ""
            elif pred in stat_rows:
                row[f"PCA{k}_IRR (95% CI)"], row[f"PCA{k}_p"] = f"{r.IRR:.4g}", "—"
            else:
                row[f"PCA{k}_IRR (95% CI)"], row[f"PCA{k}_p"] = fmt_irr(r), fmt_p(r.p)
        out.append(row)
    pd.DataFrame(out).to_csv(args.outdir / "v01_supplemental_table1.csv", index=False)
    LOG.info("V-01 Supplemental Table 1:\n%s", pd.DataFrame(out).to_string(index=False))


def v02_pca_by_term(raw_df, args, P) -> None:
    case_df, _, _ = P.clean_case_series(
        raw_df, infage_col=args.infage_col, combgest_col=args.combgest_col,
        pca_min=args.pca_min, pca_max=args.pca_max, pca_max_inclusive=False,
        reproduce_original_bug=False,
    )
    d = case_df.loc[case_df[args.smoking_col].isin(["Y", "N"])].copy()  # as count_data()
    d["preterm"] = (d[args.combgest_col] <= 36).astype(int)             # as count_data()
    LOG.info("V-02 analytic cases with valid cig_rec: n=%d (expected 23,686)", len(d))

    # infage units: PCA = combgest + infage, so infage must be in weeks.
    LOG.info("V-02 infage summary (weeks expected): min=%s, max=%s, median=%s",
             d[args.infage_col].min(), d[args.infage_col].max(), d[args.infage_col].median())
    mism = int((d["pca"] != d[args.combgest_col] + d[args.infage_col]).sum())
    LOG.info("V-02 rows where pca != combgest + infage: %d", mism)

    term = d.loc[d["preterm"] == 0]
    LOG.info("V-02 minimum PCA among term cases (combgest>=37): %s; minimum infage among term: %s",
             term["pca"].min(), term[args.infage_col].min())

    sub = d.loc[d["pca"].between(36, 46)]
    tab = (sub.groupby(["pca", "preterm", args.sex_col]).size()
              .unstack([1, 2], fill_value=0).sort_index())
    tab.columns = [f"{'preterm' if p else 'term'}_{s}" for p, s in tab.columns]
    tab["total"] = tab.sum(axis=1)
    tab["pct_preterm"] = 100 * tab.filter(like="preterm_").sum(axis=1) / tab["total"]
    tab.to_csv(args.outdir / "v02_pca_by_term_sex_36_46.csv")
    LOG.info("V-02 counts by PCA week x term/preterm x sex:\n%s", tab.to_string())


def v04_denominator_bfed(args) -> None:
    if args.denominator is None:
        LOG.info("V-04 skipped (no --denominator given)")
        return
    try:
        import pyarrow.parquet as pq
        names = pq.ParquetFile(args.denominator).schema.names
    except Exception:
        names = list(pd.read_parquet(args.denominator).columns)
    hit = [c for c in names if c.lower() == args.bfed_col.lower()]
    LOG.info("V-04 denominator has %d columns; BFED column present: %s", len(names), hit or "NO")
    # A-11: total live births in the denominator (Results: n=30,656,644, 2014-2021)
    base = pd.read_parquet(args.denominator, columns=["sex", "cig_rec"])
    LOG.info("A-11 denominator rows: %d; by sex: %s; rows with cig_rec in {Y,N}: %d",
             len(base), base["sex"].value_counts(dropna=False).to_dict(),
             int(base["cig_rec"].isin(["Y", "N"]).sum()))
    if not hit:
        return
    den = pd.read_parquet(args.denominator, columns=[hit[0], "sex", "combgest", "cig_rec"])
    col = hit[0]
    LOG.info("V-04 denominator BFED distribution (n=%d):\n%s", len(den),
             den[col].value_counts(dropna=False).to_string())
    den["preterm"] = np.where(den["combgest"] >= 99, np.nan, (den["combgest"] < 37).astype(float))
    by = pd.crosstab([den["sex"], den["preterm"]], den[col], dropna=False, margins=True)
    by.to_csv(args.outdir / "v04_denominator_bfed_by_sex_preterm.csv")
    LOG.info("V-04 BFED by sex x preterm:\n%s", by.to_string())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pipeline-dir", type=Path, default=Path(__file__).resolve().parent,
                   help="Folder with the R1 scripts; defaults to this script's own folder")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--exposure-dir", type=Path, default=Path("output_data/risk_exposure"))
    p.add_argument("--denominator", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=Path("r2_checks_out"))
    p.add_argument("--bfed-col", default="bfed")
    p.add_argument("--infage-col", default="infage")
    p.add_argument("--combgest-col", default="combgest")
    p.add_argument("--sex-col", default="sex")
    p.add_argument("--smoking-col", default="cig_rec")
    p.add_argument("--pca-min", type=int, default=36)
    p.add_argument("--pca-max", type=int, default=90)
    p.add_argument("--peak1", type=int, default=43)
    p.add_argument("--peak2", type=int, default=47)
    p.add_argument("--only", nargs="*", choices=["v01", "v02", "v04"], default=["v01", "v02", "v04"])
    args = p.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.outdir / "r2_checks.log")])

    sys.path.insert(0, str(args.pipeline_dir.resolve()))
    import types
    from audit_subject_breakdown import clean_case_series, read_any
    from run_windowed_nb_models import fit_windowed_models, irr_table
    P = types.SimpleNamespace(clean_case_series=clean_case_series,
                              fit_windowed_models=fit_windowed_models, irr_table=irr_table)

    if {"v01", "v02"} & set(args.only):
        LOG.info("Loading %s", args.input)
        raw_df = read_any(args.input)
        LOG.info("Raw shape: %s", raw_df.shape)
        if "v01" in args.only:
            v01_supp_table1(raw_df, args, P)
        if "v02" in args.only:
            v02_pca_by_term(raw_df, args, P)
    if "v04" in args.only:
        v04_denominator_bfed(args)
    LOG.info("DONE. See %s", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

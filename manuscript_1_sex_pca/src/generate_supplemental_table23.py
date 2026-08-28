#!/usr/bin/env python3
"""
generate_supplemental_table23.py

Rebuilds Supplemental Tables 2 and 3 (profile-specific relative risks and
derived absolute risks, early and late developmental windows) on the
CORRECTED case series/models, fixing several real problems found while
investigating the reviewer comment this responds to ("remove the Prevalence
column from Supplemental Table 2, move that information into a proper
Table 1" — see generate_table1.py for the Table 1 half).

WHAT WAS WRONG WITH THE CURRENT (DOCX) SUPPLEMENTAL TABLES 2/3
------------------------------------------------------------------------------
1. Stale numbers. Every RR in the current Supplemental Table 2 traces back to
   the PRE-correction pipeline (e.g. "female preterm nonsmoking" RR=2.89),
   which flatly contradicts Figure 4A / Supplemental Table 1 in the SAME
   document, already rebuilt this session on the corrected pipeline, which
   report Preterm birth IRR=7.14 for the identical early-window model. Two
   tables in one manuscript reporting different numbers for the same
   coefficient is the kind of thing a reviewer or editor will catch.
2. A CI that cannot be correct. "female preterm smoking" in the current
   Supplemental Table 2 shows RR=17.58 with 95% CI (0.67-1.55) — a CI that
   does not contain its own point estimate. That interval is actually
   Supplemental Table 1's Smoking x Preterm interaction-term CI, reused by
   mistake for a compound profile it doesn't describe.
3. Absolute Risk silently ignores sex. In the current early-window table
   (Supplemental Table 2), "female fullterm nonsmoking" and "male fullterm
   nonsmoking" show the IDENTICAL Absolute Risk (33824.15) despite having
   different RRs (1.00 vs 1.06) — same for every other smoking/preterm pair.
   The late-window table (Supplemental Table 3) does NOT have this problem
   (its male/female absolute risks differ correctly), so this looks like a
   one-off computation bug specific to the early-window table, not an
   intentional simplification.
4. No script anywhere in this repo or the original notebook builds
   "Population attributable component" or "Population weighted risk" (grep
   confirms neither term exists anywhere in the notebook), and the numbers
   in the current tables for those two columns could not be independently
   verified or reverse-engineered with confidence from the visible values.
   Per corresponding-author decision (2026-08-11): "Population attributable
   component" is now computed via the standard generalized/category-specific
   population attributable fraction formula (see population_attributable_
   component() below) — a well-established textbook quantity, not a guess
   at the original bespoke computation. "Population weighted risk" had no
   defensible standard-formula candidate at all and is DROPPED entirely
   rather than invented.

WHAT THIS SCRIPT COMPUTES, AND HOW
------------------------------------------------------------------------------
- Relative Risk (95% CI) per profile: a genuine linear contrast against the
  SAME fitted NegativeBinomial model objects Figure 4/Supplemental Table 1
  use (via run_windowed_nb_models.fit_windowed_models(), not re-derived by
  hand), using statsmodels' own `model.t_test()` for the contrast SE/CI/p —
  standard delta-method inference on a linear combination of coefficients,
  not a new statistical method. PCA's own coefficient is excluded from the
  contrast because it's held equal between the numerator and reference
  profile (same PCA week), so it cancels in the ratio, matching how the
  original table's reference-profile RR was already exactly 1.00 regardless
  of PCA.
- Prevalence per profile: TRUE joint prevalence (not product of marginals)
  computed directly from the population denominator file, with preterm
  defined as combgest<37 (matching the manuscript's own stated definition
  and risk_exposure_fixed.py — NOT the notebook's inconsistent combgest<38
  used in its own compute_stratified_prevalence(), a discrepancy found while
  reconstructing this).
- Absolute Risk per profile (per 100,000 live births): baseline_rate (the
  empirical reference-profile case rate: observed reference-profile cases in
  that window / reference-profile live births x 100,000) x RR_profile. This
  is model-derived (uses the corrected model's RR) AND prevalence-informed
  (baseline anchored to the real reference-profile denominator), matching
  the current tables' own caption text, and now correctly varies by sex
  (fixing problem #3 above).

USAGE
-----
    python generate_supplemental_table23.py \
        --input output_data/controls/matched_cases_cdc.xlsx \
        --exposure-dir output_data/risk_exposure \
        --denominator standardized/consolidated/denominator.parquet \
        --outdir .
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_subject_breakdown import clean_case_series, read_any  # noqa: E402
from run_windowed_nb_models import fit_windowed_models  # noqa: E402

LOG = logging.getLogger("generate_supplemental_table23")

# (label, sex, smoke, preterm) — 8 mutually exclusive profiles, matching the
# current Supplemental Table 2/3 row set exactly (row order preserved).
PROFILES = [
    ("female fullterm nonsmoking", 0, 0, 0),
    ("female fullterm smoking",    0, 1, 0),
    ("female preterm nonsmoking",  0, 0, 1),
    ("female preterm smoking",     0, 1, 1),
    ("male fullterm nonsmoking",   1, 0, 0),
    ("male fullterm smoking",      1, 1, 0),
    ("male preterm nonsmoking",    1, 0, 1),
    ("male preterm smoking",       1, 1, 1),
]



# ---------------------------------------------------------------------------
# Profile-specific RR/CI/p via linear contrasts on the fitted model
# ---------------------------------------------------------------------------

def profile_contrast_vector(model, sex: int, smoke: int, preterm: int) -> np.ndarray:
    """
    Build a contrast row matching model.params' column order. const/pca are
    left at 0 (they cancel between numerator and reference profile, held at
    the same PCA week); alpha (if present, the NB dispersion param appended
    by NegativeBinomial.fit()) is also left at 0 since it's not part of the
    linear predictor.
    """
    c = pd.Series(0.0, index=model.params.index)
    for name, val in (("sex", sex), ("smoke", smoke), ("preterm", preterm)):
        if name in c.index:
            c[name] = val
    if "smoke*preterm" in c.index:
        c["smoke*preterm"] = smoke * preterm
    if "sex*preterm" in c.index:
        c["sex*preterm"] = sex * preterm
    return c.values


def profile_rr_table(model) -> pd.DataFrame:
    rows = []
    for label, sex, smoke, preterm in PROFILES:
        contrast = profile_contrast_vector(model, sex, smoke, preterm)
        # Point estimate (exp(contrast . params)) doesn't need the covariance
        # matrix and is always available; the CI/p-value do, and can fail on
        # a poorly-conditioned fit (e.g. near-separation in a sparse
        # stratum) even when the point estimate itself is fine. Don't let a
        # missing covariance matrix take down every other row/table.
        rr = float(np.exp(np.dot(contrast, model.params.values)))
        try:
            tt = model.t_test(contrast)
            ci_lo, ci_hi = np.exp(tt.conf_int()[0])
            p = float(tt.pvalue) if np.ndim(tt.pvalue) == 0 else float(tt.pvalue[0])
        except ValueError as e:
            LOG.warning("  No covariance available for profile %r (%s) — CI/p set to NaN: %s",
                        label, e, model.model.__class__.__name__)
            ci_lo = ci_hi = p = np.nan
        rows.append({
            "Labels": label, "sex": sex, "smoke": smoke, "preterm": preterm,
            "Relative Risk": rr, "RR_CI_low": float(ci_lo), "RR_CI_high": float(ci_hi),
            "RR_p": p,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# True joint prevalence from the population denominator
# ---------------------------------------------------------------------------

def joint_prevalence(
    denominator_path: Path,
    *,
    sex_col: str = "sex",
    combgest_col: str = "combgest",
    smoking_col: str = "cig_rec",
) -> pd.DataFrame:
    den = pd.read_parquet(denominator_path)
    n0 = len(den)
    den = den.loc[den[combgest_col] < 99].copy()  # drop unknown-gestation sentinel first
    den = den.loc[den[smoking_col].isin(["Y", "N"])].copy()
    den = den.loc[den[sex_col].isin(["M", "F"])].copy()
    LOG.info("Denominator: n=%d -> n=%d after dropping unknown gestation/invalid "
              "smoking/sex", n0, len(den))

    den["_preterm"] = (den[combgest_col] < 37).astype(int)  # matches Methods: <37 completed weeks
    total_n = len(den)

    rows = []
    for label, sex, smoke, preterm in PROFILES:
        sex_val = "M" if sex == 1 else "F"
        smoke_val = "Y" if smoke == 1 else "N"
        mask = (
            (den[sex_col] == sex_val)
            & (den[smoking_col] == smoke_val)
            & (den["_preterm"] == preterm)
        )
        n_profile = int(mask.sum())
        rows.append({
            "Labels": label, "sex": sex, "smoke": smoke, "preterm": preterm,
            "n_live_births": n_profile, "Prevalence": n_profile / total_n,
        })
    out = pd.DataFrame(rows)
    LOG.info("Joint prevalence sums to %.4f across the 8 profiles (should be ~1.0)",
              out["Prevalence"].sum())
    return out


# ---------------------------------------------------------------------------
# Absolute risk: empirical reference-profile rate x model-derived RR
# ---------------------------------------------------------------------------

def observed_profile_counts(x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """x/y are the aggregated (pca week x sex x smoke x preterm) strata rows
    and counts returned by fit_windowed_models() — sum counts within each of
    the 8 profiles, collapsing across PCA week, for the window x/y cover."""
    df = x[["sex", "smoke", "preterm"]].copy()
    df["count"] = y
    grouped = df.groupby(["sex", "smoke", "preterm"], as_index=False)["count"].sum()
    return grouped


def population_attributable_component(table: pd.DataFrame) -> pd.Series:
    """
    Category-specific population attributable fraction (generalized Levin's
    formula for multiple mutually exclusive exposure categories; Miettinen
    1974): PAF_i = prevalence_i*(RR_i-1) / (1 + sum_j prevalence_j*(RR_j-1)).
    Per-category PAF_i values sum to the overall population attributable
    fraction across all 8 profiles. Applied per corresponding-author
    decision (2026-08-11) as the standard candidate for what "Population
    attributable component" meant in the original table; "Population
    weighted risk" had no defensible candidate at all and is dropped
    entirely rather than guessed at (see module docstring point 4).
    """
    denom = 1 + (table["Prevalence"] * (table["Relative Risk"] - 1)).sum()
    return table["Prevalence"] * (table["Relative Risk"] - 1) / denom


def build_supplemental_table(
    window_label: str,
    model,
    x: pd.DataFrame,
    y: np.ndarray,
    prevalence: pd.DataFrame,
) -> pd.DataFrame:
    rr = profile_rr_table(model)
    obs = observed_profile_counts(x, y)

    table = rr.merge(prevalence, on=["Labels", "sex", "smoke", "preterm"])
    table = table.merge(obs, on=["sex", "smoke", "preterm"], how="left")
    table["count"] = table["count"].fillna(0)

    ref = table.loc[(table["sex"] == 0) & (table["smoke"] == 0) & (table["preterm"] == 0)].iloc[0]
    if ref["n_live_births"] == 0:
        raise ValueError(f"[{window_label}] Reference profile (female fullterm nonsmoking) "
                          f"has zero live births in the denominator — cannot anchor Absolute Risk.")
    baseline_rate = ref["count"] / ref["n_live_births"] * 100_000
    LOG.info("[%s] Reference-profile (female fullterm nonsmoking) empirical rate: "
              "%d cases / %d live births = %.2f per 100,000",
              window_label, int(ref["count"]), int(ref["n_live_births"]), baseline_rate)

    table["Absolute Risk (per 100,000 live births)"] = table["Relative Risk"] * baseline_rate
    table["Population attributable component"] = population_attributable_component(table)
    LOG.info("[%s] Overall population attributable fraction (sum of per-profile "
              "components): %.3f", window_label, table["Population attributable component"].sum())

    table = table.rename(columns={
        "RR_CI_low": "RR_CI_low_unused", "RR_CI_high": "RR_CI_high_unused",
    })
    table["Relative Risk (95% CI)"] = table.apply(
        lambda r: f"{r['Relative Risk']:.2f} ({r['RR_CI_low_unused']:.2f}-{r['RR_CI_high_unused']:.2f})",
        axis=1,
    )
    out = table[[
        "Labels", "Relative Risk (95% CI)", "RR_p",
        "Population attributable component",
        "Absolute Risk (per 100,000 live births)",
    ]].rename(columns={"RR_p": "p"})
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--exposure-dir", type=Path, default=Path("output_data/risk_exposure"))
    p.add_argument("--denominator", required=True, type=Path,
                    help="Population denominator (live births) file, e.g. "
                         "standardized/consolidated/denominator.parquet")
    p.add_argument("--outdir", type=Path, default=Path("."))

    p.add_argument("--infage-col", default="infage")
    p.add_argument("--combgest-col", default="combgest")
    p.add_argument("--sex-col", default="sex")
    p.add_argument("--smoking-col", default="cig_rec")
    p.add_argument("--pca-min", type=int, default=36)
    p.add_argument("--pca-max", type=int, default=90)
    p.add_argument("--peak1", type=int, default=43)
    p.add_argument("--peak2", type=int, default=47)
    p.add_argument("--peak3", type=int, default=None, help="Defaults to --pca-max.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    peak3 = args.peak3 if args.peak3 is not None else args.pca_max
    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.outdir / "supplemental_table23_run.log")],
    )

    for pth in (args.input, args.denominator):
        if not pth.exists():
            LOG.error("Not found: %s", pth)
            return 1

    LOG.info("NOTE: 'Population attributable component' uses the standard generalized-PAF "
             "formula (see module docstring point 4), not the original bespoke computation "
             "(unverifiable). 'Population weighted risk' is dropped entirely — no formula.")

    raw_df = read_any(args.input)
    case_df, _, trail = clean_case_series(
        raw_df, infage_col=args.infage_col, combgest_col=args.combgest_col,
        pca_min=args.pca_min, pca_max=args.pca_max, pca_max_inclusive=False,
        reproduce_original_bug=False,
    )
    trail.to_frame().to_csv(args.outdir / "audit_trail_supplemental_table23.csv", index=False)

    _, x1, x2, y1, y2, model1, model2 = fit_windowed_models(
        case_df, peak1=args.peak1, peak2=args.peak2, peak3=peak3,
        exposure_dir=args.exposure_dir,
        combgest_col=args.combgest_col, smoking_col=args.smoking_col, sex_col=args.sex_col,
    )

    prevalence = joint_prevalence(
        args.denominator, sex_col=args.sex_col,
        combgest_col=args.combgest_col, smoking_col=args.smoking_col,
    )
    prevalence.to_csv(args.outdir / "profile_prevalence.csv", index=False)

    table2 = build_supplemental_table("early window", model1, x1, y1, prevalence)
    table3 = build_supplemental_table("late window", model2, x2, y2, prevalence)

    table2.to_csv(args.outdir / "supplemental_table2_corrected.csv", index=False)
    table3.to_csv(args.outdir / "supplemental_table3_corrected.csv", index=False)

    LOG.info("Supplemental Table 2 (early window, corrected):\n%s", table2.to_string(index=False))
    LOG.info("Supplemental Table 3 (late window, corrected):\n%s", table3.to_string(index=False))
    LOG.info("DONE. See %s", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

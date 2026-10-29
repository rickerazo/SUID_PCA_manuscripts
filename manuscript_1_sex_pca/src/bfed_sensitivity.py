#!/usr/bin/env python3
"""
bfed_sensitivity.py -- Breastfeeding sensitivity analysis (manuscript Methods,
"Exposure and covariates"; Results, "Confirmatory windowed models").

Breastfeeding is not a covariate in the main models (it is recorded only at
discharge and is a postnatal, time-varying exposure). This script tests whether
breastfeeding status modifies the sex-specific trajectories by re-running the
ENTIRE pipeline with the case file filtered on BFED at the very beginning
(before any cleaning), once for BFED=Y and once for BFED=N, and comparing the
resulting models. BFED is NOT added as a covariate or stratification variable
inside the models.

CDC documentation: BFED, position 569, "Infant Being Breastfed";
Y = Yes, N = No, U = Unknown or not stated. U (and missing) cases belong to
neither subset; they are counted and reported.

For each subset (ALL = reproduction of the main analysis, Y, N):
  1. Filter raw case file on BFED (ALL: no filter).
  2. clean_case_series() with the main-analysis settings (PCA 36-90, exclusive upper bound).
  3. Windowed negative binomial models (early <43, late 47-90): full IRR tables.
  4. Full-curve NB GAM with sex-specific PCA smooths: Figure 3 panels rendered
     for the subset (reference stratum smoke=0, preterm=0).
  5. Observed weekly case counts by sex across PCA.
Then compares Y vs N:
  - every windowed-model coefficient side by side, with the ratio of IRRs
    (Y/N) and a Wald z-test on the difference in log IRR (independent subsets);
  - PCA distribution of cases, Y vs N, within each sex (medians, two-sample KS);
  - fitted log(male/female) GAM ratio across PCA, overlaid for ALL / Y / N.

OFFSETS. The model offset is the product of the marginal proportions
(smoking x sex x preterm) in the natality denominator. Two versions are
produced when --denominator is given:
  bfed     : subset models use proportions computed within denominator births
             with the same BFED value. This is the version reported in the
             manuscript (compare_windowed_bfed.csv).
  national : subset models use the same national pickles as the main analysis.
             Kept as a check; it gives the same sex results.
The ALL run always uses the national pickles and must reproduce the published
estimates (early Male IRR 1.14, late Male IRR 1.36); PASS/FAIL is logged.

USAGE
    python bfed_sensitivity.py \
        --input output_data/controls/matched_cases_cdc.xlsx \
        --exposure-dir output_data/risk_exposure \
        --denominator /path/to/denominator.parquet \
        --outdir bfed_sensitivity_out
Add --alpha-grid-size 4 --cv-folds 2 for a fast dry run.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

LOG = logging.getLogger("bfed_sensitivity")

EXPECTED_ALL = {  # published main-analysis estimates (run_windowed_nb_models.py)
    "early": (1.139275, 0.846135, 1.533973),
    "late": (1.357037, 1.293553, 1.423637),
}
COEFS = ["PCA (per week)", "Male (vs female)", "Maternal smoking", "Preterm birth",
         "Smoking x Preterm", "Sex x Preterm"]


# ---------------------------------------------------------------------------
# Offsets
# ---------------------------------------------------------------------------

def _proportions(s: pd.Series, a, b) -> dict:
    # Mirrors risk_exposure_fixed.compute_exposure(): share of each level among
    # ALL non-null values of the column (unknown codes stay in the denominator).
    vc = s.value_counts()
    tot = vc.sum()
    return {0: vc.get(a, 0) / tot, 1: vc.get(b, 0) / tot}


def build_bfed_exposure_dirs(denominator: Path, national_dir: Path, outdir: Path,
                             bfed_col: str) -> dict[str, Path] | None:
    try:
        import pyarrow.parquet as pq
        names = pq.ParquetFile(denominator).schema.names
    except Exception:
        names = list(pd.read_parquet(denominator).columns)
    hit = [c for c in names if c.lower() == bfed_col.lower()]
    if not hit:
        LOG.warning("Denominator has no %s column: BFED-specific offsets NOT available; "
                    "only the national-offset version will be produced.", bfed_col)
        return None
    col = hit[0]
    den = pd.read_parquet(denominator, columns=[col, "cig_rec", "sex", "combgest"])
    den = den.rename(columns={col: "bfed"})
    rows, dirs = [], {}
    for val in ("Y", "N"):
        d = den.loc[den["bfed"] == val].copy()
        known = d.loc[d["combgest"] < 99]
        pre = (known["combgest"] < 37).astype(int)
        dics = {
            "smoke_exposure": _proportions(d["cig_rec"], "N", "Y"),
            "sex_exposure": _proportions(d["sex"], "F", "M"),
            "term_exposure": _proportions(pre, 0, 1),
        }
        tgt = outdir / f"exposure_bfed_{val}"
        tgt.mkdir(parents=True, exist_ok=True)
        for name, dic in dics.items():
            with open(tgt / f"{name}.pkl", "wb") as f:
                pickle.dump(dic, f)
            rows.append({"bfed": val, "births": len(d), "exposure": name, "p0": dic[0], "p1": dic[1]})
        shutil.copy(national_dir / "bedshare_exposure.pkl", tgt / "bedshare_exposure.pkl")
        dirs[val] = tgt
    for name in ("smoke_exposure", "sex_exposure", "term_exposure"):
        with open(national_dir / f"{name}.pkl", "rb") as f:
            dic = pickle.load(f)
        rows.append({"bfed": "ALL(national)", "births": np.nan, "exposure": name, "p0": dic[0], "p1": dic[1]})
    tab = pd.DataFrame(rows)
    tab.to_csv(outdir / "offset_proportions.csv", index=False)
    LOG.info("Offset proportions by BFED:\n%s", tab.to_string(index=False))
    return dirs


# ---------------------------------------------------------------------------
# One full pipeline run on one subset
# ---------------------------------------------------------------------------

def run_subset(raw_df, label, exposure_dir, args, P, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    LOG.info("=" * 78)
    LOG.info("[%s] raw cases after BFED filter: %d (exposure dir: %s)", label, len(raw_df), exposure_dir)

    case_df, _, trail = P.clean_case_series(
        raw_df, infage_col=args.infage_col, combgest_col=args.combgest_col,
        pca_min=args.pca_min, pca_max=args.pca_max, pca_max_inclusive=False,
        reproduce_original_bug=False,
    )
    trail.to_frame().to_csv(outdir / "audit_trail.csv", index=False)

    # Observed weekly counts by sex (valid cig_rec only, as count_data() does).
    obs = case_df.loc[case_df[args.smoking_col].isin(["Y", "N"])]
    weekly = obs.groupby(["pca", args.sex_col]).size().unstack(fill_value=0)
    weekly.to_csv(outdir / "weekly_counts_by_sex.csv")
    sparse = weekly.loc[(weekly.min(axis=1) < 5)].index.tolist()
    LOG.info("[%s] analytic cases (valid cig_rec): %d; PCA weeks with <5 cases in either sex: %s",
             label, len(obs), sparse or "none")

    _, x1, x2, y1, y2, m1, m2 = P.fit_windowed_models(
        case_df, peak1=args.peak1, peak2=args.peak2, peak3=args.pca_max,
        exposure_dir=exposure_dir, combgest_col=args.combgest_col,
        smoking_col=args.smoking_col, sex_col=args.sex_col,
    )
    res = {"label": label, "n_analytic": len(obs), "cases": obs}
    for win, m, y, x in (("early", m1, y1, x1), ("late", m2, y2, x2)):
        conv = bool(m.mle_retvals.get("converged", False))
        t = P.irr_table(m, "pca", int(y.sum()))
        t.to_csv(outdir / f"irr_{win}.csv", index=False)
        LOG.info("[%s] %s window: %d cases, %d strata rows, converged=%s\n%s",
                 label, win, int(y.sum()), len(x), conv, t.to_string(index=False))
        # keep log-scale estimates for the Y vs N comparison
        se = m.bse
        res[win] = {
            "table": t, "converged": conv, "n": int(y.sum()),
            "params": m.params, "bse": se,
        }

    if not args.skip_gam:
        try:
            curves = P.run_nb_splines(
                case_df, var1="pca", exposure_dir=exposure_dir, outdir=outdir / "gam",
                pca_min=args.pca_min, pca_max=args.pca_max, peak_knot=args.peak_knot,
                a=args.a, b=args.b, smoke_ref=0, preterm_ref=0, nb_alpha=args.nb_alpha,
                alpha_grid_size=args.alpha_grid_size, cv_folds=args.cv_folds,
                combgest_col=args.combgest_col, smoking_col=args.smoking_col,
                sex_col=args.sex_col,
            )
            f = pd.DataFrame({"pca": curves["pca_grid_f"], "mu_f": curves["mu_f"]})
            mm = pd.DataFrame({"pca": curves["pca_grid_m"], "mu_m": curves["mu_m"]})
            c = f.merge(mm, on="pca").sort_values("pca")
            c["log_mf_ratio"] = np.log(c["mu_m"] / c["mu_f"])
            c.to_csv(outdir / "gam_fitted_ref_stratum.csv", index=False)
            res["gam"] = c
        except Exception as e:  # report, don't abort the other subsets
            LOG.error("[%s] GAM failed: %r", label, e)
    return res


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def compare(results: dict, outdir: Path, tag: str, args) -> None:
    Y, N, A = results["Y"], results["N"], results["ALL"]
    rows = []
    for win in ("early", "late"):
        for name in COEFS:
            r = {"window": win, "predictor": name}
            for lab, R in (("ALL", A), ("Y", Y), ("N", N)):
                t = R[win]["table"].set_index("Predictor")
                if name in t.index:
                    q = t.loc[name]
                    r[f"{lab}_IRR (95% CI)"] = f"{q.IRR:.2f} ({q.CI_low:.2f}\u2013{q.CI_high:.2f})"
                    r[f"{lab}_p"] = q.p
            # Wald test, independent subsets: log(IRR_Y) - log(IRR_N)
            key = {"PCA (per week)": "pca", "Male (vs female)": "sex", "Maternal smoking": "smoke",
                   "Preterm birth": "preterm", "Smoking x Preterm": "smoke*preterm",
                   "Sex x Preterm": "sex*preterm"}[name]
            bY, sY = Y[win]["params"].get(key), Y[win]["bse"].get(key)
            bN, sN = N[win]["params"].get(key), N[win]["bse"].get(key)
            if None not in (bY, sY, bN, sN):
                d, sd = bY - bN, float(np.hypot(sY, sN))
                r["ratio_Y_over_N (95% CI)"] = (f"{np.exp(d):.2f} ({np.exp(d - 1.96 * sd):.2f}"
                                                f"\u2013{np.exp(d + 1.96 * sd):.2f})")
                r["p_difference"] = 2 * stats.norm.sf(abs(d / sd))
            rows.append(r)
        for lab, R in (("ALL", A), ("Y", Y), ("N", N)):
            LOG.info("[%s] %s window n=%d converged=%s", lab, win, R[win]["n"], R[win]["converged"])
    cmp_ = pd.DataFrame(rows)
    cmp_.to_csv(outdir / f"compare_windowed_{tag}.csv", index=False)
    LOG.info("Windowed-model comparison (%s offsets):\n%s", tag, cmp_.to_string(index=False))

    # PCA distribution of cases, Y vs N, by sex
    drows = []
    for sex in ("F", "M"):
        py = Y["cases"].loc[Y["cases"][args.sex_col] == sex, "pca"]
        pn = N["cases"].loc[N["cases"][args.sex_col] == sex, "pca"]
        ks = stats.ks_2samp(py, pn)
        drows.append({"sex": sex, "n_Y": len(py), "n_N": len(pn),
                      "median_PCA_Y": py.median(), "median_PCA_N": pn.median(),
                      "IQR_Y": f"{py.quantile(.25):.0f}\u2013{py.quantile(.75):.0f}",
                      "IQR_N": f"{pn.quantile(.25):.0f}\u2013{pn.quantile(.75):.0f}",
                      "KS_D": ks.statistic, "KS_p": ks.pvalue})
    dist = pd.DataFrame(drows)
    dist.to_csv(outdir / "compare_pca_distribution.csv", index=False)
    LOG.info("Case PCA distribution, BFED=Y vs N:\n%s", dist.to_string(index=False))

    # Overlay: fitted log(M/F) across PCA
    if all("gam" in R for R in (A, Y, N)):
        fig, ax = plt.subplots(figsize=(7, 4))
        for lab, R, ls in (("All cases", A, "-"), ("BFED = Y", Y, "--"), ("BFED = N", N, ":")):
            g = R["gam"]
            ax.plot(g["pca"], g["log_mf_ratio"], ls, label=lab)
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvspan(43, 46.99, color="grey", alpha=0.15)
        ax.set_xlabel("Postconceptional age (weeks)")
        ax.set_ylabel("Fitted log(male / female)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / f"compare_gam_log_mf_{tag}.png", dpi=300)
        plt.close(fig)

    # Overlay: observed PCA distributions (share of cases per week) by sex
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, sex in zip(axes, ("F", "M")):
        for lab, R, ls in (("BFED = Y", Y, "-"), ("BFED = N", N, "--")):
            s = R["cases"].loc[R["cases"][args.sex_col] == sex, "pca"].value_counts(normalize=True).sort_index()
            ax.plot(s.index, 100 * s.values, ls, label=lab)
        ax.set_title("Female" if sex == "F" else "Male")
        ax.set_xlabel("Postconceptional age (weeks)")
    axes[0].set_ylabel("% of cases")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "compare_pca_distribution.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pipeline-dir", type=Path, default=Path(__file__).resolve().parent,
                   help="Folder with the R1 scripts; defaults to this script's own folder")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--exposure-dir", type=Path, default=Path("output_data/risk_exposure"))
    p.add_argument("--denominator", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=Path("bfed_sensitivity_out"))
    p.add_argument("--bfed-col", default="bfed")
    p.add_argument("--infage-col", default="infage")
    p.add_argument("--combgest-col", default="combgest")
    p.add_argument("--sex-col", default="sex")
    p.add_argument("--smoking-col", default="cig_rec")
    p.add_argument("--pca-min", type=int, default=36)
    p.add_argument("--pca-max", type=int, default=90)
    p.add_argument("--peak1", type=int, default=43)
    p.add_argument("--peak2", type=int, default=47)
    # GAM settings: defaults identical to generate_figure3_gam_curves.py
    p.add_argument("--peak-knot", type=int, default=46)
    p.add_argument("--a", type=int, default=10)
    p.add_argument("--b", type=int, default=10)
    p.add_argument("--nb-alpha", type=float, default=0.10)
    p.add_argument("--alpha-grid-size", type=int, default=20)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--skip-gam", action="store_true")
    args = p.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.outdir / "bfed_sensitivity.log")])

    sys.path.insert(0, str(args.pipeline_dir.resolve()))
    import types
    from audit_subject_breakdown import clean_case_series, read_any
    from run_windowed_nb_models import fit_windowed_models, irr_table
    from generate_figure3_gam_curves import run_nb_splines
    P = types.SimpleNamespace(clean_case_series=clean_case_series, fit_windowed_models=fit_windowed_models,
                              irr_table=irr_table, run_nb_splines=run_nb_splines)

    raw = read_any(args.input)
    LOG.info("Raw case file: %s", raw.shape)
    bf = [c for c in raw.columns if c.lower() == args.bfed_col.lower()]
    if not bf:
        LOG.error("Case file has no %s column", args.bfed_col)
        return 1
    bcol = bf[0]
    bvals = raw[bcol].astype("string").str.strip().str.upper()
    dist = (pd.crosstab(bvals.fillna("<missing>"), raw[args.sex_col], margins=True))
    dist.to_csv(args.outdir / "bfed_distribution_raw.csv")
    LOG.info("BFED distribution in the raw case file (by sex):\n%s", dist.to_string())

    subsets = {"ALL": raw, "Y": raw.loc[bvals == "Y"], "N": raw.loc[bvals == "N"]}

    offset_versions = {"national": {"Y": args.exposure_dir, "N": args.exposure_dir}}
    if args.denominator is not None:
        dirs = build_bfed_exposure_dirs(args.denominator, args.exposure_dir, args.outdir, args.bfed_col)
        if dirs:
            offset_versions["bfed"] = dirs

    all_res = run_subset(subsets["ALL"], "ALL", args.exposure_dir, args, P, args.outdir / "ALL")
    for win, (irr, lo, hi) in EXPECTED_ALL.items():
        q = all_res[win]["table"].set_index("Predictor").loc["Male (vs female)"]
        ok = np.allclose([q.IRR, q.CI_low, q.CI_high], [irr, lo, hi], atol=5e-4)
        LOG.info("Reproduction check, ALL %s Male: %.4f (%.4f-%.4f) vs %.4f (%.4f-%.4f) -> %s",
                 win, q.IRR, q.CI_low, q.CI_high, irr, lo, hi, "PASS" if ok else "FAIL")

    for tag, dirs in offset_versions.items():
        res = {"ALL": all_res}
        for val in ("Y", "N"):
            res[val] = run_subset(subsets[val], f"{val}-{tag}", dirs[val], args, P,
                                  args.outdir / f"{val}_{tag}")
        compare(res, args.outdir, tag, args)

    LOG.info("DONE. See %s", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

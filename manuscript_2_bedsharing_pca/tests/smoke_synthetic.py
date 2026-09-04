#!/usr/bin/env python3
"""
smoke_synthetic.py

Fabricates a tiny NCFRP-shaped dataset, exposure pickles, and (if pyarrow is
available) a denominator parquet, then exercises the audited pipeline in
manuscript_2_bedsharing_pca/src/ncfrp_manuscript_v5.py WITHOUT any real data.

It is a smoke test, not a correctness test: it checks that the code paths run,
the case-selection audit trail reconciles, the negative binomial and GAM models
fit, and the expected output files are written.

    python manuscript_2_bedsharing_pca/tests/smoke_synthetic.py
"""

from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import ncfrp_manuscript_v5 as pipe  # noqa: E402
import compare_time_axes as cta  # noqa: E402
import gam_bedshare_pca_varycoef as varycoef  # noqa: E402


def _make_cases(n: int, seed: int = 0) -> pd.DataFrame:
    """Column names/casing match a real NCFRP_SCH_pop.csv header (the loader
    lowercases); INF3gestunk / the *tricigunk flags are present too."""
    rng = np.random.default_rng(seed)
    gest = rng.integers(24, 43, n)                       # gestational weeks
    postnatal_wk = rng.integers(0, 40, n)               # postnatal weeks
    df = pd.DataFrame({
        "Unnamed: 0": np.arange(n),                       # stray index col, must be dropped
        "INFageDays": (postnatal_wk * 7 + rng.integers(0, 7, n)).astype(float),
        "INFagecat": rng.choice([2, 3], n, p=[0.4, 0.6]),  # 2=Months, 3=Days (age-unit split)
        "INF3gestage": gest.astype(float),
        "INF3gestunk": 0,
        "INF3biwt": np.clip(rng.normal(2400 + 40 * (gest - 24), 400), 700, 5200),
        "Smoker": rng.choice([1, 2], n, p=[0.45, 0.55]),   # 1=Y, 2=N, 9=unknown
        "INFsex": rng.choice([1, 2], n, p=[0.52, 0.48]),   # 1=M, 2=F, 9=unknown
        "Bed_sharing": rng.choice([1, 2], n, p=[0.4, 0.6]),  # 1=Y, 2=N, 9=unknown
        "INFdodyear": rng.integers(2004, 2020, n).astype(float),
    })
    # inject junk the loader must exclude
    junk = df.sample(frac=0.08, random_state=seed).index
    df.loc[junk[: len(junk) // 4], "INF3gestage"] = 99             # implausible gestage
    df.loc[junk[len(junk) // 4: len(junk) // 2], "INF3gestunk"] = 1  # flagged-unknown gestage
    df.loc[junk[len(junk) // 2: 3 * len(junk) // 4], "Smoker"] = 9   # invalid smoking
    df.loc[junk[3 * len(junk) // 4:], "INFageDays"] = np.nan         # missing age
    return df


def _write_exposures(exposure_dir: Path) -> None:
    exposure_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "smoke_exposure": {0: 0.934, 1: 0.066},
        "sex_exposure": {0: 0.488, 1: 0.512},
        "term_exposure": {0: 0.90, 1: 0.10},
        "bedshare_exposure": {0: 0.72, 1: 0.28},
        # joint P(smoke, preterm) keyed by (smoke, preterm) tuples (G2)
        "joint_smoke_preterm_exposure": {(0, 0): 0.85, (0, 1): 0.083,
                                         (1, 0): 0.05, (1, 1): 0.017},
    }
    for name, d in payload.items():
        with open(exposure_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(d, f)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def _make_synthetic_stratum_table(seed: int = 7) -> pd.DataFrame:
    """
    One row per (pca, bed_share, smoke, preterm) stratum-week, matching
    ncfrp_manuscript_v5.make_ncfrp_input's output shape, with a KNOWN
    varying-coefficient truth:

        log(mu) = a0 + a1*bed_share + a2*smoke + a3*preterm
                  + f_main(pca) + f_int(pca) * bed_share

    f_int is a smooth, bounded bump concentrated at low PCA and tapering to
    ~0 by high PCA - mirroring the manuscript's "bed-sharing risk
    concentrates earlier in development" pattern. Counts are drawn from a
    reparameterized negative binomial. bed_share=1 rows are deliberately made
    SPARSER at high PCA (mirroring the real trigger for the retired
    construction's degeneracy), so this fixture actually exercises the fix
    rather than an easy, uniformly-dense case.
    """
    rng = np.random.default_rng(seed)
    pca_weeks = np.arange(36, 76)
    a0, a1, a2, a3 = 0.5, 0.3, 0.2, 0.1
    alpha_true = 0.3
    rows = []
    for pca in pca_weeks:
        p_bed_present = max(0.15, 1.0 - (pca - 36) / 60.0)  # thins out at high pca
        f_main = 0.02 * (pca - 55) - 0.001 * (pca - 55) ** 2
        f_int = 1.2 * np.exp(-((pca - 40) ** 2) / (2 * 8.0 ** 2))
        for smoke in (0, 1):
            for preterm in (0, 1):
                for bed in (0, 1):
                    if bed == 1 and rng.random() > p_bed_present:
                        continue
                    log_mu = a0 + a1 * bed + a2 * smoke + a3 * preterm + f_main + f_int * bed
                    mu = np.exp(log_mu) * 8.0
                    size = 1.0 / alpha_true
                    p = size / (size + mu)
                    count = rng.negative_binomial(size, p)
                    if count < 1:
                        continue  # count_ncfrp_data never emits zero-count rows
                    rows.append({"pca": float(pca), "bed_share": bed, "smoke": smoke,
                                "preterm": preterm, "count": float(count),
                                "log_exposure": 0.0})
    return pd.DataFrame(rows)


def test_varycoef_replaces_degenerate_construction() -> None:
    """A3.27/A3.28: confirms (a) the OLD bed_share*pca-then-spline construction
    still degenerates on a fixture built to trigger it (regression-of-the-
    regression - if this stops triggering, strengthen the fixture, not the
    production code), and (b) the NEW varying-coefficient construction
    recovers sane, bounded estimates and the injected early>late shape."""
    input_df = _make_synthetic_stratum_table(seed=7)

    # --- OLD construction, reproduced inline (the block itself was retired
    # from run_piecewise_and_gam; fit_nb_gam/BSplines/GLMGam it relied on are
    # unchanged and still importable).
    x_linear, x_smooth, y, offset, _ = pipe.prepare_curve_data_ncfrp(input_df, "pca")
    xs_old = np.column_stack([x_smooth[:, 0],
                              x_smooth[:, 0] * x_linear["bed_share"].to_numpy()])
    old_res, _ = pipe.fit_nb_gam(x_linear, xs_old, y, offset)
    old_irr = pipe._coef_table(old_res)
    # The OLD interaction smooth's knots are placed from the zero-inflated
    # bed_share*pca product, so several of its basis functions (the ones
    # nearest the phantom zero cluster) have literally zero data support and
    # collapse to ~0 coef/SE - a deterministic linear-algebra fact independent
    # of statsmodels version or optimizer path, unlike the exact SE magnitude
    # of the real-data blow-up (which also depends on the joint offset and
    # CV alpha selection). This is the reliable, environment-robust signature
    # to check for a "does this fixture still trigger the bug" regression
    # test - if it stops triggering, strengthen the fixture, not this check.
    old_int_terms = old_irr.filter(regex=r"^x1_s", axis=0)
    n_collapsed = int((old_int_terms["coef"].abs() < 1e-6).sum())
    _check(n_collapsed >= 3,
           f"fixture reproduces the OLD construction's known degeneracy: "
           f"{n_collapsed} interaction-basis terms collapsed to ~0 coef "
           f"(no real data support at those knots)")

    # --- NEW construction
    X, y2, offset2, design_info = varycoef.build_varying_coef_design(input_df)
    new_res = varycoef.fit_bedshare_pca_varycoef(X, y2, offset2, label="synthetic")
    new_irr = pipe._coef_table(new_res)
    _check(new_irr["std_err"].max() < 5,
           f"NEW construction: all std errs well-bounded (max={new_irr['std_err'].max():.3g})")
    _check((new_irr["IRR"] > 0.01).all() and (new_irr["IRR"] < 100).all(),
           "NEW construction: bounded, plausible IRRs")
    _check((new_irr["IRR_ci_low"] <= new_irr["IRR"] + 1e-9).all()
           and (new_irr["IRR"] <= new_irr["IRR_ci_high"] + 1e-9).all(),
           "NEW construction: every CI brackets its point estimate")

    mu_bs_40, _, _ = varycoef.predict_with_delta_ci(
        new_res, design_info, np.array([40.0]), bed_share=1, smoke=0, preterm=0, offset_value=0.0)
    mu_ns_40, _, _ = varycoef.predict_with_delta_ci(
        new_res, design_info, np.array([40.0]), bed_share=0, smoke=0, preterm=0, offset_value=0.0)
    mu_bs_70, _, _ = varycoef.predict_with_delta_ci(
        new_res, design_info, np.array([70.0]), bed_share=1, smoke=0, preterm=0, offset_value=0.0)
    mu_ns_70, _, _ = varycoef.predict_with_delta_ci(
        new_res, design_info, np.array([70.0]), bed_share=0, smoke=0, preterm=0, offset_value=0.0)
    ratio_40, ratio_70 = mu_bs_40[0] / mu_ns_40[0], mu_bs_70[0] / mu_ns_70[0]
    _check(ratio_40 > ratio_70,
           f"NEW construction recovers the injected early>late bed-share effect "
           f"shape (ratio@pca40={ratio_40:.2f} > ratio@pca70={ratio_70:.2f})")


def test_check_design_rank() -> None:
    rng = np.random.default_rng(0)
    good = rng.normal(size=(200, 5))
    pipe.check_design_rank(good, [f"c{i}" for i in range(5)], "good-design")
    print("  ok  check_design_rank accepts a well-conditioned design")

    bad = good.copy()
    bad[:, 4] = bad[:, 0]  # duplicated column -> rank deficient
    try:
        pipe.check_design_rank(bad, [f"c{i}" for i in range(5)], "bad-design")
        raise AssertionError("check_design_rank did not raise on a rank-deficient design")
    except ValueError:
        print("  ok  check_design_rank raises on a rank-deficient design")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ncfrp_smoke_"))
    print(f"workdir: {tmp}")
    (tmp / "standardized").mkdir(parents=True, exist_ok=True)

    cases = _make_cases(9000)
    data_path = tmp / "standardized" / "NCFRP_SCH_LR_BW_pop.csv"
    cases.to_csv(data_path, index=False)
    _write_exposures(tmp / "output_data" / "risk_exposure")

    # ---- 1. audited loader + audit trail reconciliation -------------------
    df, audit = pipe.load_ncfrp_data(data_path, pca_min=36, pca_max=90,
                                     audit_csv=tmp / "output_data" / "ncfrp_flow_audit.csv")
    _check(len(df) > 200, f"loader kept a usable cohort (n={len(df)})")
    # every step's n_after == previous n_after - n_removed  (recode-only steps: n_removed==0)
    steps = audit.to_dict("records") if hasattr(audit, "to_dict") else audit
    recon = all(r["n_after"] == r["n_before"] - r["n_removed"] for r in audit.to_dict("records"))
    _check(recon, "audit trail reconciles at every step")
    _check(df["inf3gestage"].max() <= pipe.POSTTERM_CAP,
           "post-term capped, implausible gestage excluded")
    _check(df["infageweeks"].min() > pipe.FIRST_WEEK_CUT, "first-week exclusion applied")

    # ---- 2. two-population split ----------------------------------------
    win = pipe.split_analytic_windows(df, peak1=43, peak2=47)
    _check(len(win["early"]) + len(win["late"]) + len(win["peak"]) == len(df),
           "early + late + peak == full analytic N")

    # ---- 3. counts + offset + interaction design ----------------------
    inp = pipe.make_ncfrp_input("pca", df,
                                exposure_dir=tmp / "output_data" / "risk_exposure",
                                min_count=1)
    _check({"bed_share:smoke", "bed_share:preterm"}.issubset(
        set(pipe._add_interactions(inp).columns)), "interaction columns build")
    x1, x2, y1, y2, o1, o2, early, late = pipe.prepare_piecewise_data(inp, "pca", 43, 47, 90)
    _check(list(x1.columns)[1:] == pipe.DESIGN_COLS, "design matrix has the prespecified terms")

    # ---- 4. models fit ------------------------------------------------
    from statsmodels.discrete.discrete_model import NegativeBinomial
    m_early = NegativeBinomial(y1, x1, offset=o1).fit(maxiter=200, disp=False)
    _check(np.isfinite(m_early.llf), "early-window NB model fits")
    xl, xs, y, off, curve = pipe.prepare_curve_data_ncfrp(inp, "pca")
    gam_res, knots = pipe.fit_nb_gam(xl, xs, y, off)
    _check(len(knots["knots_main"]) >= 2, "GAM fits with >=2 interior knots")

    # ---- 5. single-source IRR table + CI-bracket assertion -----------
    irr = pipe.save_nb_model_outputs(m_early, str(tmp / "output_data" / "ncfrp" / "smoke_early"))
    _check((irr["IRR_ci_low"] <= irr["IRR"] + 1e-9).all()
           and (irr["IRR"] <= irr["IRR_ci_high"] + 1e-9).all(),
           "every reported CI brackets its point estimate")

    # ---- 6. full main() if parquet I/O is available -----------------
    try:
        import pyarrow  # noqa: F401
        den = pd.DataFrame({"combgest": np.random.default_rng(1).integers(24, 43, 50000)})
        (tmp / "standardized" / "consolidated").mkdir(parents=True, exist_ok=True)
        den.to_parquet(tmp / "standardized" / "consolidated" / "denominator.parquet")
        rc = pipe.main(["--data-path", str(data_path),
                        "--exposure-dir", str(tmp / "output_data" / "risk_exposure"),
                        "--outdir", str(tmp),
                        "--no-sex-strata", "--min-count", "1",
                        "--pca-min", "36", "--pca-max", "90"])
        _check(rc == 0, "ncfrp_manuscript_v5.main() returns 0")
        for rel in ["output_data/ncfrp_flow_audit.csv",
                    "output_data/ncfrp/two_population_Ns.csv",
                    "output_data/ncfrp/all_early_irr.csv",
                    "output_data/ncfrp/all_late_irr.csv",
                    "output_data/ncfrp/all_gam_exp.csv",
                    "output_figures/ncfrp/Figure2_ALL.png",
                    "output_figures/ncfrp/Figure4_ALL.png",
                    "output_figures/ncfrp/gam/Figure3_ALL.png"]:
            _check((tmp / rel).exists(), f"wrote {rel}")

        # lab-meeting add-ons: the model-based ones fail soft, but the
        # deterministic descriptive table must be written every time.
        _check((tmp / "output_data/ncfrp/stratified"
                / "under1mo_by_smoke_bedshare.csv").exists(),
               "wrote the <1-month smoke x bed-sharing table")

        rc2 = cta.main(["--data-path", str(data_path),
                        "--exposure-dir", str(tmp / "output_data" / "risk_exposure"),
                        "--pca-min", "36", "--pca-max", "90",
                        "--outdir", str(tmp / "time_axes")])
        _check(rc2 == 0, "compare_time_axes.main() returns 0")
        for rel in ["time_axes/reporting_artifact/reporting_artifact_summary.txt",
                    "time_axes/reporting_artifact/periodogram_chrono.csv",
                    "time_axes/reporting_artifact/spikes_by_age_unit.csv"]:
            _check((tmp / rel).exists(), f"wrote {rel}")
    except ImportError:
        print("  skip  full main() (pyarrow not installed) - core pipeline still verified")

    # ---- 7. bed_share x PCA varying-coefficient replacement (A3.27/A3.28) --
    test_check_design_rank()
    test_varycoef_replaces_degenerate_construction()

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

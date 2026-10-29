#!/usr/bin/env python3
"""
bfed_sensitivity.py -- Breastfeeding sensitivity analysis for the bed-sharing
manuscript (A/B design, mirroring manuscript_1_sex_pca/src/bfed_sensitivity.py).

Breastfeeding is not a covariate in the main models. This script tests whether
the bed-sharing results depend on breastfeeding status by re-running the ENTIRE
pipeline (ncfrp_manuscript_v5.py, same settings as the primary run) with the
case file filtered on the NCFRP breastfeeding field BEFORE any cleaning:

    group Y   INF3breast == 1   infant ever breastfed
    group N   INF3breast == 2   never breastfed
    group U   anything else     unknown / not recorded (code 9 or blank)
    group K   Y or N            known status (for the K vs U missingness check)
    group ALL no filter         must reproduce the primary run (PASS/FAIL logged)

Breastfeeding is never added as a covariate or stratum inside the models.

It also reports the breastfeeding missingness the manuscript must disclose:
share of the ANALYTIC cohort with unknown status, overall and by bed sharing,
maternal smoking, preterm birth, sex, developmental window, and death year
(bfed_missingness.csv/.md).

OFFSETS. Group Y and N models are fit twice:
    bfed      offsets built from births/respondents with the same breastfeeding
              status: joint P(smoke, preterm) from denominator rows with
              bfed == 'Y'/'N', bed-sharing prevalence from PRAMS respondents with
              brstfed == 1/2 (1 = ever breastfed, 2 = never; the file is recoded,
              the ~82% 'yes' share is checked against the denominator's 82%).
              This is the version to report.
    national  the primary run's national offsets (check only).
Groups U, K and ALL always use the national offsets.

COMPARISON (bfed_comparison_<tag>.csv/.md): every windowed-model term
(PCA/week, bed sharing, smoking, preterm, the two product terms) and the GAM
main effects, side by side for ALL / Y / N / U / K, with the Y/N ratio of IRRs
and a Wald z test on the difference in log IRR (independent subsets), and the
same K/U contrast. Plus the PCA distribution of cases Y vs N within bed-sharing
status (medians, two-sample KS) and an overlay figure.

USAGE (from the work dir, after run_all.sh has produced output_data/risk_exposure):
    python src/bfed_sensitivity.py \
        --data-path standardized/NCFRP_SCH_LR_BW_pop.csv \
        --denominator standardized/consolidated/denominator.parquet \
        --prams standardized/PRAMS/prams.parquet \
        --national-exposure-dir output_data/risk_exposure \
        --outdir out_bfed \
        --pca-min 36 --pca-max 90 --year-min 2008 --year-max 2019 --first-week-cut 1
Add --national-offsets-only to skip the bfed-offset runs; --skip-runs to only
rebuild the comparison from existing run_<group> folders.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ncfrp_manuscript_v5 as pipe            # noqa: E402
import risk_exposure_joint as rej             # noqa: E402

LOG = logging.getLogger("bfed_sensitivity")

# primary-run values the ALL group must reproduce (AJE/v3, joint offsets)
EXPECTED_ALL = {"early": 6.448, "late": 2.097}
TERMS = ["pca", "bed_share", "smoke", "preterm", "bed_share:smoke", "bed_share:preterm"]
LABELS = {"pca": "PCA (per week)", "bed_share": "Bed sharing", "smoke": "Maternal smoking",
          "preterm": "Preterm birth", "bed_share:smoke": "Bed sharing x smoking",
          "bed_share:preterm": "Bed sharing x preterm birth"}


# ---------------------------------------------------------------------------
# Breastfeeding groups and missingness
# ---------------------------------------------------------------------------

def bfed_group(series: pd.Series, yes_code: float, no_code: float) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce")
    g = pd.Series("U", index=series.index, dtype=object)
    g[v == yes_code] = "Y"
    g[v == no_code] = "N"
    return g


def cohort_group(cohort: pd.DataFrame, args) -> pd.Series:
    """Breastfeeding group from the loader output itself. load_ncfrp_data()
    lowercases column names and resets the index, so never join on index."""
    col = [c for c in cohort.columns if c.lower() == args.bfed_col.lower()]
    if not col:
        raise KeyError(f"{args.bfed_col} not carried through load_ncfrp_data()")
    return bfed_group(cohort[col[0]], args.yes_code, args.no_code)


def missingness_report(raw: pd.DataFrame, cohort: pd.DataFrame, group: pd.Series,
                       args, outdir: Path) -> pd.DataFrame:
    """Share of the ANALYTIC cohort by breastfeeding group, overall and by strata.
    `group` must be aligned to `cohort` (use cohort_group)."""
    c = cohort.copy()
    c["bfed"] = group.values
    c["window"] = np.where(c["pca"] < args.peak1, "early (<43)",
                           np.where(c["pca"] >= args.peak2, "late (>=47)", "peak (43-46)"))
    ycol = [k for k in c.columns if k.lower() == "infdodyear"]
    if ycol:
        c["death_year"] = c[ycol[0]].astype(int)
    strata = [("overall", None), ("bed_share", "bed_share"), ("maternal_smoking", "cig_rec"),
              ("preterm", "preterm"), ("sex", "sex"), ("window", "window")]
    if "death_year" in c:
        strata.append(("death_year", "death_year"))
    rows = []
    for name, col in strata:
        groups = [("all", c)] if col is None else [(str(k), g) for k, g in c.groupby(col)]
        for level, g in groups:
            n = len(g); vc = g["bfed"].value_counts()
            rows.append({"stratum": name, "level": level, "n": n,
                         "n_Y": int(vc.get("Y", 0)), "n_N": int(vc.get("N", 0)), "n_U": int(vc.get("U", 0)),
                         "pct_Y": 100 * vc.get("Y", 0) / n, "pct_N": 100 * vc.get("N", 0) / n,
                         "pct_unknown": 100 * vc.get("U", 0) / n})
    tab = pd.DataFrame(rows)
    tab.to_csv(outdir / "bfed_missingness.csv", index=False)
    with open(outdir / "bfed_missingness.md", "w") as f:
        f.write("# Breastfeeding status (NCFRP INF3breast) in the analytic cohort\n\n")
        f.write(tab.to_markdown(index=False, floatfmt=".1f") + "\n")
    LOG.info("Breastfeeding status in the analytic cohort:\n%s", tab.to_string(index=False))
    return tab


# ---------------------------------------------------------------------------
# Offsets by breastfeeding status
# ---------------------------------------------------------------------------

def build_bfed_exposure_dirs(args, outdir: Path) -> dict[str, Path]:
    den = pd.read_parquet(args.denominator, columns=["bfed", "cig_rec", "combgest", "sex"])
    prams = pd.read_parquet(args.prams, columns=["brstfed", "sleepown_raw"] +
                            ([args.prams_weight_col] if args.prams_weight_col else []))
    LOG.info("denominator bfed codes: %s", den["bfed"].value_counts(dropna=False).to_dict())
    LOG.info("PRAMS brstfed codes: %s", prams["brstfed"].value_counts(dropna=False).to_dict())
    known = den["bfed"].isin(["Y", "N"])
    p_den = float((den.loc[known, "bfed"] == "Y").mean())
    pk = prams["brstfed"].isin([args.prams_yes, args.prams_no])
    p_pr = float((prams.loc[pk, "brstfed"] == args.prams_yes).mean())
    LOG.info("ever-breastfed share: denominator %.3f, PRAMS (code %s = yes) %.3f", p_den, args.prams_yes, p_pr)
    if abs(p_den - p_pr) > 0.10:
        LOG.warning("PRAMS 'yes' share differs from the denominator by >10 points; check --prams-yes/--prams-no coding")
    rows, dirs = [], {}
    for g, den_code, pr_code in (("Y", "Y", args.prams_yes), ("N", "N", args.prams_no)):
        d = den.loc[den["bfed"] == den_code].copy()
        d["preterm"] = rej.compute_preterm(d)
        smoke = rej.compute_exposure(d.loc[d["cig_rec"].isin(["Y", "N"])], "cig_rec", "N", "Y")
        sex = rej.compute_exposure(d.loc[d["sex"].isin(["F", "M"])], "sex", "F", "M")
        term = rej.compute_exposure(d.dropna(subset=["preterm"]), "preterm", 0, 1)
        joint, ctx = rej.compute_joint_smoke_preterm(d)
        pr = prams.loc[prams["brstfed"] == pr_code].copy()
        pr["bedshare"] = rej.compute_bed_share(pr)
        if args.prams_weight_col:
            w = pr[args.prams_weight_col].astype(float)
            p_bs = float(w[pr["bedshare"] == 1].sum() / w.sum()); bedshare = {0: 1 - p_bs, 1: p_bs}
        else:
            bedshare = rej.compute_exposure(pr, "bedshare", 0, 1)
        tgt = outdir / f"exposure_bfed_{g}"; tgt.mkdir(parents=True, exist_ok=True)
        for name, dic in (("smoke_exposure", smoke), ("sex_exposure", sex), ("term_exposure", term),
                          ("bedshare_exposure", bedshare)):
            rej._dump(dic, tgt / f"{name}.pkl"); pd.DataFrame([dic]).to_csv(tgt / f"{name}.csv", index=False)
            rows.append({"group": g, "births": len(d), "prams_n": len(pr), "exposure": name, "p0": dic[0], "p1": dic[1]})
        rej._dump(joint, tgt / "joint_smoke_preterm_exposure.pkl")
        pd.DataFrame([{f"smoke{s}_preterm{pt}": v for (s, pt), v in joint.items()}]).to_csv(
            tgt / "joint_smoke_preterm_exposure.csv", index=False)
        for (s, pt), v in sorted(joint.items()):
            rows.append({"group": g, "births": len(d), "prams_n": len(pr), "exposure": f"joint smoke{s}_preterm{pt}", "p0": np.nan, "p1": v})
        dirs[g] = tgt
    # national, for reference
    for name in ("smoke_exposure", "term_exposure", "bedshare_exposure"):
        with open(Path(args.national_exposure_dir) / f"{name}.pkl", "rb") as f:
            dic = pickle.load(f)
        rows.append({"group": "ALL (national)", "births": np.nan, "prams_n": np.nan, "exposure": name, "p0": dic[0], "p1": dic[1]})
    tab = pd.DataFrame(rows); tab.to_csv(outdir / "offset_proportions.csv", index=False)
    LOG.info("Offset proportions by breastfeeding group:\n%s", tab.to_string(index=False))
    return dirs


# ---------------------------------------------------------------------------
# Pipeline runs
# ---------------------------------------------------------------------------

def run_pipeline(case_csv: Path, exposure_dir: Path, run_dir: Path, args) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "ncfrp_manuscript_v5.py"),
           "--data-path", str(case_csv.resolve()), "--exposure-dir", str(exposure_dir.resolve()),
           "--outdir", str(run_dir.resolve()),
           "--pca-min", str(args.pca_min), "--pca-max", str(args.pca_max),
           "--peak1", str(args.peak1), "--peak2", str(args.peak2),
           "--min-count", str(args.min_count), "--first-week-cut", str(args.first_week_cut),
           "--offset-mode", "joint", "--gam-alpha-grid", str(args.gam_alpha_grid),
           "--no-sex-strata", "--no-extra-strata"]
    if args.year_min is not None: cmd += ["--year-min", str(args.year_min)]
    if args.year_max is not None: cmd += ["--year-max", str(args.year_max)]
    LOG.info("RUN %s", " ".join(cmd))
    with open(run_dir / "pipeline.log", "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE.parent))
    if rc != 0:
        LOG.error("pipeline exited %d for %s (see pipeline.log); using whatever outputs it wrote "
                  "(the windowed models are written before the GAM)", rc, run_dir)


def read_run(run_dir: Path) -> dict:
    d = run_dir / "output_data" / "ncfrp"
    out = {}
    for win in ("early", "late"):
        p = d / f"all_{win}_irr.csv"
        out[win] = pd.read_csv(p).set_index("term") if p.exists() else None
    p = d / "all_gam_exp.csv"
    out["gam"] = pd.read_csv(p).set_index("term") if p.exists() else None
    p = run_dir / "output_data" / "ncfrp" / "two_population_Ns.csv"
    out["Ns"] = pd.read_csv(p).iloc[0].to_dict() if p.exists() else {}
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def fmt(row) -> str:
    return f"{row['IRR']:.2f} ({row['IRR_ci_low']:.2f}-{row['IRR_ci_high']:.2f})"


def wald(a, b):
    d, sd = a["coef"] - b["coef"], float(np.hypot(a["std_err"], b["std_err"]))
    return np.exp(d), np.exp(d - 1.96 * sd), np.exp(d + 1.96 * sd), 2 * stats.norm.sf(abs(d / sd))


def compare(runs: dict[str, dict], outdir: Path, tag: str) -> pd.DataFrame:
    rows = []
    for win, key in (("early", "early"), ("late", "late"), ("full-curve GAM", "gam")):
        for term in TERMS:
            r = {"model": win, "term": LABELS[term]}
            tabs = {g: R[key] for g, R in runs.items() if R.get(key) is not None}
            for g in ("ALL", "Y", "N", "K", "U"):
                t = tabs.get(g)
                r[f"{g} IRR (95% CI)"] = fmt(t.loc[term]) if (t is not None and term in t.index) else ""
            for a, b, name in (("Y", "N", "Y/N"), ("K", "U", "known/unknown")):
                ta, tb = tabs.get(a), tabs.get(b)
                if ta is not None and tb is not None and term in ta.index and term in tb.index:
                    ratio, lo, hi, p = wald(ta.loc[term], tb.loc[term])
                    r[f"ratio {name} (95% CI)"] = f"{ratio:.2f} ({lo:.2f}-{hi:.2f})"; r[f"p diff {name}"] = p
            if any(v for k, v in r.items() if k.endswith("(95% CI)")):
                rows.append(r)
    cmp_ = pd.DataFrame(rows)
    ns = {g: R.get("Ns", {}) for g, R in runs.items()}
    nrow = pd.DataFrame([{"model": "cohort", "term": k, **{f"{g} IRR (95% CI)": f"{int(ns[g].get(key, 0)):,}" if ns.get(g) else "" for g in ("ALL", "Y", "N", "K", "U")}}
                         for k, key in (("Analytic N", "full_analytic_N"), ("Early-window n", "early_n"), ("Late-window n", "late_n"))])
    cmp_ = pd.concat([cmp_, nrow], ignore_index=True)
    cmp_.to_csv(outdir / f"bfed_comparison_{tag}.csv", index=False)
    with open(outdir / f"bfed_comparison_{tag}.md", "w") as f:
        f.write(f"# Breastfeeding sensitivity, {tag} offsets: windowed and GAM estimates by breastfeeding group\n\n")
        f.write(cmp_.to_markdown(index=False) + "\n\n")
        f.write("Y = ever breastfed, N = never, U = unknown, K = known (Y or N), ALL = primary cohort. "
                "Ratio = IRR_a / IRR_b with a Wald test on the difference in log IRR (independent subsets).\n")
    LOG.info("Comparison (%s offsets):\n%s", tag, cmp_.to_string(index=False))
    return cmp_


def pca_distribution(cohort: pd.DataFrame, group: pd.Series, outdir: Path) -> None:
    c = cohort.copy(); c["bfed"] = group.values
    rows = []
    for bs in ("Y", "N"):
        py = c.loc[(c["bed_share"] == bs) & (c["bfed"] == "Y"), "pca"]
        pn = c.loc[(c["bed_share"] == bs) & (c["bfed"] == "N"), "pca"]
        if len(py) and len(pn):
            ks = stats.ks_2samp(py, pn)
            rows.append({"bed_sharing": bs, "n_breastfed": len(py), "n_not": len(pn),
                         "median_PCA_breastfed": py.median(), "median_PCA_not": pn.median(),
                         "IQR_breastfed": f"{py.quantile(.25):.0f}-{py.quantile(.75):.0f}",
                         "IQR_not": f"{pn.quantile(.25):.0f}-{pn.quantile(.75):.0f}",
                         "KS_D": ks.statistic, "KS_p": ks.pvalue})
    dist = pd.DataFrame(rows); dist.to_csv(outdir / "bfed_pca_distribution.csv", index=False)
    LOG.info("PCA distribution of cases, breastfed vs not, by bed sharing:\n%s", dist.to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, bs in zip(axes, ("Y", "N")):
        for lab, g, ls in (("Breastfed", "Y", "-"), ("Not breastfed", "N", "--"), ("Unknown", "U", ":")):
            s = c.loc[(c["bed_share"] == bs) & (c["bfed"] == g), "pca"].value_counts(normalize=True).sort_index()
            if len(s): ax.plot(s.index, 100 * s.values, ls, label=lab)
        ax.set_title("Bed sharing" if bs == "Y" else "Not bed sharing"); ax.set_xlabel("Postconceptional age (weeks)")
    axes[0].set_ylabel("% of cases"); axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(outdir / "bfed_pca_distribution.png", dpi=300); plt.close(fig)


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--denominator", type=Path, required=True)
    p.add_argument("--prams", type=Path, required=True)
    p.add_argument("--national-exposure-dir", type=Path, default=Path("output_data/risk_exposure"))
    p.add_argument("--outdir", type=Path, default=Path("out_bfed"))
    p.add_argument("--bfed-col", default="INF3breast")
    p.add_argument("--yes-code", type=float, default=1); p.add_argument("--no-code", type=float, default=2)
    p.add_argument("--prams-yes", type=float, default=1); p.add_argument("--prams-no", type=float, default=2)
    p.add_argument("--prams-weight-col", default=None)
    p.add_argument("--pca-min", type=int, default=pipe.PCA_MIN); p.add_argument("--pca-max", type=int, default=pipe.PCA_MAX)
    p.add_argument("--peak1", type=int, default=pipe.PEAK1); p.add_argument("--peak2", type=int, default=pipe.PEAK2)
    p.add_argument("--min-count", type=int, default=pipe.MIN_COUNT)
    p.add_argument("--first-week-cut", type=int, default=pipe.FIRST_WEEK_CUT)
    p.add_argument("--year-min", type=int, default=None); p.add_argument("--year-max", type=int, default=None)
    p.add_argument("--gam-alpha-grid", type=int, default=0)
    p.add_argument("--national-offsets-only", action="store_true")
    p.add_argument("--skip-runs", action="store_true", help="rebuild comparisons from existing run_* folders")
    p.add_argument("--reuse", action="store_true", help="skip pipeline runs whose outputs already exist")
    args = p.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.outdir / "bfed_sensitivity.log")])

    raw = pd.read_csv(args.data_path, low_memory=False)
    col = [c for c in raw.columns if c.lower() == args.bfed_col.lower()]
    if not col:
        LOG.error("case file has no %s column", args.bfed_col); return 2
    group = bfed_group(raw[col[0]], args.yes_code, args.no_code)
    LOG.info("%s in the raw file: %s", col[0], group.value_counts().to_dict())

    # analytic cohort (ALL) for the missingness report and PCA distributions
    cohort, _ = pipe.load_ncfrp_data(args.data_path, pca_min=args.pca_min, pca_max=args.pca_max,
                                     first_week_cut=args.first_week_cut, year_min=args.year_min,
                                     year_max=args.year_max, audit_csv=None)
    cgroup = cohort_group(cohort, args)
    LOG.info("%s in the ANALYTIC cohort (n=%d): %s", col[0], len(cohort), cgroup.value_counts().to_dict())
    miss = missingness_report(raw, cohort, cgroup, args, args.outdir)
    pca_distribution(cohort, cgroup, args.outdir)

    # subset case files
    subsets = {"ALL": raw, "Y": raw[group == "Y"], "N": raw[group == "N"], "U": raw[group == "U"], "K": raw[group.isin(["Y", "N"])]}
    files = {}
    for g, df in subsets.items():
        f = args.outdir / f"cases_{g}.csv"; df.to_csv(f, index=False); files[g] = f
        LOG.info("subset %s: %d raw rows", g, len(df))

    versions = {"national": {g: args.national_exposure_dir for g in subsets}}
    if not args.national_offsets_only:
        dirs = build_bfed_exposure_dirs(args, args.outdir)
        versions["bfed"] = {"Y": dirs["Y"], "N": dirs["N"]}

    results = {}
    for tag, dmap in versions.items():
        runs = {}
        for g, exp_dir in dmap.items():
            run_dir = args.outdir / f"run_{g}_{tag}"
            done = (run_dir / "output_data" / "ncfrp" / "all_late_irr.csv").exists()
            if not args.skip_runs and not (args.reuse and done):
                run_pipeline(files[g], Path(exp_dir), run_dir, args)
            elif done:
                LOG.info("reusing existing outputs in %s", run_dir)
            runs[g] = read_run(run_dir)
        if tag == "national" and runs.get("ALL", {}).get("early") is not None:
            e = runs["ALL"]["early"].loc["bed_share", "IRR"]; l = runs["ALL"]["late"].loc["bed_share", "IRR"]
            ok = abs(e - EXPECTED_ALL["early"]) < 0.01 and abs(l - EXPECTED_ALL["late"]) < 0.01
            LOG.info("ALL reproduction of the primary run: bed_share early %.3f (expect %.3f), late %.3f (expect %.3f) -> %s",
                     e, EXPECTED_ALL["early"], l, EXPECTED_ALL["late"], "PASS" if ok else "FAIL")
        if tag == "bfed":  # show ALL/U/K from the national version alongside the bfed-offset Y/N
            for g in ("ALL", "U", "K"):
                runs[g] = results["national"][g]
        results[tag] = runs
        compare(runs, args.outdir, tag)

    # headline summary
    tag = "bfed" if "bfed" in results else "national"
    Y, N = results[tag]["Y"], results[tag]["N"]
    unk = miss.loc[miss["stratum"] == "overall", "pct_unknown"].iloc[0]
    with open(args.outdir / "bfed_sensitivity_summary.md", "w") as f:
        f.write("# Breastfeeding sensitivity analysis: summary\n\n")
        f.write(f"- Breastfeeding status unknown for {unk:.1f}% of the analytic cohort "
                f"(n = {int(miss.loc[miss['stratum']=='overall','n_U'].iloc[0]):,} of {int(miss.loc[miss['stratum']=='overall','n'].iloc[0]):,}); "
                f"ever breastfed {int(miss.loc[miss['stratum']=='overall','n_Y'].iloc[0]):,}, never {int(miss.loc[miss['stratum']=='overall','n_N'].iloc[0]):,}.\n")
        for win in ("early", "late"):
            if Y.get(win) is not None and N.get(win) is not None:
                ratio, lo, hi, pv = wald(Y[win].loc["bed_share"], N[win].loc["bed_share"])
                f.write(f"- Bed sharing, {win} window ({tag} offsets): breastfed {fmt(Y[win].loc['bed_share'])} vs not {fmt(N[win].loc['bed_share'])}; "
                        f"ratio {ratio:.2f} ({lo:.2f}-{hi:.2f}), P = {pv:.2f}.\n")
        f.write(f"\nFull tables: bfed_comparison_{tag}.csv/.md, bfed_missingness.csv/.md, bfed_pca_distribution.csv/.png; "
                "per-group pipeline outputs in run_<group>_<offsets>/.\n")
    LOG.info("Done. Summary: %s", args.outdir / "bfed_sensitivity_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

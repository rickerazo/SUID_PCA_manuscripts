#!/usr/bin/env python3
"""
risk_exposure_joint.py  (T5 / G2)

Builds the population exposure offsets that ncfrp_manuscript_v5.py multiplies
into each PCA x bed_share x smoke x preterm stratum's denominator.

Extends manuscript 1's
`../manuscript_1_sex_pca/src/risk_exposure_fixed.py` - the `compute_exposure`,
`compute_preterm` and `compute_bed_share` helpers are reused verbatim (below) -
with a JOINT smoking x preterm distribution instead of the product of the two
marginals. This is the G2 decision: the archived pipeline formed the stratum
offset as P(smoke) * P(preterm) * P(bed-share), assuming population independence,
which the case data contradict (smoking and bed sharing co-occur; smoking and
preterm co-occur). The joint P(smoke, preterm) comes straight from the CDC
live-birth denominator, where both fields are on every record.

Bed sharing stays a separate MARGINAL: PRAMS is a different survey that does not
link to the birth file and has no reliable gestational age, so the offset is
    exposure(smoke, preterm, bed_share)
        = P_joint(smoke, preterm)   [CDC live births]
        * P(bed_share)              [PRAMS]
i.e. bed sharing is assumed independent of {smoke, preterm} in the population.

--denominator MUST point at the FULL US live-birth file
(standardized/consolidated/denominator.parquet, ~30M rows for 2014-2021), NOT
manuscript 1's matched_cases / matched_controls set. Manuscript 1's case-control
matching (on birthweight etc.) shaped that matched set; it plays no role in these
offsets or in manuscript 2's models. Birthweight is not a covariate here, and
lr_bw_impute.py only fills the birthweight column inside the NCFRP case file for
a complete Table 1. A size check below warns if --denominator looks like a
matched set rather than the full denominator.
State this assumption in the manuscript; the analysis can also be re-run with
--offset-mode marginal (ncfrp_manuscript_v5.py) as a sensitivity comparison,
using the marginal pickles this script also writes.

Outputs (into --outdir, default output_data/risk_exposure/):
    joint_smoke_preterm_exposure.pkl   {(smoke, preterm): prob} - the 2x2 joint
    smoke_exposure.pkl / .csv          {0: P(non-smoke), 1: P(smoke)}   marginal
    term_exposure.pkl  / .csv          {0: P(term),      1: P(preterm)} marginal
    sex_exposure.pkl   / .csv          {0: P(female),    1: P(male)}    marginal
    bedshare_exposure.pkl / .csv       {0: P(not),       1: P(bed-share)} marginal
    joint_exposure_summary.txt         human-readable, with the smoke x preterm
                                       odds ratio / phi for context (R1.7 style)

CAVEAT (carried over from manuscript 1): PRAMS is a weighted survey; this uses
unweighted value_counts, matching risk_exposure_fixed.py. If a survey weight
column is available, pass --prams-weight-col to use it.

USAGE
    python risk_exposure_joint.py \
        --denominator standardized/consolidated/denominator.parquet \
        --prams       /path/to/PRAMS/prams.parquet \
        --outdir      output_data/risk_exposure
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

LOG = logging.getLogger("risk_exposure_joint")

COMBGEST_UNKNOWN = 99   # NCHS "not stated" sentinel (risk_exposure_fixed.py:59)
PRETERM_WEEKS = 37


# ---------------------------------------------------------------------------
# Reused verbatim from manuscript_1_sex_pca/src/risk_exposure_fixed.py
# ---------------------------------------------------------------------------

def compute_exposure(df: pd.DataFrame, var: str, option1, option2) -> dict:
    risk_ = df[var].value_counts() / df[var].value_counts().sum()
    return {0: float(risk_.loc[option1]), 1: float(risk_.loc[option2])}


def compute_preterm(df: pd.DataFrame) -> pd.Series:
    """
    preterm = 1 if combgest < 37 completed weeks, else 0. Rows with an
    unknown/not-stated combgest (>= 99) are excluded before the proportion is
    taken; left in, "99 is not < 37" would silently count them as term.
    """
    known = df.loc[df["combgest"] < COMBGEST_UNKNOWN].copy()
    known["preterm"] = (known["combgest"] < PRETERM_WEEKS).astype(int)
    return known["preterm"]


def compute_bed_share(df_: pd.DataFrame) -> pd.Series:
    """
    PRAMS SLEEPOWN_RAW: 1 = ALWAYS alone, 2 = OFTEN/ALMOST ALWAYS alone ->
    non-bed-sharing (0); 3/4/5 (sometimes/rarely/never alone) and blank/DK/skip
    -> bed sharing (1). The blank -> 1 fall-through is a known upward bias on the
    population bed-sharing prevalence (and so a downward bias on the IRR);
    flagged in risk_exposure_fixed.py.
    """
    bs = pd.Series(1, index=df_.index, dtype="int64")
    bs.loc[df_["sleepown_raw"].isin([1, 2])] = 0
    return bs


# ---------------------------------------------------------------------------
# New: the joint smoking x preterm distribution
# ---------------------------------------------------------------------------

def compute_joint_smoke_preterm(den: pd.DataFrame) -> dict:
    """
    P(smoke, preterm) over live births. smoke = (cig_rec == 'Y');
    preterm = (combgest < 37). Rows with unknown smoking or unknown combgest are
    dropped before normalising. Returns {(smoke, preterm): prob} with all four
    cells present (0.0 if empty).
    """
    d = den.copy()
    d = d.loc[d["cig_rec"].isin(["Y", "N"])]
    d = d.loc[d["combgest"] < COMBGEST_UNKNOWN]
    d["smoke"] = (d["cig_rec"] == "Y").astype(int)
    d["preterm"] = (d["combgest"] < PRETERM_WEEKS).astype(int)

    tab = pd.crosstab(d["smoke"], d["preterm"], normalize=True)
    joint = {}
    for s in (0, 1):
        for pt in (0, 1):
            joint[(s, pt)] = float(tab.loc[s, pt]) if (s in tab.index and pt in tab.columns) else 0.0
    total = sum(joint.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"joint smoke x preterm does not sum to 1 (got {total})")

    # context stats (reviewer-1-style: is the association real?)
    ct = pd.crosstab(d["smoke"], d["preterm"])
    chi2, p, _, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    phi = float(np.sqrt(chi2 / n))
    a, b, c, e = ct.to_numpy().ravel()
    cc = 0.5
    orr = float(((e + cc) * (a + cc)) / ((b + cc) * (c + cc)))
    LOG.info("smoke x preterm among live births: OR=%.3f, phi=%.4f, chi2 p=%.3g "
             "(n=%d)", orr, phi, p, int(n))
    return joint, {"or": orr, "phi": phi, "p": p, "n": int(n), "table": ct}


# ---------------------------------------------------------------------------

def _dump(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--denominator", type=Path, required=True,
                   help="CDC live-birth denominator parquet (cig_rec, combgest, sex)")
    p.add_argument("--prams", type=Path, required=True,
                   help="PRAMS parquet with sleepown_raw")
    p.add_argument("--outdir", type=Path, default=Path("output_data/risk_exposure"))
    p.add_argument("--prams-weight-col", default=None,
                   help="optional PRAMS survey-weight column; unweighted if omitted "
                        "(matches manuscript 1)")
    p.add_argument("--seed", type=int, default=20260902)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    np.random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    den = pd.read_parquet(args.denominator)
    prams = pd.read_parquet(args.prams)
    LOG.info("denominator: %s   prams: %s", den.shape, prams.shape)
    if len(den) < 1_000_000:
        LOG.warning("--denominator has only %d rows. The full US live-birth "
                    "denominator is tens of millions; this looks like a matched "
                    "case/control set. Offsets built from it would be biased. "
                    "Point --denominator at standardized/consolidated/denominator.parquet.",
                    len(den))

    # --- marginals (same as risk_exposure_fixed.py) ---
    den = den.copy()
    den["preterm"] = compute_preterm(den)
    smoke_exposure = compute_exposure(den.loc[den["cig_rec"].isin(["Y", "N"])],
                                      "cig_rec", "N", "Y")
    sex_exposure = compute_exposure(den.loc[den["sex"].isin(["F", "M"])],
                                    "sex", "F", "M")
    term_exposure = compute_exposure(den.dropna(subset=["preterm"]), "preterm", 0, 1)

    prams = prams.copy()
    prams["bedshare"] = compute_bed_share(prams)
    if args.prams_weight_col and args.prams_weight_col in prams.columns:
        w = prams[args.prams_weight_col].astype(float)
        p_bs = float((w[prams["bedshare"] == 1].sum()) / w.sum())
        bedshare_exposure = {0: 1.0 - p_bs, 1: p_bs}
        LOG.info("bed-share prevalence is survey-weighted on %s", args.prams_weight_col)
    else:
        bedshare_exposure = compute_exposure(prams, "bedshare", 0, 1)

    # --- the joint ---
    joint, ctx = compute_joint_smoke_preterm(den)

    # --- write ---
    for name, d in [("smoke_exposure", smoke_exposure), ("sex_exposure", sex_exposure),
                    ("term_exposure", term_exposure), ("bedshare_exposure", bedshare_exposure)]:
        _dump(d, args.outdir / f"{name}.pkl")
        pd.DataFrame([d]).to_csv(args.outdir / f"{name}.csv", index=False)
    _dump(joint, args.outdir / "joint_smoke_preterm_exposure.pkl")
    pd.DataFrame([{f"smoke{s}_preterm{pt}": v for (s, pt), v in joint.items()}]).to_csv(
        args.outdir / "joint_smoke_preterm_exposure.csv", index=False)

    with open(args.outdir / "joint_exposure_summary.txt", "w") as f:
        f.write("Population exposure offsets for ncfrp_manuscript_v5.py\n")
        f.write("=" * 70 + "\n\n")
        f.write("JOINT P(smoke, preterm) over live births (used in --offset-mode joint):\n")
        for (s, pt), v in sorted(joint.items()):
            f.write(f"  smoke={s} preterm={pt}: {v:.6f}\n")
        f.write(f"\n  smoke x preterm: OR={ctx['or']:.3f}, phi={ctx['phi']:.4f}, "
                f"chi2 p={ctx['p']:.3g}, n={ctx['n']}\n")
        f.write(f"  contingency table (rows smoke 0/1, cols preterm 0/1):\n{ctx['table']}\n\n")
        f.write("MARGINALS (used in --offset-mode marginal, the sensitivity comparison):\n")
        f.write(f"  smoke_exposure    {smoke_exposure}\n")
        f.write(f"  term_exposure     {term_exposure}\n")
        f.write(f"  sex_exposure      {sex_exposure}\n")
        f.write(f"  bedshare_exposure {bedshare_exposure}\n\n")
        f.write("offset(smoke, preterm, bed_share) = joint[(smoke,preterm)] * "
                "bedshare_exposure[bed_share]   (bed sharing assumed independent "
                "of {smoke, preterm} in the population; PRAMS does not link to the "
                "birth file).\n")

    LOG.info("Wrote joint + marginal exposure files to %s", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
generate_supplemental_table4.py

Formats the late-window negative binomial model output (already fit and
validated by run_windowed_nb_models.py) into "Supplemental Table 4",
mirroring the row structure and formatting conventions of Supplemental
Table 1 (predictor | IRR (95% CI) | p), as promised in the response to the
reviewer comment about interaction-term reporting.

Does not refit anything — reads the already-validated
nb_models/corrected/irr_late_window_corrected.csv and formats it.

USAGE
-----
    python generate_supplemental_table4.py \
        --input output_data/nb_models/corrected/irr_late_window_corrected.csv \
        --outdir .
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Row order + display labels, matching Supplemental Table 1's convention.
PREDICTOR_ORDER = [
    "PCA (per week)",
    "Male (vs female)",
    "Maternal smoking",
    "Preterm birth",
    "Smoking x Preterm",
    "Sex x Preterm",
]
DISPLAY_LABEL = {
    "Smoking x Preterm": "Smoking × Preterm",
    "Sex x Preterm": "Sex × Preterm",
}


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_irr_ci(irr: float, lo: float, hi: float) -> str:
    return f"{irr:.3f} ({lo:.3f}–{hi:.3f})"


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.set_index("Predictor")
    rows = []
    for name in PREDICTOR_ORDER:
        r = idx.loc[name]
        rows.append({
            "Predictor": DISPLAY_LABEL.get(name, name),
            "IRR (95% CI)": fmt_irr_ci(r["IRR"], r["CI_low"], r["CI_high"]),
            "p": fmt_p(r["p"]),
        })

    alpha = idx.loc["Alpha (overdispersion)", "IRR"]
    n_strata = int(idx.loc["Observations (strata rows)", "IRR"])
    n_cases = int(idx.loc["Total cases in window", "IRR"])
    llf = idx.loc["Log-likelihood", "IRR"]
    pr2 = idx.loc["Pseudo-R2", "IRR"]

    rows.append({"Predictor": "Alpha", "IRR (95% CI)": f"{alpha:.3g}", "p": "—"})
    rows.append({"Predictor": "Observations (weeks)", "IRR (95% CI)": str(n_strata), "p": "—"})
    rows.append({"Predictor": "Total cases in window", "IRR (95% CI)": f"{n_cases:,}", "p": "—"})
    rows.append({"Predictor": "Log-likelihood", "IRR (95% CI)": f"{llf:.2f}", "p": "—"})
    rows.append({"Predictor": "Pseudo-R²", "IRR (95% CI)": f"{pr2:.3f}", "p": "—"})

    return pd.DataFrame(rows)


def to_markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False))
    return "\n".join([header, sep, body])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                    default=Path("nb_models/corrected/irr_late_window_corrected.csv"))
    p.add_argument("--outdir", type=Path, default=Path("."))
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input)
    table = build_table(raw)

    csv_path = args.outdir / "supplemental_table4.csv"
    md_path = args.outdir / "supplemental_table4.md"
    table.to_csv(csv_path, index=False)

    title = (
        "Supplemental Table 4. Negative binomial model estimates for the late "
        "developmental window (PCA ≥47 weeks). Entries are IRR (95% CI); "
        "Smoking × Preterm and Sex × Preterm are prespecified interaction "
        "terms, retained regardless of significance (see Methods). IRRs are "
        "exponentiated coefficients."
    )
    with open(md_path, "w") as f:
        f.write(title + "\n\n")
        f.write(to_markdown_table(table))
        f.write("\n")

    print(title)
    print()
    print(table.to_string(index=False))
    print(f"\nWrote {csv_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

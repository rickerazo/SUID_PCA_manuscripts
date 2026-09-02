#!/usr/bin/env python3
"""
generate_figure1_flow.py  (AUDIT A4.28 / R1.4 / Ed E4)

Rebuilds Figure 1 (the case-selection CONSORT flow) straight from the audit
trail that ncfrp_manuscript_v5.py writes, so every "excluded n" and "remaining
N" reconciles and the two analytic populations (descriptive/GAM cohort vs
early/late windowed cohort, with the peak set aside) are shown explicitly. The
archived Figure_1.docx did not reconcile (steps out of order, a count listed
twice with two different percentages).

Inputs (produced by a run of ncfrp_manuscript_v5.py):
    output_data/ncfrp_flow_audit.csv        one row per cleaning step
    output_data/ncfrp/two_population_Ns.csv  full / early / late / peak-excluded N

Outputs:
    output_figures/ncfrp/Figure1_flow.png / .tiff
    output_data/ncfrp/figure1_flow.txt       plain-text reconciling flow

USAGE
    python generate_figure1_flow.py --workdir .    # dir holding output_data/, output_figures/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LOG = logging.getLogger("generate_figure1_flow")


def _text_flow(audit: pd.DataFrame, pops: dict) -> str:
    lines = ["Figure 1. SUID case selection (reconciled from ncfrp_flow_audit.csv)",
             "=" * 70, ""]
    start = int(audit.iloc[0]["n_after"])
    lines.append(f"Start: all NCFRP SUID cases in the export            N = {start:,}")
    for _, r in audit.iloc[1:].iterrows():
        if int(r["n_removed"]) == 0 and "cap" in str(r["step"]).lower():
            lines.append(f"  [{r['step']}]  (recode only, {r['note']})")
            continue
        lines.append(f"  Excluded: {r['step']:<52} n = {int(r['n_removed']):,}")
        lines.append(f"           remaining                                     N = {int(r['n_after']):,}")
    lines += ["",
              f"Analytic cohort (descriptive curves, GAM, Table 1)     N = {pops['full']:,}",
              f"  Early rising window (PCA < peak1)                     n = {pops['early']:,}",
              f"  Late declining window (PCA >= peak2)                  n = {pops['late']:,}",
              f"  Peak interval (set aside from the windowed models)    n = {pops['peak']:,}",
              "",
              f"  check: early + late + peak = {pops['early'] + pops['late'] + pops['peak']:,} "
              f"(should equal {pops['full']:,})"]
    return "\n".join(lines)


def _draw(audit: pd.DataFrame, pops: dict, out_png: Path, out_tiff: Path) -> None:
    steps = []
    steps.append(("Start: NCFRP SUID cases", int(audit.iloc[0]["n_after"]), None))
    for _, r in audit.iloc[1:].iterrows():
        if int(r["n_removed"]) == 0:
            continue
        steps.append((str(r["step"]), int(r["n_after"]), int(r["n_removed"])))

    n = len(steps)
    fig, ax = plt.subplots(figsize=(9, 1.15 * n + 2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n + 2)
    ax.axis("off")

    for i, (label, remaining, removed) in enumerate(steps):
        y = n + 1 - i
        ax.add_patch(plt.Rectangle((1, y - 0.35), 5.0, 0.7, fill=True,
                                   facecolor="#eef3fb", edgecolor="#33507a"))
        ax.text(3.5, y, f"{label}\nN = {remaining:,}", ha="center", va="center", fontsize=9)
        if i < n - 1:
            ax.annotate("", xy=(3.5, y - 0.45), xytext=(3.5, y - 0.55),
                        arrowprops=dict(arrowstyle="-|>", color="#33507a"))
        if removed is not None:
            ax.add_patch(plt.Rectangle((7.0, y - 0.28), 2.6, 0.56, fill=True,
                                       facecolor="#faf0ee", edgecolor="#7a4a33"))
            ax.text(8.3, y, f"excluded\nn = {removed:,}", ha="center", va="center", fontsize=8)
            ax.annotate("", xy=(7.0, y), xytext=(6.0, y),
                        arrowprops=dict(arrowstyle="-|>", color="#7a4a33"))

    y0 = 0.7
    ax.text(3.5, y0 + 0.55,
            f"Analytic cohort  N = {pops['full']:,}\n"
            f"early n = {pops['early']:,}   peak (set aside) n = {pops['peak']:,}   "
            f"late n = {pops['late']:,}",
            ha="center", va="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_tiff, dpi=300, format="tiff", bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", type=Path, default=Path("."))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    audit_path = args.workdir / "output_data" / "ncfrp_flow_audit.csv"
    pops_path = args.workdir / "output_data" / "ncfrp" / "two_population_Ns.csv"
    if not audit_path.exists() or not pops_path.exists():
        raise FileNotFoundError(f"Run ncfrp_manuscript_v5.py first; need {audit_path} "
                                f"and {pops_path}")

    audit = pd.read_csv(audit_path)
    pr = pd.read_csv(pops_path).iloc[0]
    pops = {"full": int(pr["full_analytic_N"]), "early": int(pr["early_n"]),
            "late": int(pr["late_n"]), "peak": int(pr["peak_excluded_n"])}

    txt = _text_flow(audit, pops)
    out_txt = args.workdir / "output_data" / "ncfrp" / "figure1_flow.txt"
    out_txt.write_text(txt)
    print(txt)

    fig_dir = args.workdir / "output_figures" / "ncfrp"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _draw(audit, pops, fig_dir / "Figure1_flow.png", fig_dir / "Figure1_flow.tiff")
    LOG.info("Wrote %s and Figure1_flow.png/.tiff", out_txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""MILP lexicographic flow (Arm C) vs standard FINN flow (Arm B): same MILP-mixed
bits (MILP/artifacts/S12_dense_arms_bc_v1/lex_dsr2_fps305, pass 1 accuracy at
FPS >= 305.17, all caps 1.0, DSR 2; pass 2 min resources at fixed accuracy,
mvau_wwidth_max 80). Arm B folds with FINN auto-fold, Arm C with the MILP's
pass-2 folding. Real post-synth resources (% of XCZU7EV) + FIFO-only BRAM,
real rtlsim FPS above each group. Loading/normalisation reused from
plot_dsr_ablation_hw.py.

Usage: python analysis/report/MILP/plot_lex_arms_bc_hw.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_dsr_ablation_hw import (  # noqa: E402
    INK, SURFACE, REPO_ROOT, _style_axes, draw_bars, load, load_analytical, write_table,
)

ART = REPO_ROOT / "MILP/artifacts/S12_dense_arms_bc_v1/lex_dsr2_fps305"
OUT = Path(__file__).resolve().parent / "figures" / "lex_arms_bc_hw_resources.png"
OUT_CSV = Path(__file__).resolve().parent / "lex_arms_bc_hw_table.csv"
# (label, build tag, artifact dir with the MILP's own estimate or None)
ARMS = [
    ("FINN fold", "armB_finn_autofold", None),
    ("MILP fold", "armC_dsr2", ART),
]


def main() -> int:
    import matplotlib.pyplot as plt

    arms = ARMS + [("Analytical", None, None)]
    rows = [load(t, a) for _, t, a in ARMS] + [load_analytical()]
    write_table(OUT_CSV, ["arm", "bits", "folding"],
                [["B", "MILP mixed", "FINN auto-fold"], ["C", "MILP mixed", "MILP"],
                 ["U4 analytical", "uniform INT6 (256x256)", "analytical"]], rows)
    print(f"Wrote {OUT_CSV}")
    for (lab, *_), r in zip(arms, rows):
        print(lab.replace("\n", " "), {k: round(v, 2) if v is not None else v for k, v in r.items()})

    w = 0.16
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=SURFACE)
    _style_axes(ax)
    draw_bars(ax, rows, w)
    ax.set_xticks(range(len(rows)), [a[0] for a in arms])
    ax.set_ylabel("% of XCZU7EV", color=INK, fontsize=10, fontweight="bold")
    ax.set_xlabel("Folding", color=INK, fontsize=10, fontweight="bold")
    ax.set_title("Post Synthesis Resource Utilization: MILP vs FINN folding at Fixed Quantization, FPS target and MVAU Max. Width",
                 color=INK, fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

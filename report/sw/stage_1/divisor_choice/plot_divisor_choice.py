"""Why divisor 4: per-step % decrease in parameters vs % decrease in Dice for the
S12 dense ReLU bilinear (upsample_conv) width sweep. Data: nas/results.csv (test split).
Writes divisor_choice.csv + divisor_choice.png. Run in the dev container:
  docker exec lightmunet_dev python3 /workspace/LightM-UNet/report/sw/stage_1/divisor_choice/plot_divisor_choice.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1]))  # report/sw
import nas_stages as ns  # noqa: E402

RUNS = [(1, "12_dense_relu_u1"), (2, "12_dense_relu_u2"), (4, "12_dense_relu"),
        (8, "12_dense_relu_w8_16"), (16, "12_dense_relu_w4_8")]
rows = {r["config_name"]: r for r in csv.DictReader(open(REPO / "nas" / "results.csv", newline=""))}
pts = [(d, float(rows[f"nnUNetTrainerENet_{n}"]["params"]), float(rows[f"nnUNetTrainerENet_{n}"]["dice"]))
       for d, n in RUNS]

steps = []
for (d0, p0, y0), (d1, p1, y1) in zip(pts, pts[1:]):
    steps.append({"step": f"{d0}->{d1}", "params_from": p0, "params_to": p1, "dice_from": y0, "dice_to": y1,
                  "params_decrease_pct": (1 - p1 / p0) * 100, "dice_decrease_pct": (1 - y1 / y0) * 100,
                  "dice_decrease_points": (y0 - y1) * 100})
with open(HERE / "divisor_choice.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(steps[0]))
    w.writeheader()
    w.writerows(steps)

labels = [f"1/{d0}" if d0 > 1 else "1" for d0, _, _ in pts[:-1]]
labels = [f"{a} \u2192 1/{d1}" for a, (d1, _, _) in zip(labels, pts[1:])]
x = range(len(steps))
bw = 0.38
fig, ax = plt.subplots(figsize=(5.0, 3.8), facecolor=ns.SURFACE)
ns.style_axes(ax)
b1 = ax.bar([i - bw / 2 for i in x], [s["params_decrease_pct"] for s in steps], bw,
            color=ns.SERIES_COLORS[0], zorder=3, label="Parameters")
b2 = ax.bar([i + bw / 2 for i in x], [s["dice_decrease_pct"] for s in steps], bw,
            color=ns.SERIES_COLORS[2], zorder=3, label="Dice")
for bars in (b1, b2):
    ax.bar_label(bars, fmt="%.0f", fontsize=7.5, fontweight="bold", color=ns.INK, padding=2)
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylim(0, 100)
ns.style_title_and_labels(ax, "Cost of Each Width Reduction Step", "Channel Width Multiplier", "Decrease (%)")
fig.canvas.draw()
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontweight("bold")
for lab in (ax.xaxis.label, ax.yaxis.label):
    lab.set_fontweight("bold"); lab.set_color(ns.INK); lab.set_fontsize(11)
ax.legend(loc="upper left", frameon=True, facecolor=ns.SURFACE, edgecolor="#c3c2b7", fontsize=6.5,
          labelcolor=ns.SECONDARY_INK, borderpad=0.4, labelspacing=0.3)
fig.tight_layout()
out = HERE / "divisor_choice.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=ns.SURFACE)
print(f"Wrote {out}")
for s in steps:
    print(s["step"], f"params -{s['params_decrease_pct']:.1f}%  dice -{s['dice_decrease_pct']:.1f}% ({s['dice_decrease_points']:.1f} pts)")

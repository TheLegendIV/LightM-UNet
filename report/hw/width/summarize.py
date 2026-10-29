"""Collect report/hw/width/<arm_set>/{q,h}_int<b>/summary.csv into width_cost_summary.csv (q = 1/4 width, h = 1/2 width) and print it. Run in lightmunet_dev (or any python3)."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEV = {"lut": 230_400, "bram18": 624, "dsp": 1_728}
rows = []
for arm_set in ("relaxed_fps150", "cap40_fps150"):
    for w, wname in (("q", "1/4"), ("h", "1/2")):
        for b in (4, 6, 8):
            d = HERE / arm_set / f"{w}_int{b}"
            st = json.load(open(d / "layer_bits_folding.json"))["status"]
            r = dict(arm_set=arm_set, width=wname, bits=b, status=st, lut="", lut_pct="", bram18="", bram_pct="", dsp="", dsp_pct="", fps="")
            if st == "Optimal":
                s = next(csv.DictReader(open(d / "summary.csv")))
                lp, bp = float(s["lut_pct_of_budget"]), float(s["bram_pct_of_budget"])
                r.update(lut=round(lp / 100 * DEV["lut"]), lut_pct=round(lp, 1), bram18=round(bp / 100 * DEV["bram18"]), bram_pct=round(bp, 1),
                         dsp=int(float(s["total_dsp"])), dsp_pct=round(100 * float(s["total_dsp"]) / DEV["dsp"], 1), fps=round(float(s["fps"]), 1))
            rows.append(r)
with open(HERE / "width_cost_summary.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
for r in rows:
    print(r)

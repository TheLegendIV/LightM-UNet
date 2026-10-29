"""Print/collect maxfps_caps1/{q,h}_int<b> (q = 1/4 width, h = 1/2 width) into maxfps_summary.csv."""
import csv
from pathlib import Path

H = Path(__file__).resolve().parent / "maxfps_caps1"
DEV = {"lut": 230_400, "bram18": 624, "dsp": 1_728}
rows = []
for w, wn in (("q", "1/4"), ("h", "1/2")):
    for b in (4, 6, 8):
        s = next(csv.DictReader(open(H / f"{w}_int{b}" / "summary.csv")))
        rows.append(dict(width=wn, bits=b, status=s["status"], fps=round(float(s["fps"]), 1), cycles_per_frame=round(100e6 / float(s["fps"])), bottleneck=s["bottleneck_node"],
                         lut=round(float(s["lut_pct_of_budget"]) / 100 * DEV["lut"]), lut_pct=round(float(s["lut_pct_of_budget"]), 1),
                         bram18=round(float(s["bram_pct_of_budget"]) / 100 * DEV["bram18"]), bram_pct=round(float(s["bram_pct_of_budget"]), 1),
                         dsp=int(float(s["total_dsp"])), dsp_pct=round(float(s["dsp_pct_of_budget"]), 1)))
with open(Path(__file__).resolve().parent / "maxfps_summary.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
for r in rows:
    print(r)

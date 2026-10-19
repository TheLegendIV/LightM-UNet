"""Synthesized (out-of-context, 100 MHz target, xczu7ev) per-partition utilization of the constrained 150 FPS builds -> synth_utilization.csv.

Sources: hardware/builds/S12_512_ablation_v2/<solve>/<arm>/{8way,p7fix}/report/ooc_synth_partition_<i>.json (MILP builds) and
finn_auto/cap40_fps150/uniform_int6/report (FINN auto-fold + FINN FIFO autosize; partition 7 is post-synthesis only, it cannot be placed).
The INT4 / INT6 builds use the partition 7 rebuild that lands the 512-deep inter-block FIFO (partitions 0-6 from the 8-way build).
Run:  python analysis/report/MILP/constrained_builds/build_synth_utilization_csv.py
"""
import csv
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
B = os.path.join(REPO, "hardware", "builds", "S12_512_ablation_v2")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synth_utilization.csv")

# build id, precision, folding, arm folder, partition-7 override folder (or None), notes
# INT4 / INT6: partitions 0-6 from the 8-way build, partition 7 from the rebuild that lands the 512-deep inter-block FIFO. The original
# 8-way partition 7 kept that FIFO at depth 2 (not what the MILP / FIFO simulation specified) and is therefore not listed.
BUILDS = [
    ("int4_cap40", "INT4", "milp", "cap40_fps150/uniform_int4", "cap40_fps150/uniform_int4/p7fix",
     "partitions 0-6 from the 8-way build, partition 7 rebuilt with the inter-block FIFO fix"),
    ("int6_cap40", "INT6", "milp", "cap40_fps150/uniform_int6", "cap40_fps150/uniform_int6/p7fix",
     "partitions 0-6 from the 8-way build, partition 7 rebuilt with the inter-block FIFO fix"),
    ("mixed468_cap40", "mixed 4/6/8", "milp", "cap40_fps150/mixed_468", None, "inter-block FIFO left at depth 2 (0.4 caps rejected the grown FIFO)"),
    ("int6_cap40_finnauto", "INT6", "finn_auto", "finn_auto/cap40_fps150/uniform_int6", None,
     "FINN auto-fold + FINN rtlsim FIFO autosize; partition 7 is post-synthesis only (placement fails, UTLZ-1), no timing for it"),
]
COLS = ["LUT", "LUTRAM", "FF", "DSP", "BRAM_18K", "BRAM_36K", "URAM", "Carry", "WNS", "fmax_mhz"]


def report_dir(arm: str, kind: str) -> str:
    if arm.startswith("finn_auto"):
        return os.path.join(B, arm, "report")
    return os.path.join(B, arm, kind, "report")


def load(arm: str, i: int, kind: str = "8way") -> dict:
    with open(os.path.join(report_dir(arm, kind), f"ooc_synth_partition_{i}.json")) as f:
        return json.load(f)


rows = []
for bid, prec, folding, arm, p7_arm, notes in BUILDS:
    parts = []
    for i in range(8):
        if i == 7 and p7_arm:
            d = load(p7_arm.rsplit("/p7fix", 1)[0], 7, "p7fix")
            src = "p7fix"
        else:
            d = load(arm, i)
            src = "finn_auto" if folding == "finn_auto" else "8way"
        parts.append((i, src, d))
    total = {c: 0.0 for c in ("LUT", "LUTRAM", "FF", "DSP", "BRAM_18K", "BRAM_36K", "URAM", "Carry")}
    fmax = []
    for i, src, d in parts:
        row = {"build": bid, "precision": prec, "folding": folding, "target_fps": 150, "partition": i, "source": src,
               "synth_only": bool(d.get("synth_only", False)), "notes": ""}
        for c in COLS:
            row[c] = d.get(c, "")
        row["BRAM18_equiv"] = d["BRAM_18K"] + 2 * d["BRAM_36K"]
        rows.append(row)
        for c in total:
            total[c] += d.get(c, 0.0)
        if d.get("fmax_mhz") is not None:
            fmax.append(d["fmax_mhz"])
    tot = {"build": bid, "precision": prec, "folding": folding, "target_fps": 150, "partition": "total", "source": "sum",
           "synth_only": any(d.get("synth_only", False) for _, _, d in parts), "notes": notes, **total,
           "WNS": "", "fmax_mhz": min(fmax), "BRAM18_equiv": total["BRAM_18K"] + 2 * total["BRAM_36K"]}
    if len(fmax) < 8:
        tot["notes"] += f"; fmax_mhz = min over the {len(fmax)} partitions with timing"
    rows.append(tot)

fields = ["build", "precision", "folding", "target_fps", "partition", "source", "synth_only"] + COLS[:8] + ["BRAM18_equiv", "WNS", "fmax_mhz", "notes"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
print(f"wrote {OUT} ({len(rows)} rows)")
for r in rows:
    if r["partition"] == "total":
        print(f"{r['build']:22s} LUT {r['LUT']:9.0f} FF {r['FF']:9.0f} BRAM18eq {r['BRAM18_equiv']:7.0f} fmax {r['fmax_mhz']:.1f}")

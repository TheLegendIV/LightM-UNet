"""Skip vs non-skip total FIFO depth per run (our sizer, auto saturated, auto paced), partition 7 excluded.

skip     = the buffered FIFO feeding each Add/Concat merge (28 merges, minus the p7 one)
non-skip = every other FIFO in partitions 0-6
"""
import json
import sys

sat_p, paced_p, sizer_p, cmp_p, out_p = sys.argv[1:6]

STAGE_PART = {"initial": 0, "down1": 1, "down2": 1, "up4": 5, "up5": 6, "regular5.0": 7, "final": 7}
for i in range(4):
    STAGE_PART[f"regular1.{i}"] = 1
for i in range(8):
    STAGE_PART[f"stage2.{i}"] = 2 if i <= 4 else 3
    STAGE_PART[f"stage3.{i}"] = {0: 3, 1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 5, 7: 5}[i]
STAGE_PART["stage2.5"] = STAGE_PART["stage2.6"] = STAGE_PART["stage2.7"] = 3
STAGE_PART["regular4.0"] = STAGE_PART["regular4.1"] = 6


def pow2(d):
    return d if d <= 256 else 1 << (d - 1).bit_length()


def blank():
    return {"skip": [0, 0.0], "nonskip": [0, 0.0]}


def add(acc, kind, depth, w):
    acc[kind][0] += depth
    acc[kind][1] += depth * w / 1024


runs = {k: {p: blank() for p in range(7)}
        for k in ("sizer_nominal", "sizer_hw_realised", "sizer_pow2_all_gt256", "auto_saturated", "auto_paced")}

# FINN rounds only impl_style=vivado FIFOs to a power of 2 (get_adjusted_depth); rtl/Q_srl FIFOs stay exact.
sz = json.load(open(sizer_p))
for f in sz["intra_block_fifos"]:
    p = STAGE_PART[f["stage"]]
    if p == 7:
        continue
    kind = "skip" if f["is_skip"] else "nonskip"
    hw = pow2(f["depth"]) if str(f.get("finn_impl", "")).startswith("vivado") else f["depth"]
    add(runs["sizer_nominal"][p], kind, f["depth"], f["width_bits"])
    add(runs["sizer_hw_realised"][p], kind, hw, f["width_bits"])
    add(runs["sizer_pow2_all_gt256"][p], kind, pow2(f["depth"]), f["width_bits"])
for f in sz["inter_block_fifos"]:
    ps, pd = STAGE_PART[f["src_block"]], STAGE_PART.get(f["dst_block"], 7)
    if ps == 7 or pd == 7:
        continue
    hw = pow2(f["depth"]) if f.get("mem") == "bram" else f["depth"]
    add(runs["sizer_nominal"][ps], "nonskip", f["depth"], f["width_bits"])
    add(runs["sizer_hw_realised"][ps], "nonskip", hw, f["width_bits"])
    add(runs["sizer_pow2_all_gt256"][ps], "nonskip", pow2(f["depth"]), f["width_bits"])

cmp_ = json.load(open(cmp_p))
for lab, path, dk in [("auto_saturated", sat_p, "sat_depth"), ("auto_paced", paced_p, "paced_depth")]:
    d = json.load(open(path))
    for p in range(7):
        for e in cmp_[str(p)]["edges"]:
            add(runs[lab][p], "nonskip", e[dk], e["width_bits"])
        for r in d[str(p)]:
            a, b = r["merge_fifo"], r["main_merge_fifo"]
            s = a if a["real_depth"] >= b["real_depth"] else b
            add(runs[lab][p], "skip", s["real_depth"], s["width_bits"])
            add(runs[lab][p], "nonskip", -s["real_depth"], s["width_bits"])  # edges above included it

out = {"per_partition": {}, "total": {}}
print(f"{'run':16s}{'skip words':>12s}{'skip kbit':>11s}{'nonskip words':>15s}{'nonskip kbit':>14s}")
for lab, parts in runs.items():
    t = blank()
    for p, v in parts.items():
        for k in t:
            t[k][0] += v[k][0]
            t[k][1] += v[k][1]
    out["total"][lab] = {k: {"words": v[0], "kbit": round(v[1], 1)} for k, v in t.items()}
    out["per_partition"][lab] = {str(p): {k: {"words": v[k][0], "kbit": round(v[k][1], 1)} for k in v}
                                 for p, v in parts.items()}
    print(f"{lab:16s}{t['skip'][0]:>12d}{t['skip'][1]:>11.0f}{t['nonskip'][0]:>15d}{t['nonskip'][1]:>14.0f}")
print()
print("per-partition words (skip / nonskip)")
print(f"{'run':16s}" + "".join(f"{'p' + str(p):>18s}" for p in range(7)))
for lab, parts in runs.items():
    print(f"{lab:16s}" + "".join(f"{parts[p]['skip'][0]:>9d}/{parts[p]['nonskip'][0]:<8d}" for p in range(7)))
json.dump(out, open(out_p, "w"), indent=1)

"""Sum skip-branch FIFO depths: our sizer vs FINN auto (saturated) vs FINN auto (paced).

Inputs: JSONs written by skip_fifo_depths.py (one per FINN run) and the sizer simfifo JSON.
The skip (buffered) branch of each merge is taken as the merge input whose FIFO is deeper.
Merges are paired with the sizer's skip_FIFO entries in network order.
"""
import json
import sys

sat_p, paced_p, sizer_p, out_p = sys.argv[1:5]


def pow2(d):
    return d if d <= 256 else 1 << (d - 1).bit_length()


def merges(path):
    d = json.load(open(path))
    rows = []
    for p in sorted(d, key=int):
        for r in d[p]:
            a, b = r["merge_fifo"], r["main_merge_fifo"]
            sk = a if a["real_depth"] >= b["real_depth"] else b
            rows.append({"part": int(p), "node": r["node"], "fifo": sk["name"], "depth": sk["depth"],
                         "real_depth": sk["real_depth"], "width": sk["width_bits"]})
    return rows


sat, paced = merges(sat_p), merges(paced_p)
sz = json.load(open(sizer_p))
sk = [f for f in sz["intra_block_fifos"] if f.get("is_skip")]
assert len(sat) == len(paced) == len(sk), (len(sat), len(paced), len(sk))

tab = []
for s, a, b in zip(sk, sat, paced):
    tab.append({"sizer_name": s["name"], "part": a["part"], "node": a["node"],
                "sizer_depth": s["depth"], "sizer_pow2": pow2(s["depth"]), "sizer_w": s["width_bits"],
                "sat_depth": a["real_depth"], "sat_w": a["width"],
                "paced_depth": b["real_depth"], "paced_w": b["width"]})


def tot(key, wkey):
    return sum(r[key] for r in tab), sum(r[key] * r[wkey] for r in tab) / 1024


summary = {}
for lab, k, w in [("sizer_nominal", "sizer_depth", "sizer_w"), ("sizer_pow2", "sizer_pow2", "sizer_w"),
                  ("auto_saturated", "sat_depth", "sat_w"), ("auto_paced", "paced_depth", "paced_w")]:
    wd, kb = tot(k, w)
    summary[lab] = {"words": wd, "kbit": round(kb, 1)}

json.dump({"summary": summary, "per_merge": tab}, open(out_p, "w"), indent=1)
print(f"{'sizer_stage':24s}{'p':>2s} {'node':22s}{'sizer':>8s}{'pow2':>8s}{'sat':>8s}{'paced':>8s}")
for r in tab:
    print(f"{r['sizer_name'][:24]:24s}{r['part']:>2d} {r['node']:22s}{r['sizer_depth']:>8d}{r['sizer_pow2']:>8d}"
          f"{r['sat_depth']:>8d}{r['paced_depth']:>8d}")
print(json.dumps(summary, indent=1))

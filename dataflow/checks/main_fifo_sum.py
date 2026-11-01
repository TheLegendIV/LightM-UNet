"""Sum the main (non-skip) FIFOs: our sizer vs FINN auto saturated vs FINN auto paced.

A) main-branch merge FIFO: the FIFO feeding each Add/Concat on the non-skip side (28 merges).
B) all non-skip FIFOs in the whole net (= total FIFOs minus the 28 skip merge FIFOs).
Sizer rows apply the pow2 rounding (>256) that FINN's vivado-impl FIFOs get.
"""
import json
import sys

sat_p, paced_p, sizer_p, cmp_p, out_p = sys.argv[1:6]


def pow2(d):
    return d if d <= 256 else 1 << (d - 1).bit_length()


def merge_rows(path):
    d = json.load(open(path))
    skip, main = [], []
    for p in sorted(d, key=int):
        for r in d[p]:
            a, b = r["merge_fifo"], r["main_merge_fifo"]
            s, m = (a, b) if a["real_depth"] >= b["real_depth"] else (b, a)
            skip.append(s)
            main.append(m)
    return skip, main


def agg(fifos, dkey, wkey):
    return {"words": sum(f[dkey] for f in fifos),
            "kbit": round(sum(f[dkey] * f[wkey] for f in fifos) / 1024, 1)}


res = {}
sz = json.load(open(sizer_p))
intra = sz["intra_block_fifos"]
non_skip = [f for f in intra if not f.get("is_skip")]
merge_main = [f for f in non_skip if f["consumer_node"] in ("Add", "Concat") and "skipFIFO" not in f["name"]]
inter = sz["inter_block_fifos"]
for f in non_skip + inter:
    f["pow2"] = pow2(f["depth"])
for f in merge_main:
    f["pow2"] = pow2(f["depth"])
res["merge_count_sizer"] = len(merge_main)
res["A_main_merge_fifo"] = {
    "sizer_nominal": agg(merge_main, "depth", "width_bits"),
    "sizer_pow2": agg(merge_main, "pow2", "width_bits"),
}
res["B_all_non_skip"] = {
    "sizer_nominal": agg(non_skip + inter, "depth", "width_bits"),
    "sizer_pow2": agg(non_skip + inter, "pow2", "width_bits"),
}
res["B_all_non_skip"]["sizer_nominal_intra_only"] = agg(non_skip, "depth", "width_bits")
res["B_all_non_skip"]["sizer_pow2_intra_only"] = agg(non_skip, "pow2", "width_bits")

cmp_ = json.load(open(cmp_p))
for lab, path, dk in [("auto_saturated", sat_p, "sat_depth"), ("auto_paced", paced_p, "paced_depth")]:
    skip, main = merge_rows(path)
    res["A_main_merge_fifo"][lab] = agg(main, "real_depth", "width_bits")
    tot_w = tot_kb = 0
    n = 0
    for p in cmp_.values():
        for e in p["edges"]:
            tot_w += e[dk]
            tot_kb += e[dk] * e["width_bits"] / 1024
            n += 1
    sk = agg(skip, "real_depth", "width_bits")
    # p7 classification-head FIFOs (post final conv, ahead of LabelSelect) are not in the sizer's plan
    head = [e for e in cmp_["7"]["edges"] if e["edge"].startswith(("ChannelwiseOp_hls_0->StreamingDataWidthConverter_rtl_3",
                                                                   "StreamingDataWidthConverter_rtl_3->LabelSelect"))]
    hw = sum(e[dk] for e in head)
    hkb = sum(e[dk] * e["width_bits"] / 1024 for e in head)
    res["B_all_non_skip"][lab + "_excl_p7_head"] = {"words": tot_w - sk["words"] - hw,
                                                    "kbit": round(tot_kb - sk["kbit"] - hkb, 1)}
    res["B_all_non_skip"][lab] = {"words": tot_w - sk["words"], "kbit": round(tot_kb - sk["kbit"], 1),
                                  "total_all_fifos_words": tot_w, "total_all_fifos_kbit": round(tot_kb, 1),
                                  "n_fifos": n}
json.dump(res, open(out_p, "w"), indent=1)
print(json.dumps(res, indent=1))

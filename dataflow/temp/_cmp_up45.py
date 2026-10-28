import json, re
O = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
B = O + "S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_20261009_225105/"
N = O + "S12_dense_256_u4_analytical_v2_ft15ep_int6_fps250_lat200_milpfold_8way_20261008_040600/"
def nodes(d, p):
    est = json.load(open(d + "report/estimate_layer_cycles.json"))
    return {k.split("_", 4)[-1] if False else k: v for k, v in est.items() if k.startswith("GenericPartition_%d_" % p)}
def short(k):
    return re.sub(r"^GenericPartition_\d+_GenericPartition_\d+_", "", k)
for p in (3, 4, 5, 6):
    nb, nn = nodes(B, p), nodes(N, p)
    sb = {short(k): v for k, v in nb.items()}
    sn = {short(k): v for k, v in nn.items()}
    print("=== partition", p, "bilinear n=%d nearest n=%d" % (len(sb), len(sn)))
    print(" only in bilinear:", {k: v for k, v in sb.items() if k not in sn})
    print(" only in nearest :", {k: v for k, v in sn.items() if k not in sb})
    print(" cycle diffs     :", {k: (sn[k], sb[k]) for k in sb if k in sn and sb[k] != sn[k]})
for d, tag in ((B, "bilinear"), (N, "nearest")):
    print("=====", tag, "fifo plans")
    for p in range(8):
        try:
            f = json.load(open(d + "fifo_plan_partition%d.json" % p))
        except Exception as e:
            print(p, "ERR", e); continue
        if isinstance(f, dict):
            keys = list(f.keys())[:6]
            vals = [v for v in f.values()]
            print(p, "keys:", keys, "n:", len(f))
            tot = 0
            big = []
            for k, v in f.items():
                dep = v.get("depth") if isinstance(v, dict) else v
                if isinstance(dep, (int, float)):
                    tot += dep
                    big.append((dep, k))
            print("   total depth", tot, "top:", sorted(big, reverse=True)[:4])

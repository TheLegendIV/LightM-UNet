import json, os, sys
O = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
runs = {
 "bilinear_8way":   "S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_20261009_225105",
 "nearest_v2_8way": "S12_dense_256_u4_analytical_v2_ft15ep_int6_fps250_lat200_milpfold_8way_20261008_040600",
 "nearest_p7mvu2":  "S12_dense_256_u4_u8in_int6_p7mvu2_milpfold_probe7_20261009_145623",
 "nearest_p7mvu":   "S12_dense_256_u4_u8in_int6_p7mvu_milpfold_probe7_20261009_104011",
}
def ld(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return None
for k, d in runs.items():
    r = O + d + "/report/"
    print("=====", k, "| files:", sorted(os.listdir(r))[:40] if os.path.isdir(r) else "NO REPORT")
    if not os.path.isdir(r):
        continue
    for f in sorted(os.listdir(r)):
        if f.startswith(("ooc_synth_partition_7", "rtlsim_partition_7")) or f in ("ooc_synth_partition_0.json",) and False:
            print(f, json.dumps(ld(r + f)))
    for f in ("ooc_synth_and_timing_per_partition.json", "rtlsim_per_partition.json"):
        j = ld(r + f)
        if j:
            p7 = j.get("partition_7") or j.get("7") or (j[-1] if isinstance(j, list) else None)
            print(f, "p7 ->", json.dumps(p7)[:600])
    est = ld(r + "estimate_layer_cycles.json")
    if est:
        p7 = {n: c for n, c in est.items() if n.startswith("GenericPartition_7")}
        print("p7 nodes (est cycles), max:", sorted(p7.items(), key=lambda t: -t[1])[:8], "| n nodes:", len(p7))
        print("global max node:", sorted(est.items(), key=lambda t: -t[1])[:3])
    print("net perf:", json.dumps(ld(r + "estimate_network_performance.json")))
    print("rtlsim perf:", json.dumps(ld(r + "rtlsim_performance.json")))

import json
import os

R = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_analytical_v2_ft15ep_int6_fps250_lat200_milpfold_8way_20261008_040600/report"
for name in ("rtlsim_performance.json", "estimate_network_performance.json"):
    p = os.path.join(R, name)
    print(name, json.load(open(p)) if os.path.isfile(p) else "MISSING")
agg = json.load(open(os.path.join(R, "ooc_synth_and_timing_per_partition.json")))
print("aggregate", json.dumps(agg["aggregate"]))
for k in sorted(k for k in agg if k.startswith("partition_")):
    r = agg[k]
    print(k, {x: r.get(x) for x in ("LUT", "FF", "DSP", "BRAM_18K", "URAM", "WNS", "fmax_mhz")})
rs = json.load(open(os.path.join(R, "rtlsim_per_partition.json")))
for k in sorted(rs):
    v = rs[k]
    print(k, "deadlock=%s" % v.get("deadlock"), v.get("latency_cycles"), "%s/%s" % (v.get("N_OUT_TXNS"), v.get("expected_out_txns")))

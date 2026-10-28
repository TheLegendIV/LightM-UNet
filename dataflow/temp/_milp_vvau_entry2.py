import json
d = json.load(open("/home/thelegendiv/finn/notebooks/enet/layer_bits_folding_S12_dense_256_u4_bilinear_analytical_v1_int6_fps250_lat200.json"))
e = d["per_layer"]["up4.main_up.1"]
print({k: v for k, v in e.items() if k in ("pe", "simd", "simd_swu", "swu_cycles", "fmpad_cycles", "cycles", "mvu_cycles", "node_type", "parallel_window", "depthwise")})
print(sorted(e.keys()))

import json
d = json.load(open("/home/thelegendiv/finn/notebooks/enet/layer_bits_folding_S12_dense_256_u4_bilinear_analytical_v1_int6_fps250_lat200.json"))
print(list(d.keys()))
pl = d["per_layer"]
for k, v in pl.items():
    if "main_up" in k or "up4" in k and "main" in k or v.get("node_type") == "depthwise_vvau_slot":
        print(k, json.dumps(v)[:500])
print("n per_layer", len(pl))
print([k for k in pl if k.startswith("up")][:30])

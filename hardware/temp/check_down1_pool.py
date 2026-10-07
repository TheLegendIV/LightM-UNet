import json
FOLDING_JSON = "/home/thelegendiv/finn/notebooks/enet/layer_bits_folding_S12_dense_256_u4_analytical_v1_int6_fps250_lat200_20261006.json"
with open(FOLDING_JSON) as f:
    d = json.load(f)
per_layer = d["per_layer"]
extra_nodes = d.get("extra_nodes", {})
print("per_layer['down1.pool']:", json.dumps(per_layer.get("down1.pool"), indent=2))
print("per_layer['down1.reduce.0']:", json.dumps(per_layer.get("down1.reduce.0"), indent=2))
print("extra_nodes keys with down1:", [k for k in extra_nodes if "down1" in k])
for k in extra_nodes:
    if "down1" in k:
        print(k, "->", extra_nodes[k])
op_map = d.get("op_type_by_role") or {}
print("op_type_by_role down1 entries:", {k: v for k, v in op_map.items() if "down1" in k} if op_map else "N/A key not found")
# search any dict value anywhere with a key resembling op type map
for key in d:
    if isinstance(d[key], dict):
        sample_keys = list(d[key].keys())[:3]
        if any("down1" in k for k in d[key] if isinstance(k, str)):
            matches = {k: v for k, v in d[key].items() if isinstance(k, str) and k.startswith("down1.")}
            if matches:
                print(f"--- top-level dict '{key}' has down1.* entries ---")
                print(json.dumps(matches, indent=2)[:3000])

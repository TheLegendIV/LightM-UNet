import json
p = r"c:\DEV\repos\LightM-UNet\MILP\artifacts\S12_dense_256_u4_analytical_v1\int6_fps250_lat200\layer_bits_folding_final.json"
d = json.load(open(p))
extra = d.get("extra_nodes", {})
print("extra_nodes count:", len(extra))
hi_pe = {k: v for k, v in extra.items() if isinstance(v, dict) and v.get("pe", 1) and v.get("pe", 1) > 1}
print("entries with pe>1:", len(hi_pe))
for k, v in list(hi_pe.items())[:40]:
    print(k, "pe=", v.get("pe"), "kind=", v.get("kind"), "stage=", v.get("stage"))

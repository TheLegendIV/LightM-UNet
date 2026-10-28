import json
FOLDING_JSON = "/home/thelegendiv/finn/notebooks/enet/layer_bits_folding_S12_dense_256_u4_analytical_v1_int6_fps250_lat200_20261006.json"
with open(FOLDING_JSON) as f:
    d = json.load(f)
for key in ("inter_block_fifos", "intra_block_fifos"):
    lst = d.get(key) or []
    print(f"=== {key}: {len(lst)} entries ===")
    for e in lst:
        prod = e.get("producer", "")
        cons = e.get("consumer", "")
        if "down1" in prod or "down1" in cons or "down2" in prod or "down2" in cons:
            print(" ", json.dumps(e))

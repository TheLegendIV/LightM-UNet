import json
d = json.load(open("/tmp/layer_bits_folding_final.json"))
for k in ("intra_block_fifos", "inter_block_fifos"):
    for f in d[k]:
        s = json.dumps(f)
        if "down1" in s or "down2" in s:
            print(k, f.get("name"), "|", f["producer"], "->", f["consumer"], "| depth", f.get("depth"), f.get("depth_alloc"), "skip", f.get("is_skip"))

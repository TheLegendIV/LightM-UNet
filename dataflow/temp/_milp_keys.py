import json
d = json.load(open("/tmp/layer_bits_folding_final.json"))
print(list(d.keys()))
pl = d["per_layer"]
k = [x for x in pl if x.startswith("down1")]
print(k)
print(json.dumps({x: pl[x] for x in k[:2]}))
for kk, v in d.items():
    if kk == "per_layer":
        continue
    print(kk, type(v).__name__, (list(v)[:4] if isinstance(v, (dict, list)) else v))

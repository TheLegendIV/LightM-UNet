import json
d = json.load(open("/tmp/layer_bits_folding_final.json"))
ex = d["extra_nodes"]
for k, v in ex.items():
    if "pad" in k or k.startswith("down1"):
        print(k, json.dumps(v))
print(d["dataflow_graph"]["op_types"] if isinstance(d["dataflow_graph"].get("op_types"), dict) else "")
print([e for e in d["dataflow_graph"]["edges"] if any("pad" in str(x) for x in e)][:6])

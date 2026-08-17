import json
import sys

out_dir = sys.argv[1]

with open(f"{out_dir}/report/op_and_param_counts.json") as f:
    d = json.load(f)
ks = list(d.keys())
print("num entries:", len(ks))
for k in ks[:20]:
    print(k, d[k])

print("\n--- estimate_layer_cycles.json (first 20) ---")
with open(f"{out_dir}/report/estimate_layer_cycles.json") as f:
    c = json.load(f)
for k in list(c.keys())[:20]:
    print(k, c[k])

print("\n--- estimate_network_performance.json ---")
with open(f"{out_dir}/report/estimate_network_performance.json") as f:
    p = json.load(f)
print(json.dumps(p, indent=2))

print("\n--- estimate_layer_resources.json (total) ---")
with open(f"{out_dir}/report/estimate_layer_resources.json") as f:
    r = json.load(f)
print(json.dumps(r.get("total", {}), indent=2))
print("num layer entries:", len(r))

"""One-off: build the combined ooc_synth_and_timing_per_partition.json aggregate
report from the individually-written per-partition JSONs, since the main
sequential script was killed early (to allow parallel per-partition runs)
before it reached its own aggregation step. Same aggregation logic as
finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_partition_synth.py.
"""
import json
import os

OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v2_20260918"
report_dir = os.path.join(OUTPUT_DIR, "report")

all_results = {}
for i in range(8):
    with open(os.path.join(report_dir, f"ooc_synth_partition_{i}.json")) as f:
        all_results[f"partition_{i}"] = json.load(f)

numeric_keys = set()
for res in all_results.values():
    for k, v in res.items():
        try:
            float(v)
            numeric_keys.add(k)
        except (TypeError, ValueError):
            pass
aggregate = {}
for k in numeric_keys:
    vals = [float(all_results[p][k]) for p in all_results if k in all_results[p]]
    if k.lower().startswith("fmax") or "period" in k.lower():
        aggregate[k + "_min_across_partitions"] = min(vals)
    else:
        aggregate[k + "_sum"] = sum(vals)
all_results["aggregate"] = aggregate

out_path = os.path.join(report_dir, "ooc_synth_and_timing_per_partition.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print("Wrote", out_path)
print(json.dumps(aggregate, indent=2))

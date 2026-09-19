import json
for label, path in [("auto", "/tmp/attrs_auto.json"), ("distributed", "/tmp/attrs_distributed.json")]:
    print(f"--- {label} ---")
    d = json.load(open(path))
    for r in d:
        if "Thresholding" in r["op_type"]:
            print(r["node_name"], r["op_type"], "depth_trigger_bram=", r["attrs"].get("depth_trigger_bram"),
                  "depth_trigger_uram=", r["attrs"].get("depth_trigger_uram"),
                  "runtime_writeable_weights=", r["attrs"].get("runtime_writeable_weights"))

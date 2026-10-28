import json

PATH = (
    "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
    "partition1_refix2_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_refix2/"
    "fifo_force_report_partition_1.json"
)

with open(PATH) as f:
    data = json.load(f)

for e in data:
    if e["fifo"] in ("StreamingFIFO_rtl_13", "StreamingFIFO_rtl_103"):
        print(json.dumps(e, indent=2))

print("total entries:", len(data))
forced = [e for e in data if e.get("forced_depth") not in (None, 2)]
print("entries with non-trivial forced_depth:", len(forced))
for e in forced:
    print(e["fifo"], e["producer_role"], "->", e["consumer_role"], e["forced_depth"])

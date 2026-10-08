import json

PATH = (
    "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
    "partition1_refix2_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_refix2/"
    "fifo_force_report_partition_1.json"
)

with open(PATH) as f:
    data = json.load(f)

for e in data:
    pr = e.get("producer_role") or ""
    cr = e.get("consumer_role") or ""
    if pr.startswith("down1.") or cr.startswith("down1.") or pr.startswith("down2.") or cr.startswith("down2."):
        print(f"{e['fifo']:28s} {pr:16s} -> {cr:16s} stock={e['stock_depth']} forced={e['forced_depth']}")

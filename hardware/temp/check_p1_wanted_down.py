import json

PATH = (
    "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
    "partition1_refix2_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_refix2/"
    "fifo_plan_partition1.json"
)

with open(PATH) as f:
    plan = json.load(f)

wanted = plan["wanted"]
wanted_by_producer = plan.get("wanted_by_producer", {})
for k, v in wanted.items():
    if "down1." in k or "down2." in k:
        print(k, "->", v)
print("--- wanted_by_producer ---")
for k, v in wanted_by_producer.items():
    if "down1." in k or "down2." in k:
        print(k, "->", v)

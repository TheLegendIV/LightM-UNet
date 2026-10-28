import json
from collections import defaultdict

PATH = (
    "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
    "partition1_refix2_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_refix2/"
    "fifo_plan_partition1.json"
)

with open(PATH) as f:
    plan = json.load(f)

role_of_node = plan["role_of_node"]  # node_name -> role string

# Print roles for the specific nodes of interest
interesting = [
    "Thresholding_rtl_0", "Thresholding_rtl_1", "MVAU_rtl_0", "MVAU_rtl_1",
    "Thresholding_rtl_25", "Thresholding_rtl_26", "Thresholding_rtl_27",
    "MVAU_rtl_16", "MVAU_rtl_17", "MVAU_rtl_18",
]
for n in interesting:
    print(n, "->", role_of_node.get(n))

print("--- reverse: role -> node, for down1./down2. roles ---")
by_role = defaultdict(list)
for node_name, role in role_of_node.items():
    if role and (role.startswith("down1.") or role.startswith("down2.")):
        by_role[role].append(node_name)
for role in sorted(by_role):
    nodes = by_role[role]
    flag = "  <-- DUPLICATE" if len(nodes) > 1 else ""
    print(f"{role:20s} -> {nodes}{flag}")

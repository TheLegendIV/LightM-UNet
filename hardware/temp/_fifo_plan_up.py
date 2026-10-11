import json, os
O = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
B = O + "S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_20261009_225105/"
print("intermediate_models:", sorted(os.listdir(B + "intermediate_models"))[:60])
for p in (5, 6):
    f = json.load(open(B + "fifo_plan_partition%d.json" % p))
    roles = f["role_of_node"]
    print("===== BILINEAR P%d" % p)
    print(" up roles:", {k: v for k, v in roles.items() if v.startswith("up")})
    for k, v in f["wanted"].items():
        if k.startswith("up"):
            print("  WANTED %-28s depth=%-6s skip=%-5s w=%s %s -> %s" % (k, v["depth"], v["is_skip"], v["width_bits"], v["producer_node"], v["consumer_node"]))
    un = [u for u in f["unresolved"] if u[0].startswith("up") or u[1].startswith("up")]
    print(" unresolved (up*):", un)

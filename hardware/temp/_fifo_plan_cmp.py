import json
O = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
B = O + "S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_20261009_225105/"
N = O + "S12_dense_256_u4_analytical_v2_ft15ep_int6_fps250_lat200_milpfold_8way_20261008_040600/"
for tag, d in (("BILINEAR", B), ("NEAREST", N)):
    for p in (4, 5, 6):
        f = json.load(open(d + "fifo_plan_partition%d.json" % p))
        print("=====", tag, "P%d" % p, {k: (len(v) if hasattr(v, "__len__") else v) for k, v in f.items()})
        print(" unresolved:", json.dumps(f["unresolved"])[:1500])
        w = f["wanted"]
        items = sorted(w.items(), key=lambda t: -(t[1] if isinstance(t[1], (int, float)) else 0))
        print(" wanted (top 12):", items[:12])
        print(" wanted_by_producer sample:", list(f["wanted_by_producer"].items())[:6])
        print(" roles sample:", list(f["role_of_node"].items())[:4])
        ups = {k: v for k, v in f["role_of_node"].items() if "up" in str(v).lower()}
        print(" up roles:", ups)

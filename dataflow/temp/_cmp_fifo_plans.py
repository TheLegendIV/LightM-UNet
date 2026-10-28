import json
base = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/"
old = json.load(open(base + "fifo_plan_partition1.json"))
new = json.load(open("/tmp/p1fix_bridge_out/fifo_plan_partition1.json"))
print("matched old/new:", len(old["wanted"]), len(new["wanted"]))
print("unresolved old/new:", len(old["unresolved"]), len(new["unresolved"]))
ok, nk = set(old["wanted"]), set(new["wanted"])
print("only old:", sorted(ok - nk))
print("only new:", sorted(nk - ok))
for k in sorted(ok & nk):
    if old["wanted"][k]["depth"] != new["wanted"][k]["depth"]:
        print("depth diff", k)
ro, rn = old["role_of_node"], new["role_of_node"]
print("role_of_node diffs (node: old -> new):")
for n in sorted(set(ro) | set(rn)):
    if ro.get(n) != rn.get(n):
        print(" ", n, ro.get(n), "->", rn.get(n))

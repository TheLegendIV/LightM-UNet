import json, glob, os
D = glob.glob("/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_u8in_int6_p1fix_milpfold_probe1_*")[0]
OLD = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/"
new = json.load(open(D + "/intermediate_models/supported_op_partitions/fifo_force_report_partition_1.json"))
old = json.load(open(OLD + "intermediate_models/supported_op_partitions/fifo_force_report_partition_1.json"))
def key(e):
    return (e["producer_node"], e["consumer_node"])
print("fifos old/new:", len(old), len(new))
print("unassigned new:")
for e in new:
    if e["forced_depth"] is None:
        print("  ", e["producer_node"], "->", e["consumer_node"], e["stock_depth"])
print("unassigned old:", sum(e["forced_depth"] is None for e in old))
print("thr_swap new:", [(e["producer_node"], e["consumer_node"], e["forced_depth"]) for e in new if e.get("forced_via_thr_role_swap")])
print("thr_swap old:", [(e["producer_node"], e["consumer_node"], e["forced_depth"]) for e in old if e.get("forced_via_thr_role_swap")])
od = {key(e): e["forced_depth"] for e in old}
nd = {key(e): e["forced_depth"] for e in new}
diff = [(k, od.get(k), nd.get(k)) for k in set(od) | set(nd) if od.get(k) != nd.get(k)]
print("forced-depth diffs old->new:", len(diff))
for d in sorted(diff, key=str)[:15]:
    print("  ", d)
print("skip depths >1000:", [(e["producer_node"], e["consumer_node"], e["forced_depth"]) for e in new if (e["forced_depth"] or 0) > 1000])
print("report files:", os.listdir(D + "/report"))
import onnx
m = onnx.load(D + "/intermediate_models/supported_op_partitions/partition_1.onnx")
names = [n.name for n in m.graph.node]
print("stitched model nodes:", len(names), "non-prefixed:", [n for n in names if not n.startswith("GenericPartition_1_")][:10])
print("metadata:", {p.key: p.value for p in m.metadata_props if p.key in ("vivado_stitch_proj", "wrapper_filename")})

import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

OUT = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_nearest_conv_upsample_256_w8_16_v4_trained_rtl_mvau_8way_full_256x256_20260928_043156"
parent = ModelWrapper(OUT + "/intermediate_models/dataflow_parent_built.onnx")
sdp_nodes = parent.get_nodes_by_op_type("StreamingDataflowPartition")
print(f"num partitions: {len(sdp_nodes)}")

# 1) check every partition's ORIGINAL kernel model has unique node names already
any_collision = False
for i, sdp in enumerate(sdp_nodes):
    fn = getCustomOp(sdp).get_nodeattr("model")
    m = ModelWrapper(fn)
    names = [n.name for n in m.graph.node]
    dupes = {n for n in names if names.count(n) > 1}
    status = "COLLISION" if dupes else "ok"
    if dupes:
        any_collision = True
    print(f"[{sdp.name}] file={fn} n_nodes={len(names)} unique={len(set(names))} {status} dupes={dupes}")

print("\nany_collision across original 8 partitions:", any_collision)

# 2) check for the literal name "IODMA_hls_0" (or IODMA_hls* prefix) pre-existing
#    in ANY partition BEFORE our script ever touched it (would indicate a real
#    pre-existing collision risk with our newly inserted node name)
print("\npre-existing IODMA-named nodes in any original partition:")
for i, sdp in enumerate(sdp_nodes):
    fn = getCustomOp(sdp).get_nodeattr("model")
    m = ModelWrapper(fn)
    hits = [n.name for n in m.graph.node if "IODMA" in n.name]
    if hits:
        print(f"  {sdp.name}: {hits}")
print("  (none printed above means none found)")

# 3) compare our two augmented copies against their pre-IODMA originals:
#    did GiveUniqueNodeNames rename any PRE-EXISTING node (not just add the new one)?
def compare(orig_fn, new_fn, label):
    mo = ModelWrapper(orig_fn)
    mn = ModelWrapper(new_fn)
    orig_names = [n.name for n in mo.graph.node]
    new_names = [n.name for n in mn.graph.node]
    new_dupes = {n for n in new_names if new_names.count(n) > 1}
    added = [n for n in new_names if n not in orig_names]
    removed = [n for n in orig_names if n not in new_names]
    renamed_existing = [n for n in orig_names if n not in new_names and n not in removed]
    print(f"\n[{label}] orig n={len(orig_names)} new n={len(new_names)} new_dupes={new_dupes}")
    print(f"  added (should be exactly the new IODMA node): {added}")
    print(f"  missing from new (should be empty -- would mean a pre-existing node got renamed away): {removed}")

fn0 = getCustomOp(sdp_nodes[0]).get_nodeattr("model")
fn7 = getCustomOp(sdp_nodes[-1]).get_nodeattr("model")
copy0 = OUT + "/iodma_driver_20260928_124546/kernel_models/GenericPartition_0_with_input_iodma.onnx"
copy7 = OUT + "/iodma_driver_20260928_124546/kernel_models/GenericPartition_7_with_output_iodma.onnx"
compare(fn0, copy0, "GenericPartition_0")
compare(fn7, copy7, "GenericPartition_7")

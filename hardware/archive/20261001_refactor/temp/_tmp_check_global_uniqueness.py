import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

OUT = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_nearest_conv_upsample_256_w8_16_v4_trained_rtl_mvau_8way_full_256x256_20260928_043156"
parent = ModelWrapper(OUT + "/intermediate_models/dataflow_parent_built.onnx")
sdp_nodes = parent.get_nodes_by_op_type("StreamingDataflowPartition")

all_names_global = []
per_partition = {}
for i, sdp in enumerate(sdp_nodes):
    fn = getCustomOp(sdp).get_nodeattr("model")
    m = ModelWrapper(fn)
    names = [n.name for n in m.graph.node]
    per_partition[sdp.name] = names
    expected_prefix = f"{sdp.name}_"
    n_prefixed = sum(1 for n in names if n.startswith(expected_prefix))
    n_not_prefixed = [n for n in names if not n.startswith(expected_prefix)]
    print(f"[{sdp.name}] n={len(names)} prefixed_with_own_name={n_prefixed}/{len(names)} "
          f"not_prefixed={n_not_prefixed if len(n_not_prefixed) < 10 else n_not_prefixed[:10] + ['...']}")
    all_names_global.extend(names)

print("\n=== GLOBAL uniqueness across all 8 partitions combined ===")
dupes_global = {n for n in all_names_global if all_names_global.count(n) > 1}
print("total nodes across all partitions:", len(all_names_global))
print("unique names across all partitions:", len(set(all_names_global)))
print("GLOBAL duplicate names (should be empty for a clean combine):", dupes_global)
if dupes_global:
    for d in dupes_global:
        owners = [p for p, names in per_partition.items() if d in names]
        print(f"  '{d}' appears in: {owners}")

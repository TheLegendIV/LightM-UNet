from qonnx.core.modelwrapper import ModelWrapper
from finn_stage_partition import compute_8way_boundaries

m = ModelWrapper(
    "finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_preamble_20260913_091715/"
    "intermediate_models/step_enet_convert_to_hw.onnx"
)
b = compute_8way_boundaries(m)
print(b)

nodes = list(m.graph.node)
edges = [0] + [b[k] for k in ("down1_start", "down2_start", "q2_start", "q3_start", "q4_start", "up4_start", "up5_start")] + [len(nodes)]
for i in range(8):
    lo, hi = edges[i], edges[i + 1]
    last_op = nodes[hi - 1].op_type if hi > lo else None
    print(f"partition {i}: [{lo}, {hi}) size={hi - lo} last_node_op={last_op}")

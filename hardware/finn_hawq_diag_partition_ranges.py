"""Diagnostic: for the HAWQ preamble's step_enet_convert_to_hw.onnx (post-
convert_to_hw, PRE-partition, 594 nodes), print the ordered list of
weight-bearing node (index, op_type, name), and figure out exactly how
many of them fall in each of the 8-way partition_id node-index ranges
(from assign_stage_partition_ids_8way's own boundary log:
down1=7 down2=81 q2=186 q3=291 q4=396 up4=500 up5=551), to correlate 1:1
against hardware/outputs/finn_exports/quantEnet_s19_hawq_block_int8_conv_order.json's
ordered logical-name list (129 entries total).

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \
        /home/thelegendiv/finn/notebooks/enet/finn_hawq_diag_partition_ranges.py <preamble_output_dir>
"""
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402

WEIGHT_OP_TYPES = {"MatrixVectorActivation", "MVAU", "MVAU_hls", "MVAU_rtl", "VVAU", "VVAU_hls", "VVAU_rtl"}
POOL_OP_TYPES = {"StreamingMaxPool", "StreamingMaxPool_hls", "StreamingMaxPool_rtl"}

BOUNDARIES = {
    "initial": (0, 7),
    "stage1(down1+regular1)": (7, 81),
    "p2_stage23_q1": (81, 186),
    "p3_stage23_q2": (186, 291),
    "p4_stage23_q3": (291, 396),
    "p5_stage23_q4": (396, 500),
    "stage4(up4+regular4)": (500, 551),
    "stage5(up5+regular5+final)": (551, 594),
}


def main():
    preamble_dir = sys.argv[1]
    ckpt = os.path.join(preamble_dir, "intermediate_models", "step_enet_convert_to_hw.onnx")
    model = ModelWrapper(ckpt)
    print(f"Loaded {ckpt}: {len(model.graph.node)} nodes")

    conv_order_file = os.path.join(
        "/home/thelegendiv/finn/notebooks/enet", "quantEnet_s19_hawq_block_int8_conv_order.json"
    )
    with open(conv_order_file) as f:
        logical = json.load(f)
    print(f"Logical name list: {len(logical)} entries")

    # ordered (index, op_type, name) for weight-bearing (conv-like or pool) nodes
    weight_like = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type in WEIGHT_OP_TYPES or node.op_type in POOL_OP_TYPES:
            weight_like.append((idx, node.op_type, node.name))

    print(f"\nFound {len(weight_like)} weight-like nodes (conv/pool) in full pre-partition graph")
    if len(weight_like) != len(logical):
        print(f"WARNING: count mismatch vs logical list ({len(weight_like)} vs {len(logical)})")

    # zip positionally against the full logical list (best-effort, printed for inspection)
    print(f"\n{'idx':5s} {'op_type':22s} {'node name':30s} {'logical_name (positional)':30s}")
    for i, (idx, op_type, name) in enumerate(weight_like):
        logical_name = logical[i]["logical_name"] if i < len(logical) else "<OUT OF RANGE>"
        print(f"{idx:5d} {op_type:22s} {name:30s} {logical_name:30s}")

    # count how many weight-like nodes fall in each partition's raw node-index range
    print("\nWeight-like node counts per partition node-index range:")
    for label, (lo, hi) in BOUNDARIES.items():
        cnt = sum(1 for idx, _, _ in weight_like if lo <= idx < hi)
        print(f"  {label:30s} range=[{lo},{hi}) weight_like_count={cnt}")


if __name__ == "__main__":
    main()

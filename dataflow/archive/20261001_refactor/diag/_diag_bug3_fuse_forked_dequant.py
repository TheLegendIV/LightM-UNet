"""Diagnostic (not part of the build pipeline): re-run
step_fuse_forked_dequant_into_duplicate_threshold's inner logic with debug
prints, directly on the real step_absorb_leftover_scale_before_matmul.onnx
checkpoint, to see exactly why the FMPadding_Pixel input-repointing fails
for the two main_up chains (Bug #3 root cause investigation)."""
import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
import numpy as np

from finn_enet_build_fixups import _walk_back_through_transpose_and_affine

BASE = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_preamble_20260908_223941/intermediate_models/"
model = ModelWrapper(BASE + "step_absorb_leftover_scale_before_matmul.onnx")
graph = model.graph

blocking_ops = ("FMPadding_Pixel", "Im2Col", "MaxPoolNHWC")
for node in list(graph.node):
    if node.op_type not in ("Mul", "Add"):
        continue
    if model.is_fork_node(node) or model.is_join_node(node):
        continue
    consumer = model.find_consumer(node.output[0])
    if consumer is None or consumer.op_type not in blocking_ops:
        continue
    print("=" * 60)
    print("node:", node.name, node.op_type, "output:", node.output[0])
    print("consumer:", consumer.name, consumer.op_type, "inputs:", list(consumer.input))
    walk = _walk_back_through_transpose_and_affine(model, node.output[0])
    if walk is None:
        print("  walk -> None")
        continue
    mt, scale, bias, transposes, chain_nodes = walk
    print("  mt:", mt.name, "scale:", scale, "bias:", bias)
    print("  transposes:", [t.name for t in transposes])
    print("  chain_nodes:", [c.name for c in chain_nodes])
    print("  is node.output[0] in consumer.input?", node.output[0] in list(consumer.input))

import dataclasses
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
os.environ.setdefault("FINN_ROOT", "/home/thelegendiv/finn")

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
BUILD_TAG = "debug_p1_join_v3"
os.environ["FINN_BUILD_DIR"] = os.path.join(ENET_DIR, "finn_build_tmp", BUILD_TAG)

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from finn.builder.build_dataflow_steps import step_specialize_layers, step_target_fps_parallelization

import finn_enet_ip_build_partitioned_8way as base
from finn_partition_build_steps import step_create_dataflow_partition_multi
from finn_s12_build_steps import load_partition_logical_names, WEIGHT_OP_TYPES

PREAMBLE_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", "S12_dense_256_u4_analytical_v1_ft15ep_preamble_20261005_220558")
CONV_ORDER = os.path.join(ENET_DIR, "quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_conv_order.json")
PART_IDX = 1
OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"debug_{BUILD_TAG}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)
flat_ckpt = os.path.join(PREAMBLE_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
parent = step_create_dataflow_partition_multi(ModelWrapper(flat_ckpt), cfg)
sdp_nodes = parent.get_nodes_by_op_type("StreamingDataflowPartition")
sdp = [n for n in sdp_nodes if n.name == f"GenericPartition_{PART_IDX}"][0]
kernel_model = ModelWrapper(getCustomOp(sdp).get_nodeattr("model"))
kernel_model = step_specialize_layers(kernel_model, cfg)
kernel_model = kernel_model.transform(GiveUniqueNodeNames(sdp.name + "_"))
kernel_model = kernel_model.transform(GiveReadableTensorNames())
kernel_model = step_target_fps_parallelization(kernel_model, cfg)
kernel_model = kernel_model.transform(GiveUniqueNodeNames())

adds = [n for n in kernel_model.graph.node if n.op_type.startswith("AddStreams")]
add0 = adds[0]
print("add0:", add0.name)
for inp in add0.input:
    print(f"  --- walking back from input {inp} ---")
    n = kernel_model.find_producer(inp)
    depth = 0
    while n is not None and depth < 8:
        print("   ", n.name, n.op_type)
        n = kernel_model.find_producer(n.input[0])
        depth += 1

print()
print("MVAU_rtl_0 consumer chain (shortcut_proj):")
mvau0 = [n for n in kernel_model.graph.node if n.name == "MVAU_rtl_0"][0]
c = kernel_model.find_consumer(mvau0.output[0])
depth = 0
while c is not None and depth < 8:
    print("   ", c.name, c.op_type)
    c = kernel_model.find_consumer(c.output[0])
    depth += 1

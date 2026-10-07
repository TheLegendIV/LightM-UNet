import dataclasses
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
os.environ.setdefault("FINN_ROOT", "/home/thelegendiv/finn")

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
BUILD_TAG = "debug_p1_join_v2"
os.environ["FINN_BUILD_DIR"] = os.path.join(ENET_DIR, "finn_build_tmp", BUILD_TAG)

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from finn.builder.build_dataflow_steps import step_specialize_layers, step_target_fps_parallelization

import finn_enet_ip_build_partitioned_8way as base
from finn_partition_build_steps import step_create_dataflow_partition_multi
from finn_s12_build_steps import (
    load_partition_logical_names, WEIGHT_OP_TYPES, _resolve_folding_entry, _stage_leaf,
    _claimed_mvau_names, _find_join_thresholds, _previous_block_name,
)

PREAMBLE_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", "S12_dense_256_u4_analytical_v1_ft15ep_preamble_20261005_220558")
CONV_ORDER = os.path.join(ENET_DIR, "quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_conv_order.json")
FOLDING_JSON = os.path.join(ENET_DIR, "layer_bits_folding_S12_dense_256_u4_analytical_v1_int6_fps250_lat200_20261006.json")
PART_IDX = 1
OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"debug_{BUILD_TAG}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(FOLDING_JSON) as f:
    folding_block = json.load(f)
per_layer = folding_block["per_layer"]

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

logical = load_partition_logical_names(PREAMBLE_DIR, CONV_ORDER)
logical_names = logical[PART_IDX][0]
prev_block = _previous_block_name(CONV_ORDER, logical_names[0])
print("prev_block:", prev_block)

weight_nodes = [n for n in kernel_model.graph.node if n.op_type in WEIGHT_OP_TYPES]
print("num weight nodes:", len(weight_nodes), "num logical names:", len(logical_names))
for node, logical_name in list(zip(weight_nodes, logical_names))[:6]:
    entry, json_key = _resolve_folding_entry(logical_name, per_layer)
    print(f"  {node.name:12s} logical={logical_name:25s} resolved_key={json_key}")

claimed = _claimed_mvau_names(kernel_model, logical_names, per_layer)
print("claimed count:", len(claimed))
print("claimed names:", sorted(claimed)[:10])

joins = _find_join_thresholds(kernel_model, logical_names, prev_block, claimed)
print("joins found:", len(joins))
for name, (block, kind) in joins.items():
    print(" ", name, "->", block, kind)

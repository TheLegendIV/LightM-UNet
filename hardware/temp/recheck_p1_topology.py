import dataclasses
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
os.environ.setdefault("FINN_ROOT", "/home/thelegendiv/finn")

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
BUILD_TAG = "debug_p1_topo_recheck_v1"
os.environ["FINN_BUILD_DIR"] = os.path.join(ENET_DIR, "finn_build_tmp", BUILD_TAG)

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames

import finn_enet_ip_build_partitioned_8way as base
from finn_partition_build_steps import step_create_dataflow_partition_multi
from finn_s12_build_steps import (
    step_specialize_layers, step_target_fps_parallelization,
    load_partition_logical_names, _resolve_folding_entry,
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
logical = load_partition_logical_names(PREAMBLE_DIR, CONV_ORDER)
logical_names = logical[PART_IDX][0]

kernel_model = ModelWrapper(getCustomOp(sdp).get_nodeattr("model"))
kernel_model = step_specialize_layers(kernel_model, cfg)
kernel_model = kernel_model.transform(GiveUniqueNodeNames(sdp.name + "_"))
kernel_model = kernel_model.transform(GiveReadableTensorNames())
kernel_model = step_target_fps_parallelization(kernel_model, cfg)
kernel_model = kernel_model.transform(GiveUniqueNodeNames())

WEIGHT_OP_TYPES = {"MVAU_rtl", "MVAU_hls", "VVAU_rtl", "VVAU_hls"}
weight_nodes = [n for n in kernel_model.graph.node if n.op_type in WEIGHT_OP_TYPES]

name_of = {}
for node, ln in zip(weight_nodes, logical_names):
    entry, json_key = _resolve_folding_entry(ln, per_layer)
    name_of[node.name] = f"{ln} [{json_key}]" if json_key else f"{ln} <NO MATCH>"

# Find the 1st AddStreams (down1's add)
adds = [n for n in kernel_model.graph.node if n.op_type.startswith("AddStreams")]
add0 = adds[0]
print("=== add0 inputs, backward walk ===")
for i, inp in enumerate(add0.input):
    print(f"-- input {i} --")
    cur = kernel_model.find_producer(inp)
    for hop in range(10):
        if cur is None:
            break
        label = name_of.get(cur.name, "")
        print(f"   {cur.op_type:30s} {cur.name:25s} {label}")
        if cur.op_type in WEIGHT_OP_TYPES and hop > 0:
            # keep going a bit further to also see one more hop back for context
            pass
        nxt = kernel_model.find_producer(cur.input[0]) if len(cur.input) > 0 else None
        cur = nxt

print("=== Dup forward walk (both branches) ===")
dups = [n for n in kernel_model.graph.node if n.op_type.startswith("DuplicateStreams")]
print("num DuplicateStreams in partition:", len(dups))
dup0 = dups[0]
print("first dup:", dup0.name)
for oi, out_t in enumerate(dup0.output):
    print(f"-- dup output {oi} --")
    cur = kernel_model.find_consumer(out_t)
    for hop in range(10):
        if cur is None:
            break
        label = name_of.get(cur.name, "")
        print(f"   {cur.op_type:30s} {cur.name:25s} {label}")
        nxt = kernel_model.find_consumer(cur.output[0]) if len(cur.output) > 0 else None
        cur = nxt

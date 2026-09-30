"""Cheap, no-Vivado extraction of FINN's auto-fold PE/SIMD folding choice
(step_target_fps_parallelization, target_fps=250 -- same as the
baseline_both_off_autofold partition-2 build) for ALL 8 partitions of the
8-way split, not just partition 2.

Only runs: step_create_dataflow_partition_multi -> (per partition)
step_specialize_layers -> step_target_fps_parallelization. No hw_codegen,
no ipgen, no FIFO depths, no stitched IP, no OOC synth -- this never touches
Vivado/Vitis HLS, so it's fast (graph transforms + analytical cycle
estimates only).

Output schema matches dump_autofold_config.py / the MILP-bridged
hawq_folding_config_partition2.json files (dict of node_name -> {"PE": int,
"SIMD": int}), one JSON file per partition.

Usage: python3 dump_autofold_config_all_partitions.py <hawq_preamble_output_dir> <tag> <output_dir>
"""
import sys
import os
import json
import dataclasses

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

if len(sys.argv) < 4:
    print("Usage: dump_autofold_config_all_partitions.py <hawq_preamble_output_dir> <tag> <output_dir>")
    sys.exit(1)
HAWQ_PREAMBLE_DIR = sys.argv[1]
TAG = sys.argv[2]
OUTPUT_DIR = sys.argv[3]

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers,
    step_target_fps_parallelization,
)
from finn.util.fpgadataflow import is_fpgadataflow_node  # noqa: E402

SOURCE_CKPT = os.path.join(HAWQ_PREAMBLE_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")

os.makedirs(OUTPUT_DIR, exist_ok=True)
assert os.path.exists(SOURCE_CKPT), f"missing {SOURCE_CKPT}"

cfg = base.cfg_stitched_ip_partitioned_8way
print(f"Tag         : {TAG}")
print(f"target_fps  : {cfg.target_fps}")
print(f"Source ckpt : {SOURCE_CKPT}")
print(f"Output dir  : {OUTPUT_DIR}")

flat_model = ModelWrapper(SOURCE_CKPT)
print("Running step_create_dataflow_partition_multi (re-split, deterministic)...")
parent_model = step_create_dataflow_partition_multi(flat_model, cfg)

sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
print(f"Got {len(sdp_nodes)} partitions: {[n.name for n in sdp_nodes]}")

for idx, sdp_node in enumerate(sdp_nodes):
    sdp_inst = getCustomOp(sdp_node)
    partition_model_fn = sdp_inst.get_nodeattr("model")
    prefix = sdp_node.name + "_"
    print(f"\n=== Partition {idx} -> {sdp_node.name} -> {partition_model_fn} ===")
    kernel_model = ModelWrapper(partition_model_fn)
    print(f"Loaded raw partition {idx} model: {len(kernel_model.graph.node)} nodes")

    kernel_model = step_specialize_layers(kernel_model, cfg)
    kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
    kernel_model = kernel_model.transform(GiveReadableTensorNames())

    kernel_model = step_target_fps_parallelization(kernel_model, cfg)

    config = {"Defaults": {}}
    for node in kernel_model.graph.node:
        if not is_fpgadataflow_node(node):
            continue
        inst = getCustomOp(node)
        entry = {}
        for attr in ("PE", "SIMD"):
            try:
                entry[attr] = inst.get_nodeattr(attr)
            except Exception:
                pass
        if entry:
            config[node.name] = entry

    out_json = os.path.join(OUTPUT_DIR, f"autofold_config_partition{idx}.json")
    with open(out_json, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {len(config) - 1} node folding entries to {out_json}")

print("\nDone -- all 8 partitions' auto-fold configs written.")

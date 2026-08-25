"""Analytical (no-Vivado) BRAM/LUT comparison of partition 2 between the
REAL leaky S19 model (original 8-way run) and the ReLU-only variant
(finn_relu_only_preamble.py's output), both taken through the exact same
cheap pipeline: re-split (step_create_dataflow_partition_multi) ->
step_specialize_layers -> step_target_fps_parallelization ->
step_apply_folding_config -> step_minimize_bit_width -> FINN's own
res_estimation analysis pass (no hw_codegen/ipgen/synthesis -- this is the
same closed-form per-node resource-estimation FINN uses for its own
"estimate_layer_resources.json" report, just invoked directly here so both
variants can be compared side by side before committing to a ~5hr real OOC
synth run).

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \
        /home/thelegendiv/finn/notebooks/enet/finn_relu_vs_leaky_partition2_estimate.py
"""
import functools
import os
import sys
import dataclasses

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402

import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
)
from finn.analysis.fpgadataflow.res_estimation import res_estimation  # noqa: E402

PARTITION_IDX = 2

LEAKY_RUN_DIR = os.path.join(
    base.ENET_DIR, "finn_deployment_outputs",
    "stitched_ip_partitioned8way_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260820_101224",
)
RELU_RUN_DIR = os.path.join(
    base.ENET_DIR, "finn_deployment_outputs", "relu_only_preamble_20260824_195233",
)

VARIANTS = {
    "leaky (real S19)": os.path.join(LEAKY_RUN_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx"),
    "relu-only": os.path.join(RELU_RUN_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx"),
}


def build_partition2(source_ckpt: str, cfg):
    flat_model = ModelWrapper(source_ckpt)
    parent_model = step_create_dataflow_partition_multi(flat_model, cfg)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    sdp_node = sdp_nodes[PARTITION_IDX]
    sdp_inst = getCustomOp(sdp_node)
    partition_model_fn = sdp_inst.get_nodeattr("model")
    prefix = sdp_node.name + "_"
    kernel_model = ModelWrapper(partition_model_fn)

    kernel_model = step_specialize_layers(kernel_model, cfg)
    kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
    kernel_model = kernel_model.transform(GiveReadableTensorNames())
    kernel_model = step_target_fps_parallelization(kernel_model, cfg)
    kernel_model = step_apply_folding_config(kernel_model, cfg)
    kernel_model = step_minimize_bit_width(kernel_model, cfg)
    return kernel_model


def summarize(kernel_model, label: str, cfg):
    res = kernel_model.analysis(functools.partial(res_estimation, fpgapart=cfg._resolve_fpga_part()))
    total_bram = 0.0
    thr_bram = 0.0
    thr_count = 0
    op_counts: dict[str, int] = {}
    for node in kernel_model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    for node_name, r in res.items():
        bram = r.get("BRAM_18K", 0)
        total_bram += bram
        if "Thresholding" in node_name:
            thr_bram += bram
            thr_count += 1
    print(f"\n=== {label} ===")
    print(f"  node count: {len(kernel_model.graph.node)}  op_type counts: {dict(sorted(op_counts.items()))}")
    print(f"  Thresholding_rtl nodes: {thr_count}")
    print(f"  Total estimated BRAM_18K: {total_bram:.1f}")
    print(f"  Thresholding_rtl estimated BRAM_18K: {thr_bram:.1f} ({100*thr_bram/total_bram if total_bram else 0:.1f}% of total)")
    return total_bram, thr_bram, thr_count


if __name__ == "__main__":
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir="/tmp/_estimate_scratch")
    results = {}
    for label, ckpt in VARIANTS.items():
        print(f"\nBuilding partition {PARTITION_IDX} for variant '{label}' from {ckpt} ...")
        km = build_partition2(ckpt, cfg)
        results[label] = summarize(km, label, cfg)

    print("\n=== Comparison ===")
    for label, (total_bram, thr_bram, thr_count) in results.items():
        print(f"  {label}: total_BRAM_18K={total_bram:.1f}  thresholding_BRAM_18K={thr_bram:.1f}  n_thresholds={thr_count}")

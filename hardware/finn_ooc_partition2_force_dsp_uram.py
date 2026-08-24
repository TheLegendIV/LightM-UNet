"""Rebuild S19 8-way-partitioned build's Partition 2 (the single largest
partition in the real, already-completed 8-way OOC-synth run -- 179,084 LUT
/ 146,866 FF / real fmax 127.6 MHz / 977.2 img/s rtlsim, see hardware/results.csv
config=s19_double_mid_partition_2_ooc_synth) from scratch with resType="dsp"
forced on every MVAU node and ram_style="ultra" forced on every
ConvolutionInputGenerator (SWU) node, all the way through REAL Vivado OOC
synthesis -- the same forcing technique already validated on the toy
single-block probe (finn_enet_ip_build_s19_single_block.py /
config=s19_single_block_force_dsp_uram_ooc_synth), but this time on an
actual full-size partition from the real 8-way split instead of a synthetic
minimal model.

Why not just reuse the original partition_2.onnx directly: that file (in
the original run's intermediate_models/supported_op_partitions/) was
overwritten IN PLACE by _build_one_partition (finn_partition_build_steps.py)
with the fully hw_ipgen'd + stitched-IP result -- by the time that build
finished, resType/ram_style were already baked into generated Verilog, so
there is nothing left to force. Instead, this script starts one step
earlier: from assign_stage_partition_ids_8way.onnx (the full, flat,
not-yet-split graph with partition_id already assigned on every node,
still present in the original run's intermediate_models/), re-runs
step_create_dataflow_partition_multi (deterministic given the same
partition_id assignment + same cfg) to regenerate a byte-for-byte
equivalent fresh partition_2 sub-model, then proceeds through the exact
same per-partition step sequence _build_one_partition uses
(specialize_layers -> target_fps_parallelization -> apply_folding_config ->
minimize_bit_width) but inserts the DSP/URAM-forcing step right before
hw_codegen, then continues through hw_ipgen -> set_fifo_depths ->
CreateStitchedIP -> measure_rtlsim_performance -> out_of_context_synthesis.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \
        /home/thelegendiv/finn/notebooks/enet/finn_ooc_partition2_force_dsp_uram.py
"""

import os
import sys
import dataclasses
from datetime import datetime

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
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
    step_measure_rtlsim_performance,
    step_out_of_context_synthesis,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP  # noqa: E402

PARTITION_IDX = 2

ORIG_RUN_DIR = os.path.join(
    base.ENET_DIR,
    "finn_deployment_outputs",
    "stitched_ip_partitioned8way_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260820_101224",
)
SOURCE_CKPT = os.path.join(ORIG_RUN_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(
    base.ENET_DIR, "finn_deployment_outputs", f"partition2_force_dsp_uram_{timestamp}"
)


def step_force_dsp_and_uram(model):
    n_dsp = 0
    n_uram = 0
    for node in model.graph.node:
        if "MVAU" in node.op_type:
            getCustomOp(node).set_nodeattr("resType", "dsp")
            n_dsp += 1
        elif "ConvolutionInputGenerator" in node.op_type:
            getCustomOp(node).set_nodeattr("ram_style", "ultra")
            n_uram += 1
    print(
        f"[step_force_dsp_and_uram] forced resType=dsp on {n_dsp} MVAU node(s), "
        f"ram_style=ultra on {n_uram} ConvolutionInputGenerator node(s)"
    )
    return model


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)

    print(f"Source checkpoint: {SOURCE_CKPT}")
    print(f"Output dir       : {OUTPUT_DIR}")

    flat_model = ModelWrapper(SOURCE_CKPT)

    print("Running step_create_dataflow_partition_multi (re-split, deterministic)...")
    parent_model = step_create_dataflow_partition_multi(flat_model, cfg)

    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    print(f"Got {len(sdp_nodes)} partitions: {[n.name for n in sdp_nodes]}")
    sdp_node = sdp_nodes[PARTITION_IDX]
    sdp_inst = getCustomOp(sdp_node)
    partition_model_fn = sdp_inst.get_nodeattr("model")
    print(f"Partition {PARTITION_IDX} -> {sdp_node.name} -> {partition_model_fn}")

    prefix = sdp_node.name + "_"
    kernel_model = ModelWrapper(partition_model_fn)
    print(f"Loaded raw partition {PARTITION_IDX} model: {len(kernel_model.graph.node)} nodes")

    print("Running: step_specialize_layers")
    kernel_model = step_specialize_layers(kernel_model, cfg)
    kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
    kernel_model = kernel_model.transform(GiveReadableTensorNames())
    print("Running: step_target_fps_parallelization")
    kernel_model = step_target_fps_parallelization(kernel_model, cfg)
    print("Running: step_apply_folding_config")
    kernel_model = step_apply_folding_config(kernel_model, cfg)
    print("Running: step_minimize_bit_width")
    kernel_model = step_minimize_bit_width(kernel_model, cfg)
    print("Running: step_force_dsp_and_uram")
    kernel_model = step_force_dsp_and_uram(kernel_model)
    print("Running: step_hw_codegen")
    kernel_model = step_hw_codegen(kernel_model, cfg)
    print("Running: step_hw_ipgen")
    kernel_model = step_hw_ipgen(kernel_model, cfg)
    print("Running: step_set_fifo_depths")
    kernel_model = step_set_fifo_depths(kernel_model, cfg)
    print("Running: CreateStitchedIP")
    kernel_model = kernel_model.transform(
        CreateStitchedIP(cfg._resolve_fpga_part(), cfg.synth_clk_period_ns, prefix.rstrip("_"), False)
    )
    final_fn = os.path.join(OUTPUT_DIR, "partition2_force_dsp_uram_stitched.onnx")
    kernel_model.save(final_fn)

    print("Running: step_measure_rtlsim_performance")
    kernel_model = step_measure_rtlsim_performance(kernel_model, cfg)
    kernel_model.save(final_fn)

    print("Running: step_out_of_context_synthesis")
    kernel_model = step_out_of_context_synthesis(kernel_model, cfg)
    kernel_model.save(final_fn)

    print(f"Done. Reports in {OUTPUT_DIR}/report")

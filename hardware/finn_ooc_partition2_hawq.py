"""Rebuild S19 HAWQ per-block-bit-width Partition 2 (down2 + stage2.0..stage2.5
partial, see finn_hawq_folding_bridge.py's docstring for the exact node-index
range) from the fresh HAWQ preamble checkpoint
(finn_hawq_preamble.py's assign_stage_partition_ids_8way.onnx), applying the
ILP-derived per-layer PE/SIMD folding config bridged from
compression/hawq/folding_block_s19.json (via finn_hawq_folding_bridge.py's
already-generated hawq_folding_config_partition2.json), all the way through
REAL Vivado OOC synthesis.

Mirrors hardware/finn_ooc_partition2_force_dsp_uram.py's per-partition rebuild
template (specialize_layers -> target_fps_parallelization ->
apply_folding_config -> minimize_bit_width -> hw_codegen -> hw_ipgen ->
set_fifo_depths -> CreateStitchedIP -> measure_rtlsim_performance ->
out_of_context_synthesis), but WITHOUT the DSP/URAM-forcing step (that was a
separate, unrelated resource-type experiment) -- this run's whole point is to
measure the HAWQ per-block bit-width + ILP folding config's actual resource
usage against the uniform-bit-width baseline
(config=s19_double_mid_partition_2_ooc_synth in hardware/results.csv).

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \
        /home/thelegendiv/finn/notebooks/enet/finn_ooc_partition2_hawq.py
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

# Fixed to the already-completed HAWQ preamble run (finn_hawq_preamble.py) --
# hardcoded (not re-discovered) since that run is done and its
# assign_stage_partition_ids_8way.onnx / hawq_folding_config_partition2.json
# are both already on disk.
HAWQ_PREAMBLE_DIR = os.path.join(
    base.ENET_DIR, "finn_deployment_outputs", "hawq_preamble_20260824_210157",
)
SOURCE_CKPT = os.path.join(HAWQ_PREAMBLE_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
BRIDGED_FOLDING_CONFIG = os.path.join(HAWQ_PREAMBLE_DIR, "hawq_folding_config_partition2.json")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(
    base.ENET_DIR, "finn_deployment_outputs", f"partition2_hawq_block_{timestamp}"
)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    assert os.path.exists(SOURCE_CKPT), f"missing {SOURCE_CKPT}"
    assert os.path.exists(BRIDGED_FOLDING_CONFIG), f"missing {BRIDGED_FOLDING_CONFIG}"

    cfg = dataclasses.replace(
        base.cfg_stitched_ip_partitioned_8way,
        output_dir=OUTPUT_DIR,
        folding_config_file=BRIDGED_FOLDING_CONFIG,
    )

    print(f"Source checkpoint : {SOURCE_CKPT}")
    print(f"Folding config    : {BRIDGED_FOLDING_CONFIG}")
    print(f"Output dir        : {OUTPUT_DIR}")

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
    print("Running: step_apply_folding_config (HAWQ bridged config)")
    kernel_model = step_apply_folding_config(kernel_model, cfg)
    print("Running: step_minimize_bit_width")
    kernel_model = step_minimize_bit_width(kernel_model, cfg)
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
    final_fn = os.path.join(OUTPUT_DIR, "partition2_hawq_block_stitched.onnx")
    kernel_model.save(final_fn)

    print("Running: step_measure_rtlsim_performance")
    kernel_model = step_measure_rtlsim_performance(kernel_model, cfg)
    kernel_model.save(final_fn)

    print("Running: step_out_of_context_synthesis")
    kernel_model = step_out_of_context_synthesis(kernel_model, cfg)
    kernel_model.save(final_fn)

    print(f"Done. Reports in {OUTPUT_DIR}/report")

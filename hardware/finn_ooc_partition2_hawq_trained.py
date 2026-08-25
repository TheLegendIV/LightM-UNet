"""Rebuild the REAL TRAINED per-block HAWQ S19 checkpoint's Partition 2 (down2
+ stage2.0..stage2.5 partial -- see finn_hawq_folding_bridge.py's docstring
for the exact node-index range) from finn_hawq_preamble_trained.py's
assign_stage_partition_ids_8way.onnx, applying the ILP-derived per-layer
PE/SIMD folding config bridged from compression/hawq/folding_block_s19.json
(via finn_hawq_folding_bridge.py's hawq_folding_config_partition2.json,
regenerated against THIS preamble run), all the way through REAL Vivado OOC
synthesis.

Identical to finn_ooc_partition2_hawq.py except HAWQ_PREAMBLE_DIR points at
the trained-checkpoint preamble run and OUTPUT_DIR/stitched-filename use a
"_trained" suffix -- this run's whole point is to confirm the real trained
checkpoint's resource/timing numbers match the fresh-weight HAWQ run
(architecture + bit-width + folding config are identical; only weight VALUES
differ, which FINN's resource estimate should not depend on).

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \
        /home/thelegendiv/finn/notebooks/enet/finn_ooc_partition2_hawq_trained.py
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

# Fixed to the already-completed TRAINED HAWQ preamble run
# (finn_hawq_preamble_trained.py) -- hardcoded (not re-discovered) since
# that run's assign_stage_partition_ids_8way.onnx is already on disk and its
# hawq_folding_config_partition2.json is generated (via
# finn_hawq_folding_bridge.py) before this script is launched.
HAWQ_PREAMBLE_DIR = os.path.join(
    base.ENET_DIR, "finn_deployment_outputs", "hawq_trained_preamble_20260825_084955",
)
SOURCE_CKPT = os.path.join(HAWQ_PREAMBLE_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
BRIDGED_FOLDING_CONFIG = os.path.join(HAWQ_PREAMBLE_DIR, "hawq_folding_config_partition2.json")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(
    base.ENET_DIR, "finn_deployment_outputs", f"partition2_hawq_block_trained_{timestamp}"
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
    print("Running: step_apply_folding_config (HAWQ bridged config, trained checkpoint)")
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
    final_fn = os.path.join(OUTPUT_DIR, "partition2_hawq_block_trained_stitched.onnx")
    kernel_model.save(final_fn)

    print("Running: step_measure_rtlsim_performance")
    kernel_model = step_measure_rtlsim_performance(kernel_model, cfg)
    kernel_model.save(final_fn)

    print("Running: step_out_of_context_synthesis")
    kernel_model = step_out_of_context_synthesis(kernel_model, cfg)
    kernel_model.save(final_fn)

    print("Done. Reports in", os.path.join(OUTPUT_DIR, "report"))

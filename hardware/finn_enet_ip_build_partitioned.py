"""FINN stitched-IP + OOC synthesis build for S19 (or structurally similar
ENet-family exports), using a 5-way STAGE-BASED GRAPH PARTITION instead of
the single monolithic stitch used by finn_enet_ip_build_decomposed_prelu.py.

Why: the monolithic single-partition stitch showed near-quadratic
Vivado `create_bd_cell` cost as the design grew (1665 cells -> multi-day
CreateStitchedIP, see session notes 2026-08-10..15). Splitting into 5
independent, much smaller partitions (initial / stage1 / stage2+3 / stage4
/ stage5, cut at each down/upsampling bottleneck's first op -- see
finn_stage_partition.py's module docstring for exactly how boundaries are
detected) means each partition's own CreateStitchedIP pays the quadratic
cost on a much smaller n, and partitions can be built concurrently since
they don't depend on each other until the final combining stitch.

Reuses the same proven tidy/streamline/convert_to_hw pipeline as
finn_enet_ip_build_decomposed_prelu.py (imported, not duplicated). Only the
partitioning/stitching tail differs -- see finn_partition_build_steps.py.

Run inside the FINN container:
    docker exec <container_id> python /home/thelegendiv/finn/notebooks/enet/finn_enet_ip_build_partitioned.py <model_name> [auto_fifo_strategy]

STATUS: new, not yet run end-to-end. step_combine_partitions intentionally
does NOT auto-invoke Vivado yet (writes the tcl and stops) -- run it
manually once to validate before trusting an unattended full build.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from finn_enet_build_decomposed_prelu import (
    ENET_DIR,
    step_enet_tidy,
    step_fuse_leaky_relu_to_threshold,
    step_enet_streamline,
    step_absorb_leftover_scale_before_matmul,
    step_fuse_forked_dequant_into_duplicate_threshold,
    _fixup_degenerate_signed_bias,
    step_enet_convert_to_hw,
)

from finn_stage_partition import assign_stage_partition_ids
from finn_partition_build_steps import (
    step_create_dataflow_partition_multi,
    step_build_all_partitions,
    step_combine_partitions,
)


def step_build_all_partitions_capped(model, cfg):
    # max_workers=3 (not unbounded): each partition's own step_hw_ipgen already
    # parallelizes HLS synth internally (NUM_DEFAULT_WORKERS=4), so unbounded
    # partition-level concurrency risks CPU oversubscription (up to 5x4=20
    # processes on 12 cores). Capping to 3 still lets small partitions finish
    # and free a slot for the next one without saturating the machine.
    # (Plain named function, not functools.partial -- FINN's
    # resolve_build_steps() unconditionally does step_fn.__name__, which
    # functools.partial objects don't have.)
    return step_build_all_partitions(model, cfg, parallel=True, max_workers=3)

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig

# ── paths ────────────────────────────────────────────────────────────────────
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "quantEnet_s19_double_mid_int8"
FIFO_STRATEGY = sys.argv[2] if len(sys.argv) > 2 else "largefifo_rtlsim"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(
    ENET_DIR, "finn_deployment_outputs", f"stitched_ip_partitioned_{MODEL_NAME}_{FIFO_STRATEGY}_{timestamp}"
)

XCZU7EV = {"LUT": 230400, "BRAM_18K": 624, "DSP": 1728}

enet_ip_partitioned_steps = [
    "step_qonnx_to_finn",
    step_enet_tidy,
    step_fuse_leaky_relu_to_threshold,
    step_enet_streamline,
    step_absorb_leftover_scale_before_matmul,
    step_fuse_forked_dequant_into_duplicate_threshold,
    _fixup_degenerate_signed_bias,
    step_enet_convert_to_hw,
    assign_stage_partition_ids,          # <-- NEW: sets partition_id 0..4
    step_create_dataflow_partition_multi,  # <-- NEW: no single-partition assert
    step_build_all_partitions_capped,    # <-- NEW: per-partition specialize/fold/codegen/ipgen/fifo/stitch, parallel (max_workers=3)
    step_combine_partitions,             # <-- NEW: small top-level stitch (writes tcl, run manually first)
    "step_generate_estimate_reports",
    "step_measure_rtlsim_performance",
    "step_out_of_context_synthesis",
]

cfg_stitched_ip_partitioned = DataflowBuildConfig(
    output_dir          = OUTPUT_DIR,
    mvau_wwidth_max     = 80,
    target_fps          = 250,
    synth_clk_period_ns = 10.0,
    fpga_part           = "xczu7ev-ffvc1156-2-e",
    auto_fifo_strategy  = build_cfg.AutoFIFOSizingMethod(FIFO_STRATEGY),
    steps               = enet_ip_partitioned_steps,
    generate_outputs    = [
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        build_cfg.DataflowOutputType.OOC_SYNTH,
    ],
    save_intermediate_models = True,
)

if __name__ == "__main__":
    print(f"Model : {MODEL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"FIFO strategy: {FIFO_STRATEGY}")
    print(f"Steps : {[s if isinstance(s, str) else s.__name__ for s in enet_ip_partitioned_steps]}")
    print()

    build.build_dataflow_cfg(MODEL_FILE, cfg_stitched_ip_partitioned)

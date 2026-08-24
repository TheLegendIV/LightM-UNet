"""FINN stitched-IP + OOC synthesis build for S19, using an 8-WAY
STAGE-BASED GRAPH PARTITION (5-way's stage2/3 partition -- 419 nodes, the
one that took 2+ days and showed near-quadratic Vivado create_bd_cell
blowup -- split into 4 roughly-symmetrical quarters; the first quarter
absorbs the down2 downsample even though that breaks exact symmetry).

Verified analytically first (no Vivado) via finn_verify_8way_stage23_split.py
against the existing step_enet_convert_to_hw.onnx checkpoint: the 4 new
stage2/3 quarters come out to ~105 nodes / ~34-35k LUT (~15% of xczu7ev)
each, well balanced (see hardware/outputs/verify_8way_stage23_split_result.json).

Identical to finn_enet_ip_build_partitioned.py except:
  - imports assign_stage_partition_ids_8way instead of assign_stage_partition_ids
  - OUTPUT_DIR naming says "8way"
  - max_workers raised to 4 (8 much-smaller partitions instead of 5 larger
    ones -- still capped well under 8*4=32 to avoid oversubscribing the
    12-core box during per-partition HLS synth, which itself already runs
    NUM_DEFAULT_WORKERS=4 internally)

Run inside the FINN container:
    docker exec <container_id> python /home/thelegendiv/finn/notebooks/enet/finn_enet_ip_build_partitioned_8way.py <model_name> [auto_fifo_strategy]

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

from finn_stage_partition import assign_stage_partition_ids_8way
from finn_partition_build_steps import (
    step_create_dataflow_partition_multi,
    step_build_all_partitions,
    step_combine_partitions,
    step_generate_estimate_reports_multi,
    step_measure_rtlsim_performance_multi,
    step_out_of_context_synthesis_multi,
)


def step_build_all_partitions_capped(model, cfg):
    # 8 partitions now, each much smaller (~105 nodes vs up to 419 before).
    # max_workers=4 (not unbounded): each partition's own step_hw_ipgen
    # already parallelizes HLS synth internally (NUM_DEFAULT_WORKERS=4), so
    # unbounded partition-level concurrency risks CPU oversubscription (up
    # to 8x4=32 processes on 12 cores). Capping to 4 still lets small
    # partitions finish and free a slot for the next one without
    # saturating the machine.
    # (Plain named function, not functools.partial -- FINN's
    # resolve_build_steps() unconditionally does step_fn.__name__, which
    # functools.partial objects don't have.)
    return step_build_all_partitions(model, cfg, parallel=True, max_workers=4)

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig

# ── paths ────────────────────────────────────────────────────────────────────
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "quantEnet_s19_double_mid_int8"
FIFO_STRATEGY = sys.argv[2] if len(sys.argv) > 2 else "largefifo_rtlsim"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(
    ENET_DIR, "finn_deployment_outputs", f"stitched_ip_partitioned8way_{MODEL_NAME}_{FIFO_STRATEGY}_{timestamp}"
)

XCZU7EV = {"LUT": 230400, "BRAM_18K": 624, "DSP": 1728}

enet_ip_partitioned_8way_steps = [
    "step_qonnx_to_finn",
    step_enet_tidy,
    step_fuse_leaky_relu_to_threshold,
    step_enet_streamline,
    step_absorb_leftover_scale_before_matmul,
    step_fuse_forked_dequant_into_duplicate_threshold,
    _fixup_degenerate_signed_bias,
    step_enet_convert_to_hw,
    assign_stage_partition_ids_8way,     # <-- 8-way: sets partition_id 0..7
    step_create_dataflow_partition_multi,  # <-- no single-partition assert
    step_build_all_partitions_capped,    # <-- per-partition specialize/fold/codegen/ipgen/fifo/stitch, parallel (max_workers=4)
    step_combine_partitions,             # <-- top-level stitch of 8 partition IPs (writes tcl, run manually first)
    step_generate_estimate_reports_multi,  # <-- generic version assumes flat HW-node graph; combined model is all SDP nodes
    step_measure_rtlsim_performance_multi,  # <-- generic version's find_consumer/producer hits Transpose/Mul, not a HW node
    step_out_of_context_synthesis_multi,  # <-- generic version's dataflow_performance() call would fail the same way
]

cfg_stitched_ip_partitioned_8way = DataflowBuildConfig(
    output_dir          = OUTPUT_DIR,
    mvau_wwidth_max     = 80,
    target_fps          = 250,
    synth_clk_period_ns = 10.0,
    fpga_part           = "xczu7ev-ffvc1156-2-e",
    auto_fifo_strategy  = build_cfg.AutoFIFOSizingMethod(FIFO_STRATEGY),
    steps               = enet_ip_partitioned_8way_steps,
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
    print(f"Steps : {[s if isinstance(s, str) else s.__name__ for s in enet_ip_partitioned_8way_steps]}")
    print()

    build.build_dataflow_cfg(MODEL_FILE, cfg_stitched_ip_partitioned_8way)

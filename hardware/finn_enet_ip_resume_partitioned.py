"""Resume a crashed finn_enet_ip_build_partitioned.py run from its last
successful checkpoint. Same pattern as finn_enet_ip_resume.py (the
single-partition build's resume script) -- see that file's docstring for
the general mechanism (DataflowBuildConfig.start_step +
save_intermediate_models loading the checkpoint saved right before
start_step from <output_dir>/intermediate_models/).

Usage (inside the FINN container):
    python3 finn_enet_ip_resume_partitioned.py <failed_output_dir> <start_step>

Example (resuming past the assign_stage_partition_ids getCustomOp bug,
fixed 2026-08-15 -- see PARTITIONED_BUILD_LOG.md):
    python3 finn_enet_ip_resume_partitioned.py \
        finn_deployment_outputs/stitched_ip_partitioned_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260815_113231 \
        assign_stage_partition_ids

This reuses the *exact* same DataflowBuildConfig as the original partitioned
run (steps, fpga_part, mvau_wwidth_max, target_fps, generate_outputs, etc.)
via dataclasses.replace, only overriding output_dir (to point at the
existing run's directory, where intermediate_models/ already lives) and
start_step.
"""

import os
import sys
import dataclasses

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)

failed_output_dir = os.path.abspath(sys.argv[1])
start_step = sys.argv[2]

# IMPORTANT: finn_enet_ip_build_partitioned.py parses sys.argv[1]/[2] itself
# (MODEL_NAME/FIFO_STRATEGY) at import time -- if we don't clear our own
# argv first, its module-level code would misread *our* output_dir/start_step
# as MODEL_NAME/FIFO_STRATEGY (this bit exactly, ValueError: '<start_step>'
# is not a valid AutoFIFOSizingMethod). Reset argv to just the program name
# so the base module falls back to its own defaults
# ("quantEnet_s19_double_mid_int8" / "largefifo_rtlsim") -- correct as long
# as the failed run being resumed used those same defaults (true for all
# runs so far; if resuming a run with non-default MODEL_NAME/FIFO_STRATEGY,
# set them here explicitly instead before the import).
sys.argv = [sys.argv[0]]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finn_enet_ip_build_partitioned as base  # noqa: E402

import finn.builder.build_dataflow as build  # noqa: E402

if __name__ == "__main__":
    cfg_resume = dataclasses.replace(
        base.cfg_stitched_ip_partitioned,
        output_dir=failed_output_dir,
        start_step=start_step,
    )

    print(f"Resuming partitioned build in {failed_output_dir}")
    print(f"Start step: {start_step} (loads checkpoint from intermediate_models/)")

    build.build_dataflow_cfg(base.MODEL_FILE, cfg_resume)

    print("\nResume run finished.")

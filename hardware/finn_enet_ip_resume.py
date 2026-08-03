"""Resume a crashed finn_enet_ip_build.py run from its last successful checkpoint.

FINN's DataflowBuildConfig natively supports start_step/stop_step + relies on
save_intermediate_models (already enabled in finn_enet_ip_build.py) to load the
checkpoint saved right before start_step from <output_dir>/intermediate_models/.

Usage (inside the FINN container):
    python3 finn_enet_ip_resume.py <failed_output_dir> <start_step>

Example (resuming past the step_set_fifo_depths permission failure):
    python3 finn_enet_ip_resume.py \
        finn_deployment_outputs/stitched_ip_quantEnet_finn_v1_20260729_171005 \
        step_set_fifo_depths

This reuses the *exact* same DataflowBuildConfig as the original run (steps,
fpga_part, mvau_wwidth_max, target_fps, generate_outputs, etc.) via
dataclasses.replace, only overriding output_dir (to point at the existing run's
directory, where intermediate_models/ already lives) and start_step.
"""

import os
import sys
import dataclasses

# import the original build script as a module -- its `if __name__ == "__main__"`
# guard means importing it does NOT re-run the build; we just reuse its config,
# custom step functions, PATH/env setup, and MODEL_FILE.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finn_enet_ip_build as base  # noqa: E402

import finn.builder.build_dataflow as build  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    failed_output_dir = os.path.abspath(sys.argv[1])
    start_step = sys.argv[2]

    cfg_resume = dataclasses.replace(
        base.cfg_stitched_ip,
        output_dir=failed_output_dir,
        start_step=start_step,
    )

    print(f"Resuming build in {failed_output_dir}")
    print(f"Start step: {start_step} (loads checkpoint from intermediate_models/)")

    build.build_dataflow_cfg(base.MODEL_FILE, cfg_resume)

    print("\nResume run finished.")

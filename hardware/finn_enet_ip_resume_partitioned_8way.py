"""Resume a crashed finn_enet_ip_build_partitioned_8way.py run from its last
successful checkpoint. Same pattern as finn_enet_ip_resume_partitioned.py
(the 5-way build's resume script) -- see that file's docstring for the
general mechanism (DataflowBuildConfig.start_step + save_intermediate_models
loading the checkpoint saved right before start_step from
<output_dir>/intermediate_models/).

Usage (inside the FINN container):
    python3 finn_enet_ip_resume_partitioned_8way.py <failed_output_dir> <start_step>

Example (resuming past the /tmp/finn_dev_thelegendiv permission-ownership
bug, hit 2026-08-20 -- stale root-owned tmp dir from container startup,
fixed via `docker exec -u root <container> chown -R thelegendiv:thelegendiv
/tmp/finn_dev_thelegendiv`):
    python3 finn_enet_ip_resume_partitioned_8way.py \
        finn_deployment_outputs/stitched_ip_partitioned8way_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260820_101224 \
        step_build_all_partitions_capped

This reuses the *exact* same DataflowBuildConfig as the original 8-way run
(steps, fpga_part, mvau_wwidth_max, target_fps, generate_outputs, etc.) via
dataclasses.replace, only overriding output_dir (to point at the existing
run's directory, where intermediate_models/ already lives) and start_step.
"""

import os
import sys
import dataclasses

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)

failed_output_dir = os.path.abspath(sys.argv[1])
start_step = sys.argv[2]

# See finn_enet_ip_resume_partitioned.py for why argv must be reset before
# importing the base build module (it parses sys.argv[1]/[2] itself as
# MODEL_NAME/FIFO_STRATEGY at import time).
sys.argv = [sys.argv[0]]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402

import finn.builder.build_dataflow as build  # noqa: E402

if __name__ == "__main__":
    cfg_resume = dataclasses.replace(
        base.cfg_stitched_ip_partitioned_8way,
        output_dir=failed_output_dir,
        start_step=start_step,
    )

    print(f"Resuming 8-way partitioned build in {failed_output_dir}")
    print(f"Start step: {start_step} (loads checkpoint from intermediate_models/)")

    build.build_dataflow_cfg(base.MODEL_FILE, cfg_resume)

    print("\nResume run finished.")

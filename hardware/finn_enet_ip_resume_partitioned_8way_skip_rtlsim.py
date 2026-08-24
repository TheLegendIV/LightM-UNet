"""Same as finn_enet_ip_resume_partitioned_8way.py, but drops
step_measure_rtlsim_performance_multi from the step list before resuming.

Rationale: the current priority is real post-synthesis resource-consumption
numbers (step_out_of_context_synthesis_multi), not rtlsim throughput
validation. step_measure_rtlsim_performance_multi took ~2.5 hours in the
prior attempt and step_out_of_context_synthesis_multi does not depend on
its output (both only consume the model produced by step_combine_partitions;
max_cycles etc. used by OOC synthesis come from
step_generate_estimate_reports_multi, not rtlsim) -- so skipping it gets to
the resource numbers much faster.

Usage (inside the FINN container):
    python3 finn_enet_ip_resume_partitioned_8way_skip_rtlsim.py <failed_output_dir> <start_step>
"""

import os
import sys
import dataclasses

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)

failed_output_dir = os.path.abspath(sys.argv[1])
start_step = sys.argv[2]

sys.argv = [sys.argv[0]]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
from finn_partition_build_steps import step_measure_rtlsim_performance_multi  # noqa: E402

import finn.builder.build_dataflow as build  # noqa: E402

if __name__ == "__main__":
    steps_no_rtlsim = [
        s for s in base.enet_ip_partitioned_8way_steps if s is not step_measure_rtlsim_performance_multi
    ]

    cfg_resume = dataclasses.replace(
        base.cfg_stitched_ip_partitioned_8way,
        output_dir=failed_output_dir,
        start_step=start_step,
        steps=steps_no_rtlsim,
    )

    print(f"Resuming 8-way partitioned build (SKIPPING rtlsim perf step) in {failed_output_dir}")
    print(f"Start step: {start_step} (loads checkpoint from intermediate_models/)")

    build.build_dataflow_cfg(base.MODEL_FILE, cfg_resume)

    print("\nResume run finished.")

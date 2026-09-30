"""Run ONLY step_out_of_context_synthesis on an already-stitched partition-2
checkpoint (produced by finn_ooc_partition2_trained.py before it crashed at
step_measure_rtlsim_performance). Skips the entire expensive rebuild since
the stitched IP + its metadata (vivado_stitch_proj, wrapper_filename) are
still on disk and OOC synth reads them dynamically (no hardcoded name).

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_ooc_synth_only_partition2.py \\
        <stitched_checkpoint.onnx> <output_dir>
"""

import dataclasses
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

if len(sys.argv) != 3:
    print("Usage: finn_ooc_synth_only_partition2.py <stitched_checkpoint.onnx> <output_dir>")
    sys.exit(1)
STITCHED_CKPT = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn.builder.build_dataflow_steps import step_out_of_context_synthesis  # noqa: E402

cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)

print(f"Stitched checkpoint : {STITCHED_CKPT}")
print(f"Output dir          : {OUTPUT_DIR}")

kernel_model = ModelWrapper(STITCHED_CKPT)

print("Running: step_out_of_context_synthesis")
kernel_model = step_out_of_context_synthesis(kernel_model, cfg)
kernel_model.save(STITCHED_CKPT)

print(f"Done. Reports in {OUTPUT_DIR}/report")

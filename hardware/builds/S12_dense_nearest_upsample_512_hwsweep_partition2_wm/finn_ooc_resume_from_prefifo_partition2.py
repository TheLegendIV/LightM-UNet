"""Resume-from-prefifo-autosize partition-2 OOC-synth-only script.

Loads a prefifo_autosize checkpoint (saved right after the expensive
step_set_fifo_depths rtlsim autosizing, before SplitLargeFIFOs/CreateStitchedIP)
and runs ONLY the remaining steps needed for real OOC synth resource/timing
numbers: SplitLargeFIFOs -> PrepareIP -> HLSSynthIP -> CreateStitchedIP ->
step_out_of_context_synthesis. Deliberately skips step_measure_rtlsim_performance
(not needed -- only OOC synth numbers are wanted this time).

Needed because the previously-produced stitched.onnx files' vivado_stitch_proj
and per-node ip_path metadata point at /tmp/finn_dev_thelegendiv, which was
wiped by a Docker Desktop backend crash. FIFO depths are still valid in the
prefifo_autosize checkpoint (no need to re-run the expensive rtlsim autosize),
but the per-node IP and stitched Vivado project must be regenerated -- this
time landing under the new persistent FINN_BUILD_DIR baked into finn_persistent.

Run inside the FINN container:
    docker exec finn_persistent python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_ooc_resume_from_prefifo_partition2.py \\
        <prefifo_autosize_checkpoint.onnx>
"""

import os
import sys
import dataclasses

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

if len(sys.argv) != 2:
    print("Usage: finn_ooc_resume_from_prefifo_partition2.py <prefifo_autosize_checkpoint.onnx>")
    sys.exit(1)
PREFIFO_CKPT = sys.argv[1]

BASENAME = os.path.basename(PREFIFO_CKPT)
assert BASENAME.startswith("partition2_") and BASENAME.endswith("_prefifo_autosize.onnx"), BASENAME
IDENT = BASENAME[len("partition2_"):-len("_prefifo_autosize.onnx")]
OUTPUT_DIR = os.path.dirname(PREFIFO_CKPT)

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames  # noqa: E402

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs  # noqa: E402
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP  # noqa: E402
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP  # noqa: E402
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP  # noqa: E402
from finn.builder.build_dataflow_steps import step_out_of_context_synthesis  # noqa: E402

if __name__ == "__main__":
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)
    fpga_part = cfg._resolve_fpga_part()
    clk_period_ns = cfg.synth_clk_period_ns

    print(f"[{IDENT}] loading prefifo-autosize checkpoint: {PREFIFO_CKPT}")
    kernel_model = ModelWrapper(PREFIFO_CKPT)

    print(f"[{IDENT}] Running: SplitLargeFIFOs")
    kernel_model = kernel_model.transform(SplitLargeFIFOs())
    kernel_model = kernel_model.transform(GiveUniqueNodeNames())

    print(f"[{IDENT}] Running: PrepareIP + HLSSynthIP "
          f"(regenerates all per-node IP -- old paths pointed at the now-wiped /tmp)")
    kernel_model = kernel_model.transform(PrepareIP(fpga_part, clk_period_ns))
    kernel_model = kernel_model.transform(HLSSynthIP())

    print(f"[{IDENT}] Running: CreateStitchedIP")
    kernel_model = kernel_model.transform(CreateStitchedIP(fpga_part, clk_period_ns))
    final_fn = os.path.join(OUTPUT_DIR, f"partition2_{IDENT}_stitched.onnx")
    kernel_model.save(final_fn)

    print(f"[{IDENT}] Running: step_out_of_context_synthesis")
    kernel_model = step_out_of_context_synthesis(kernel_model, cfg)
    kernel_model.save(final_fn)

    print(f"[{IDENT}] Done. Reports in {OUTPUT_DIR}/report")

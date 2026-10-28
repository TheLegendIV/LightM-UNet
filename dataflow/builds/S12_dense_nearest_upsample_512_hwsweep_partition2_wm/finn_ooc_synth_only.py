"""Run ONLY step_out_of_context_synthesis on an already-stitched partition-2
checkpoint from the crashed (pre-ip_name-fix) sweep. The stitched IP itself
is untouched by that bug (only rtlsim's hardcoded wrapper-name lookup was
broken), so we can get real OOC resource/timing numbers without re-running
the expensive FIFO-autosize + CreateStitchedIP steps.

Usage:
    python3 finn_ooc_synth_only.py <stitched.onnx> <output_dir> <tag>
"""

import os
import sys
import dataclasses

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

if len(sys.argv) != 4:
    print("Usage: finn_ooc_synth_only.py <stitched.onnx> <output_dir> <tag>")
    sys.exit(1)
STITCHED_ONNX = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
TAG = sys.argv[3]

_real_argv = sys.argv
sys.argv = _real_argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from finn.builder.build_dataflow_steps import step_out_of_context_synthesis  # noqa: E402

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)
    print(f"[{TAG}] loading {STITCHED_ONNX}")
    model = ModelWrapper(STITCHED_ONNX)
    print(f"[{TAG}] Running: step_out_of_context_synthesis")
    model = step_out_of_context_synthesis(model, cfg)
    out_fn = os.path.join(OUTPUT_DIR, f"partition2_{TAG}_ooc_synth.onnx")
    model.save(out_fn)
    print(f"[{TAG}] Done. Report in {OUTPUT_DIR}/report, model saved to {out_fn}")

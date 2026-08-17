"""FINN stitched-IP + out-of-context Vivado synthesis build for QuantENet
configs that use the decomposed-PReLU activation (S5-DscNoProjDense,
S13-LeakyFrozen, S19-DoubleMid, ...).

This reuses the EXACT tidy/streamline/convert_to_hw step implementations
from finn_enet_build_decomposed_prelu.py (imported directly, not
duplicated) -- including step_fuse_leaky_relu_to_threshold,
step_absorb_leftover_scale_before_matmul,
step_fuse_forked_dequant_into_duplicate_threshold, and
_fixup_degenerate_signed_bias -- and extends that pipeline with
finn_enet_ip_build.py's real HLS-codegen -> stitched-IP -> OOC-synth tail
steps (step_hw_codegen, step_hw_ipgen, step_set_fifo_depths,
step_create_stitched_ip, step_measure_rtlsim_performance,
step_out_of_context_synthesis) and its DataflowBuildConfig knobs
(auto_fifo_strategy, target_fps=250, generate_outputs incl. STITCHED_IP/
RTLSIM_PERFORMANCE/OOC_SYNTH).

Why this file exists (rather than just running finn_enet_ip_build.py
directly on a decomposed-PReLU export): finn_enet_ip_build.py's own
step_enet_tidy/streamline/convert_to_hw implementations pre-date the
decomposed-PReLU activation convention and its associated MultiThreshold
degenerate-out_bias bug. Running it directly against a decomposed-PReLU
export fails at step_enet_convert_to_hw with:
    AssertionError: MultiThreshold_0: Signed output requires actval < 0
(confirmed against the S19 real-weight export, 2026-08-09 -- see
_fixup_degenerate_signed_bias's docstring in finn_enet_build_decomposed_prelu.py
for the full root-cause analysis). finn_enet_ip_build.py itself has only
ever been proven against older QuantReLU-only configs (O2/O8-native),
which never hit this bug because QuantReLU's activation quantizer is
unsigned (Uint8ActPerTensorFloat), not signed (Int8ActPerTensorFloat).

Run inside the FINN container:
    docker exec <container_id> python /home/thelegendiv/finn/notebooks/enet/finn_enet_ip_build_decomposed_prelu.py <model_name> [auto_fifo_strategy]

    model_name:         filename stem (no .onnx) under ENET_DIR, e.g.
                         quantEnet_s19_double_mid_int8. Defaults to
                         quantEnet_s5_dscnoproj_dense_int8.
    auto_fifo_strategy: "largefifo_rtlsim" (default, proven for O2/O8) or
                         "characterize" -- see finn_enet_ip_build.py's own
                         docstring for the tradeoff.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

# Reuse the exact, proven decomposed-PReLU tidy/streamline/convert_to_hw
# pipeline (including the degenerate-signed-bias fix and the Xilinx PATH
# setup it inherits from finn_enet_build.py). Importing does NOT execute
# finn_enet_build_decomposed_prelu.py's `if __name__ == "__main__"` build
# call -- only its module-level step/function definitions run.
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

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig

# ── paths ────────────────────────────────────────────────────────────────────
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "quantEnet_s5_dscnoproj_dense_int8"
FIFO_STRATEGY = sys.argv[2] if len(sys.argv) > 2 else "largefifo_rtlsim"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"stitched_ip_{MODEL_NAME}_{FIFO_STRATEGY}_{timestamp}")

# Real xczu7ev-ffvc1156-2-e (Zynq UltraScale+ ZU7EV) -- see finn_enet_ip_build.py
# for the detailed LUT/BRAM/DSP figure provenance (DS891 CLB resource table).
XCZU7EV = {"LUT": 230400, "BRAM_18K": 624, "DSP": 1728}

enet_ip_steps = [
    "step_qonnx_to_finn",
    step_enet_tidy,
    step_fuse_leaky_relu_to_threshold,          # <-- fuses PReLU's LeakyRelu into a single MultiThreshold
    step_enet_streamline,
    step_absorb_leftover_scale_before_matmul,   # <-- fixes leftover dequant Mul/Add before Im2Col
    step_fuse_forked_dequant_into_duplicate_threshold,  # <-- last-resort fix for FMPadding_Pixel-adjacent leftovers
    _fixup_degenerate_signed_bias,               # <-- fixes "signed output requires actval < 0"
    step_enet_convert_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",       # HW → HLS/RTL backend variants
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",   # cheap analytical estimates, kept for comparison
    "step_hw_codegen",               # Vitis HLS / RTL codegen per HW node
    "step_hw_ipgen",                 # synthesize each node's IP
    "step_set_fifo_depths",          # size + insert inter-node FIFOs
    "step_create_stitched_ip",       # combine into one Vivado IP core
    "step_measure_rtlsim_performance",   # cycle-accurate perf via Verilator
    "step_out_of_context_synthesis",     # real LUT/FF/BRAM/DSP/Fmax numbers
]

cfg_stitched_ip = DataflowBuildConfig(
    output_dir          = OUTPUT_DIR,
    mvau_wwidth_max     = 80,
    target_fps          = 250,
    synth_clk_period_ns = 10.0,
    fpga_part           = "xczu7ev-ffvc1156-2-e",
    auto_fifo_strategy  = build_cfg.AutoFIFOSizingMethod(FIFO_STRATEGY),
    steps               = enet_ip_steps,
    generate_outputs    = [
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        build_cfg.DataflowOutputType.OOC_SYNTH,
    ],
    save_intermediate_models = True,
    # mem_mode/runtime_writeable_weights left at FINN defaults (internal_decoupled,
    # runtime_writeable_weights=0) — weights fixed at build time, per user choice.
)

if __name__ == "__main__":
    print(f"Model : {MODEL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"FIFO strategy: {FIFO_STRATEGY}")
    print(f"Steps : {[s if isinstance(s, str) else s.__name__ for s in enet_ip_steps]}")
    print()

    build.build_dataflow_cfg(MODEL_FILE, cfg_stitched_ip)

"""FINN build for the minimal S19-single-bottleneck resource-probe model
(hardware/finn_export_s19_single_block.py), with an EXTRA custom step that
forcibly overrides FINN's own "auto" resource-mapping heuristics right
after specialize_layers + folding:
    - resType   = "dsp"   on every MVAU_hls/MVAU_rtl node (force DSP-based
                  multiply-accumulate instead of FINN's LUT default at
                  this narrow int8xint8 operating point)
    - ram_style = "ultra" on every ConvolutionInputGenerator_hls/_rtl node
                  (force URAM instead of BRAM/LUTRAM for the sliding-window
                  line-buffer memory)

Both nodeattrs are confirmed-real FINN options (verified 2026-08-21 via
direct source read of matrixvectoractivation.py / convolutioninputgenerator*.py
in this same container): resType accepts {"auto","lut","dsp"}, ram_style
accepts {"auto","block","distributed","ultra"} and both feed real
lut_estimation/dsp_estimation/uram_estimation methods AND real Verilog
RAM_STYLE/multiplier-primitive attributes -- so a "success" here should
show up as nonzero DSP/URAM counts in the real Vivado OOC synthesis report,
not just FINN's own analytical estimate.

Reuses the exact tidy/streamline/convert_to_hw pipeline from
finn_enet_ip_build_decomposed_prelu.py (imported, not duplicated) -- this
export uses the same decomposed-PReLU (epsilon-slope LeakyRelu) activation
convention as every other FINN*_export_*.py script in this repo.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \
        /home/thelegendiv/finn/notebooks/enet/finn_enet_ip_build_s19_single_block.py \
        [model_name] [auto_fifo_strategy]

    model_name: defaults to quantEnet_s19_single_block_int8
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from finn_enet_ip_build_decomposed_prelu import (
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
from qonnx.custom_op.registry import getCustomOp

# ── paths ────────────────────────────────────────────────────────────────────
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "quantEnet_s19_single_block_int8"
FIFO_STRATEGY = sys.argv[2] if len(sys.argv) > 2 else "largefifo_rtlsim"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"stitched_ip_{MODEL_NAME}_force_dsp_uram_{timestamp}")

XCZU7EV = {"LUT": 230400, "BRAM_18K": 624, "DSP": 1728}


def step_force_dsp_and_uram(model, cfg):
    """Force resType="dsp" on every MVAU node and ram_style="ultra" on every
    ConvolutionInputGenerator (SWU) node. Must run AFTER step_specialize_layers
    (so the concrete _hls/_rtl backend op instances -- which own these
    nodeattrs -- already exist) and after step_target_fps_parallelization
    (so PE/SIMD/SIMD folding is already set; resType/ram_style don't affect
    folding, but ordering keeps this step a pure "final override" pass)."""
    n_dsp = 0
    n_uram = 0
    for node in model.graph.node:
        if "MVAU" in node.op_type:
            inst = getCustomOp(node)
            inst.set_nodeattr("resType", "dsp")
            n_dsp += 1
        elif "ConvolutionInputGenerator" in node.op_type:
            inst = getCustomOp(node)
            inst.set_nodeattr("ram_style", "ultra")
            n_uram += 1
    print(f"[step_force_dsp_and_uram] forced resType=dsp on {n_dsp} MVAU node(s), "
          f"ram_style=ultra on {n_uram} ConvolutionInputGenerator node(s)")
    return model


enet_ip_steps = [
    "step_qonnx_to_finn",
    step_enet_tidy,
    step_fuse_leaky_relu_to_threshold,
    step_enet_streamline,
    step_absorb_leftover_scale_before_matmul,
    step_fuse_forked_dequant_into_duplicate_threshold,
    _fixup_degenerate_signed_bias,
    step_enet_convert_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    step_force_dsp_and_uram,          # <-- force DSP + URAM after folding/bit-width settle
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_measure_rtlsim_performance",
    "step_out_of_context_synthesis",
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
)

if __name__ == "__main__":
    print(f"Model : {MODEL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"FIFO strategy: {FIFO_STRATEGY}")
    print(f"Steps : {[s if isinstance(s, str) else s.__name__ for s in enet_ip_steps]}")
    print()

    build.build_dataflow_cfg(MODEL_FILE, cfg_stitched_ip)

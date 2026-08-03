"""FINN stitched-IP build script for QuantENet — ZCU7EV accelerator core only.

Run inside the FINN container:
    docker exec d345f89b4e6c python /home/thelegendiv/finn/notebooks/enet/finn_enet_ip_build.py <model_name>

Unlike finn_enet_build.py (analytical estimates only), this script actually runs
Vitis HLS / RTL codegen for every HW node, stitches the resulting IPs into a single
Vivado IP core, then verifies it with rtlsim and out-of-context synthesis. It
deliberately stops short of a full bitstream: no `board=`/`shell_flow_type=` is set,
so no PS/Zynq/DDR/DMA integration or board constraints are produced — the user
integrates the stitched IP into their own PS/DDR system manually.

Outputs (under OUTPUT_DIR):
  stitched_ip/                        Vivado IP-XACT core (component.xml + HDL) —
                                       one AXI4-Stream in, one AXI4-Stream out.
  report/rtlsim_performance.json      Cycle-accurate throughput/latency (Verilator).
  report/ooc_synth_and_timing.json    Real post-synthesis LUT/FF/BRAM/DSP/Fmax.
  report/estimate_*.json              Same analytical estimates as finn_enet_build.py,
                                       kept for side-by-side comparison.

Weights: MVAU/VVAU default to mem_mode="internal_decoupled" with
runtime_writeable_weights=0 — weights are baked in at build time (loaded once from
an internal streamer), not runtime-writeable via AXI-lite. This is the FINN default
and was NOT overridden here.

Custom streamline + convert_to_hw is identical to finn_enet_build.py (copied, not
imported, so this script has no import-time side effects tied to sys.argv) —
see that file for the residual/fork-handling rationale.
"""

import os
import sys
import shutil
from datetime import datetime

# ── FINN source paths (source install, not pip) ──────────────────────────────
sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/brevitas/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/pyverilator")
sys.path.insert(0, "/home/thelegendiv/finn/deps/finn-experimental")

# ── Xilinx tool PATH (vitis_hls / vivado) ─────────────────────────────────────
# `docker exec ... bash -c '...'` runs a non-interactive, non-login shell, which
# does NOT source ~/.bashrc -- so vitis_hls/vivado are missing from PATH even
# though they're installed, and HLSSynthIP/CreateStitchedIP/SynthOutOfContext
# (which shell out via shutil.which("vitis_hls"/"vivado")) fail with
# "vitis_hls not found in PATH". Fixing PATH here (rather than relying on
# .bashrc or the invoking shell) makes this script self-sufficient regardless
# of how it's launched.
_XILINX_BIN_DIRS = [
    "/tools/Xilinx/Vitis_HLS/2022.2/bin",
    "/tools/Xilinx/Vivado/2022.2/bin",
]
os.environ["PATH"] = os.pathsep.join(_XILINX_BIN_DIRS + [os.environ.get("PATH", "")])
os.environ.setdefault("XILINX_VIVADO", "/tools/Xilinx/Vivado/2022.2")
os.environ.setdefault("XILINX_HLS", "/tools/Xilinx/Vitis_HLS/2022.2")

# ── imports ──────────────────────────────────────────────────────────────────
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.general import (
    ConvertSubToAdd,
    ConvertDivToMul,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
    SortGraph,
)

from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    Absorb1BitMulIntoMatMul,
    Absorb1BitMulIntoConv,
    AbsorbConsecutiveTransposes,
    AbsorbTransposeIntoMultiThreshold,
)
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd,
    CollapseRepeatedMul,
)
from finn.transformation.streamline.reorder import (
    MoveAddPastMul,
    MoveScalarMulPastMatMul,
    MoveScalarAddPastMatMul,
    MoveAddPastConv,
    MoveScalarMulPastConv,
    MoveScalarMulPastConvTranspose,  # ← needed for upsampling-branch Mul (scale before ConvTranspose)
    MoveMulPastMaxPool,               # ← needed for downsampling-branch Mul (scale before MaxPool)
    MoveScalarLinearPastInvariants,
    MoveMaxPoolPastMultiThreshold,
    MoveLinearPastEltwiseAdd,   # ← KEY for residuals
    MoveLinearPastFork,         # ← KEY for fork points
    MakeMaxPoolNHWC,            # ← needed before InferStreamingMaxPool
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.sign_to_thres import ConvertSignToThres
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.infer_pixel_padding_deconv import InferPixelPaddingDeconv
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig


# ── paths ────────────────────────────────────────────────────────────────────
# Usage: python finn_enet_ip_build.py [model_name] [auto_fifo_strategy]
#   model_name:         filename stem (no .onnx) under ENET_DIR, e.g. quantEnet_O8_native
#                        defaults to quantEnet_finn_v1 (the E1 config)
#   auto_fifo_strategy:  "largefifo_rtlsim" (FINN default, whole-network stitched-IP
#                        rtlsim to measure FIFO occupancy — accurate but assembles the
#                        full Vivado block design TWICE, once here and once more in
#                        step_create_stitched_ip) or "characterize" (per-node HLS-cosim
#                        based sizing — skips the duplicate whole-network BD assembly,
#                        roughly halving total build time; step_create_stitched_ip's
#                        BD assembly still happens once regardless, since that's how
#                        the final deliverable IP/OOC-synth report is produced).
#                        Defaults to largefifo_rtlsim for backward compatibility.
ENET_DIR  = "/home/thelegendiv/finn/notebooks/enet"
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "quantEnet_finn_v1"
FIFO_STRATEGY = sys.argv[2] if len(sys.argv) > 2 else "largefifo_rtlsim"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"stitched_ip_{MODEL_NAME}_{FIFO_STRATEGY}_{timestamp}")

# ── ZCU7EV resources (for reference) ──────────────────────────────────────────
# Real xczu7ev-ffvc1156-2-e (Zynq UltraScale+ ZU7EV) device numbers, per Xilinx
# DS891's detailed CLB resource table: 230,400 CLB LUTs (NOT the 504K "System
# Logic Cells" figure quoted in Xilinx's product-selection-guide summary table --
# that's a legacy 4-input-LUT-era marketing conversion, ~2.1875x the real LUT
# count, and is NOT what Vivado/FINN report utilization against), 1,728 DSP
# slices, 312 Block RAM tiles (36Kb each) = 624 in FINN's BRAM_18K reporting
# granularity.
XCZU7EV = {"LUT": 230400, "BRAM_18K": 624, "DSP": 1728}


# ── Custom step: tidy ─────────────────────────────────────────────────────────
def step_enet_tidy(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Standard tidy-up without InsertTopK (segmentation, not classification)."""
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    return model


# ── Custom step: streamline (linear pass) ─────────────────────────────────────
def _streamline_linear(model: ModelWrapper, cfg: DataflowBuildConfig):
    """One linear-streamlining pass (same as resnet50, minus TopK absorb)."""
    for trn in [
        ConvertSubToAdd(),
        ConvertDivToMul(),
        RemoveIdentityOps(),
        CollapseRepeatedMul(),
        BatchNormToAffine(),
        ConvertSignToThres(),
        MoveAddPastMul(),
        MoveScalarAddPastMatMul(),
        MoveAddPastConv(),
        MoveScalarMulPastMatMul(),
        MoveScalarMulPastConv(),
        MoveScalarMulPastConvTranspose(),  # scale past ConvTranspose (upsampling branch)
        MoveMulPastMaxPool(),              # scale past MaxPool (downsampling shortcut branch)
        MoveScalarLinearPastInvariants(),
        MoveAddPastMul(),
        CollapseRepeatedAdd(),
        CollapseRepeatedMul(),
        AbsorbAddIntoMultiThreshold(),
        FactorOutMulSignMagnitude(),
        MoveMaxPoolPastMultiThreshold(),
        AbsorbMulIntoMultiThreshold(),
        Absorb1BitMulIntoMatMul(),
        Absorb1BitMulIntoConv(),
        RoundAndClipThresholds(),
    ]:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


# ── Custom step: streamline (non-linear pass — key for residuals) ─────────────
def _streamline_nonlinear(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Push linear ops past fork points and eltwise-Add.

    ORDER MATTERS: MoveLinearPastFork MUST run before MoveLinearPastEltwiseAdd.
    See finn_enet_build.py for the full rationale (fork-node tensor-rename
    corruption if run in the other order).
    """
    for trn in [
        MoveLinearPastFork(),
        MoveLinearPastEltwiseAdd(),
    ]:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


def step_enet_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    """4-pass alternating linear + non-linear streamlining (same as resnet50).

    Also lowers Conv→MatMul here (like standard step_streamline) so that
    Im2Col output tensors are typed as integer by InferDataTypes *before*
    step_enet_convert_to_hw runs InferQuantizedMatrixVectorActivation.
    """
    for _iter in range(4):
        model = _streamline_linear(model, cfg)
        model = _streamline_nonlinear(model, cfg)
        # tidy after each iteration — intentionally NO GiveReadableTensorNames here.
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(InferDataTypes())
        model = model.transform(SortGraph())

    model = model.transform(DoubleToSingleFloat())

    # IMPORTANT: Do NOT call AbsorbTransposeIntoMultiThreshold() here — see
    # finn_enet_build.py for the phantom-FLOAT32-tensor rationale.
    if len(model.get_nodes_by_op_type("Conv")) > 0:
        model = model.transform(LowerConvsToMatMul())
        model = model.transform(MakeMaxPoolNHWC())             # NHWC for streaming
        # AbsorbTransposeIntoMultiThreshold intentionally omitted — see above
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(AbsorbConsecutiveTransposes())

    # ConvTranspose (upsampling branch) has no native FINN HW layer — see
    # finn_enet_build.py for the InferPixelPaddingDeconv rationale.
    if len(model.get_nodes_by_op_type("ConvTranspose")) > 0:
        model = model.transform(InferPixelPaddingDeconv())
        model = model.transform(AbsorbConsecutiveTransposes())

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())  # safe to rename once all transforms done
    model = model.transform(InferDataLayouts())
    model = model.transform(InferDataTypes())   # types Im2Col outputs here
    return model


# ── Custom step: convert to HW ────────────────────────────────────────────────
def step_enet_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Convert to HW layers.

    Conv lowering is done in step_enet_streamline, so Im2Col outputs are
    already typed as integer when we arrive here.
    """
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model = model.transform(InferDataLayouts())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())

    # QONNX's InferDataTypes does not know about ConvTranspose — see
    # finn_enet_build.py for the manual INT32 annotation rationale.
    for n in model.graph.node:
        if n.op_type == "ConvTranspose":
            idt = model.get_tensor_datatype(n.input[0])
            wdt = model.get_tensor_datatype(n.input[1])
            if idt.is_integer() and wdt.is_integer():
                model.set_tensor_datatype(n.output[0], DataType["INT32"])
    model = model.transform(InferDataTypes())

    for trn in [
        to_hw.InferAddStreamsLayer,            # residual Add → AddStreams_Batch
        to_hw.InferChannelwiseLinearLayer,
        to_hw.InferStreamingMaxPool,           # MaxPool (NHWC) → StreamingMaxPool
        RoundAndClipThresholds,
        to_hw.InferBinaryMatrixVectorActivation,
        to_hw.InferQuantizedMatrixVectorActivation,  # Im2Col+T+MatMul+MT → MVAU
        to_hw.InferVectorVectorActivation,     # depthwise MatMul (sparsity-annotated) → VVAU
        to_hw.InferThresholdingLayer,
        AbsorbConsecutiveTransposes,
        to_hw.InferConvInpGen,                 # Im2Col → ConvolutionInputGenerator
        to_hw.InferDuplicateStreamsLayer,
    ]:
        model = model.transform(trn())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())

    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())
    return model


# ── Build config ───────────────────────────────────────────────────────────────
# Extends the estimate-only pipeline with FINN's standard post-folding steps
# for real IP generation (see default_build_dataflow_steps in
# build_dataflow_config.py for the canonical ordering this mirrors), stopping
# right after step_out_of_context_synthesis — no step_synthesize_bitfile,
# step_make_pynq_driver, or step_deployment_package, since those need
# board=/shell_flow_type= (full PS/Zynq system), which we're deliberately not
# building.
enet_ip_steps = [
    "step_qonnx_to_finn",
    step_enet_tidy,
    step_enet_streamline,
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

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Model : {MODEL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"FIFO strategy: {FIFO_STRATEGY}")
    print(f"Steps : {[s if isinstance(s, str) else s.__name__ for s in enet_ip_steps]}")
    print()

    build.build_dataflow_cfg(MODEL_FILE, cfg_stitched_ip)

    # ── Report ───────────────────────────────────────────────────────────────
    import json

    report_dir = os.path.join(OUTPUT_DIR, "report")

    perf_json = os.path.join(report_dir, "estimate_network_performance.json")
    if os.path.exists(perf_json):
        print("\n" + "=" * 60)
        print("NETWORK PERFORMANCE ESTIMATES (analytical)")
        print("=" * 60)
        with open(perf_json) as f:
            for k, v in json.load(f).items():
                print(f"  {k}: {v}")

    res_json = os.path.join(report_dir, "estimate_layer_resources.json")
    if os.path.exists(res_json):
        print("\n" + "=" * 60)
        print("LAYER RESOURCES (analytical)")
        print("=" * 60)
        with open(res_json) as f:
            resources = json.load(f)
        totals = {"LUT": 0, "BRAM_18K": 0, "DSP": 0}
        for layer, r in resources.items():
            for k in totals:
                totals[k] += r.get(k, 0)
        for k in totals:
            print(f"  {k:9s}: {totals[k]}  /  {XCZU7EV[k]}  ({100*totals[k]/XCZU7EV[k]:.1f}%)")

    rtlsim_json = os.path.join(report_dir, "rtlsim_performance.json")
    if os.path.exists(rtlsim_json):
        print("\n" + "=" * 60)
        print("RTLSIM PERFORMANCE (cycle-accurate, Verilator)")
        print("=" * 60)
        with open(rtlsim_json) as f:
            for k, v in json.load(f).items():
                print(f"  {k}: {v}")
    else:
        print(f"\n[WARN] No rtlsim performance report at {rtlsim_json}")

    ooc_json = os.path.join(report_dir, "ooc_synth_and_timing.json")
    if os.path.exists(ooc_json):
        print("\n" + "=" * 60)
        print("OUT-OF-CONTEXT SYNTHESIS (real Vivado numbers)")
        print("=" * 60)
        with open(ooc_json) as f:
            for k, v in json.load(f).items():
                print(f"  {k}: {v}")
    else:
        print(f"\n[WARN] No OOC synthesis report at {ooc_json}")

    stitched_ip_dir = os.path.join(OUTPUT_DIR, "stitched_ip")
    if os.path.isdir(stitched_ip_dir):
        print("\n" + "=" * 60)
        print("STITCHED IP")
        print("=" * 60)
        print(f"  Location: {stitched_ip_dir}")
        top_entries = sorted(os.listdir(stitched_ip_dir))
        print(f"  Contents: {top_entries}")
    else:
        print(f"\n[WARN] No stitched_ip directory at {stitched_ip_dir}")

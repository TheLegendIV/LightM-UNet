"""FINN estimation build script for QuantENet — ZCU7EV target.

Run inside the FINN container:
    docker exec d345f89b4e6c python /tmp/finn_enet_build.py

Custom streamline + convert_to_hw modelled on finn-examples/build/resnet50/custom_steps.py
to correctly handle residual Add → AddStreams_Batch conversion.

Key fix vs standard step_streamline:
  MoveLinearPastEltwiseAdd + MoveLinearPastFork run in 4 alternating iterations,
  which pushes Mul/BN scale ops through residual branches so that after the Add
  both inputs are "pure" streams → InferAddStreamsLayer can convert them.
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
# Usage: python finn_enet_build.py [model_name]
#   model_name: filename stem (no .onnx) under ENET_DIR, e.g. quantEnet_O8_native
#               defaults to quantEnet_finn_v1 (the E1 config)
ENET_DIR  = "/home/thelegendiv/finn/notebooks/enet"
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "quantEnet_finn_v1"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"estimates_{MODEL_NAME}_{timestamp}")

# ── ZCU7EV resources (for reference) ─────────────────────────────────────────
# LUT: 48000 | BRAM: 216 | DSP: 192


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

    This is the critical transformation that makes residual connections
    compatible with FINN:
      Before:  branch → Mul(scale) → Add ← Mul(scale) ← branch
      After:   branch ──────────→ Add → Mul(scale)
    Both Add inputs are now scale-free → InferAddStreamsLayer can convert.

    ORDER MATTERS: MoveLinearPastFork MUST run before MoveLinearPastEltwiseAdd.
    MoveLinearPastEltwiseAdd.move_node() renames a Mul/Add producer's output
    tensor without checking whether that producer is a fork node (i.e. also
    feeds another consumer, such as a shortcut-branch Conv). If a forking
    Mul/Add is fed straight into MoveLinearPastEltwiseAdd first, the rename
    corrupts the other consumer's input reference, leaving a dangling tensor
    name (e.g. "Mul_10_out0") with no producer — which later makes
    LowerConvsToMatMul's Transpose/MatMul inputs FLOAT32 and blocks MVAU
    conversion. Running MoveLinearPastFork first duplicates any forking
    Mul/Add so each consumer gets its own private producer, avoiding the
    corruption entirely.
    """
    for trn in [
        MoveLinearPastFork(),
        MoveLinearPastEltwiseAdd(),
    ]:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


def step_enet_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    """8-pass alternating linear + non-linear streamlining (same as resnet50,
    but with double the iteration count -- see below).

    Also lowers Conv→MatMul here (like standard step_streamline) so that
    Im2Col output tensors are typed as integer by InferDataTypes *before*
    step_enet_convert_to_hw runs InferQuantizedMatrixVectorActivation.
    Without this, Im2Col outputs default to FLOAT32 and MVAU conversion
    is skipped for all lowered convolutions.

    Iteration count bumped from 4 to 8 (diagnosed on
    quantEnet_s13_leaky_frozen_int8): some deeper/more-complex fork+affine
    chains (Mul AND Add both present, not just a bare Mul, on a residual
    shortcut branch) need more than 4 alternating passes of
    MoveLinearPastFork/MoveLinearPastEltwiseAdd to fully cancel out before
    reaching step_enet_convert_to_hw -- leaving a leftover generic Mul/Add
    stranded in front of a residual join that InferAddStreamsLayer then
    refuses to convert (both inputs must already be "pure"/scale-free
    streams), which cascades into a non-HW island large enough to trip
    step_create_dataflow_partition's cycle-free assertion ("partition
    depends on itself"). Extra iterations on already-converged configs
    (E1/O8/S5) are safe no-ops since the transforms are idempotent once
    nothing more can be pushed/absorbed.
    """
    for _iter in range(8):
        model = _streamline_linear(model, cfg)
        model = _streamline_nonlinear(model, cfg)
        # tidy after each iteration — intentionally NO GiveReadableTensorNames here:
        # renaming tensor outputs without updating all consumer references creates
        # phantom tensors that break LowerConvsToMatMul's Transpose inputs.
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(InferDataTypes())
        model = model.transform(SortGraph())

    model = model.transform(DoubleToSingleFloat())

    # Lower Conv to MatMul here so InferDataTypes can type Im2Col outputs
    # before InferQuantizedMatrixVectorActivation needs them.
    #
    # IMPORTANT: Do NOT call AbsorbTransposeIntoMultiThreshold() here.
    # After LowerConvsToMatMul(), there are back-to-back NHWC→NCHW→NHWC
    # Transpose pairs: T1 (from MT output) + T2 (from LowerConvsToMatMul).
    # AbsorbTransposeIntoMultiThreshold removes T1, leaving T2's input as
    # a phantom FLOAT32 tensor with no producer → MatMul inputs become FLOAT32
    # → InferQuantizedMatrixVectorActivation refuses to convert them to MVAU.
    # Instead, let AbsorbConsecutiveTransposes cancel T1+T2 directly, so the
    # MatMul input becomes the MT output tensor (UINT8) → MVAU conversion works.
    if len(model.get_nodes_by_op_type("Conv")) > 0:
        model = model.transform(LowerConvsToMatMul())
        model = model.transform(MakeMaxPoolNHWC())             # NHWC for streaming
        # AbsorbTransposeIntoMultiThreshold intentionally omitted — see above
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(AbsorbConsecutiveTransposes())

    # ConvTranspose (upsampling branch) has no native FINN HW layer, and
    # leaving it as a generic ONNX op interleaved in the middle of the
    # network creates multiple disjoint non-HW islands that break
    # step_create_dataflow_partition's single-partition assumption
    # ("cycle-free graph violated"). InferPixelPaddingDeconv rewrites each
    # ConvTranspose (NCHW) into Transpose + FMPadding_Pixel (already an HW
    # custom op) + Im2Col + MatMul + Transpose (NHWC), matching the same
    # lowering LowerConvsToMatMul does for regular Conv. The resulting
    # Im2Col/MatMul then convert to ConvolutionInputGenerator/MVAU in
    # step_enet_convert_to_hw exactly like regular convolutions.
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

    # QONNX's InferDataTypes does not know about ConvTranspose, so its output
    # tensor is always left as FLOAT32 even when input/weight are integer.
    # This blocks InferThresholdingLayer for any MultiThreshold that directly
    # follows a ConvTranspose (the upsampling-block main branch). Manually
    # annotate the output as INT32 (matching the convention used for MVAU
    # accumulator outputs) whenever input+weight are both integer.
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


# ── Build config ──────────────────────────────────────────────────────────────
enet_estimate_steps = [
    "step_qonnx_to_finn",
    step_enet_tidy,
    step_enet_streamline,
    step_enet_convert_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",       # HW → HLS/RTL backend variants; required before folding/perf analysis
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
]

cfg_estimates = DataflowBuildConfig(
    output_dir          = OUTPUT_DIR,
    mvau_wwidth_max     = 80,
    target_fps          = 1000,
    synth_clk_period_ns = 10.0,
    fpga_part           = "xczu7ev-ffvc1156-2-e",
    steps               = enet_estimate_steps,
    generate_outputs    = [build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
    save_intermediate_models = True,
)

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Model : {MODEL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Steps : {[s if isinstance(s, str) else s.__name__ for s in enet_estimate_steps]}")
    print()

    build.build_dataflow_cfg(MODEL_FILE, cfg_estimates)

    # ── Report ────────────────────────────────────────────────────────────────
    report_dir = os.path.join(OUTPUT_DIR, "report")
    perf_json  = os.path.join(report_dir, "estimate_network_performance.json")
    cycles_json = os.path.join(report_dir, "estimate_layer_cycles.json")
    res_json   = os.path.join(report_dir, "estimate_layer_resources.json")

    import json

    if os.path.exists(perf_json):
        print("\n" + "=" * 60)
        print("NETWORK PERFORMANCE ESTIMATES")
        print("=" * 60)
        with open(perf_json) as f:
            perf = json.load(f)
        for k, v in perf.items():
            print(f"  {k}: {v}")
    else:
        print(f"\n[WARN] No performance report at {perf_json}")

    if os.path.exists(cycles_json):
        print("\n" + "=" * 60)
        print("LAYER CYCLES")
        print("=" * 60)
        with open(cycles_json) as f:
            cycles = json.load(f)
        total = 0
        for layer, c in cycles.items():
            print(f"  {layer}: {c}")
            total += c
        print(f"  TOTAL: {total}")

    if os.path.exists(res_json):
        print("\n" + "=" * 60)
        print("LAYER RESOURCES")
        print("=" * 60)
        # Real xczu7ev-ffvc1156-2-e (Zynq UltraScale+ ZU7EV) device numbers, per
        # Xilinx DS891: 504,000 LUTs, 1,728 DSP slices, 312 Block RAM tiles
        # (36Kb each) = 624 in FINN's BRAM_18K reporting granularity. NOT the
        # same part as whatever the previous 48000/216/192 placeholder budget
        # was based on -- that was wrong by ~10x on LUT/DSP.
        XCZU7EV = {"LUT": 504000, "BRAM_18K": 624, "DSP": 1728}
        with open(res_json) as f:
            resources = json.load(f)
        totals = {"LUT": 0, "BRAM_18K": 0, "DSP": 0}
        for layer, r in resources.items():
            for k in totals:
                totals[k] += r.get(k, 0)
        print(f"  LUT      : {totals['LUT']}  /  {XCZU7EV['LUT']}  ({100*totals['LUT']/XCZU7EV['LUT']:.1f}%)")
        print(f"  BRAM_18K : {totals['BRAM_18K']}  /  {XCZU7EV['BRAM_18K']}  ({100*totals['BRAM_18K']/XCZU7EV['BRAM_18K']:.1f}%)")
        print(f"  DSP      : {totals['DSP']}  /  {XCZU7EV['DSP']}  ({100*totals['DSP']/XCZU7EV['DSP']:.1f}%)")

    # ── Post-HW diagnostic ────────────────────────────────────────────────────
    hw_model_path = os.path.join(OUTPUT_DIR, "intermediate_models", "step_enet_convert_to_hw.onnx")
    if not os.path.exists(hw_model_path):
        # try with standard name
        hw_model_path = os.path.join(OUTPUT_DIR, "intermediate_models", "step_convert_to_hw.onnx")
    if os.path.exists(hw_model_path):
        print("\n" + "=" * 60)
        print("HW CONVERSION DIAGNOSTIC")
        print("=" * 60)
        DOMAIN = "finn.custom_op.fpgadataflow"
        HW_OPS = {"MVAU", "Thresholding", "FMPadding", "ConvolutionInputGenerator",
                  "StreamingMaxPool", "ConvTranspose", "AddStreams", "AddStreams_Batch",
                  "FMPadding_Batch", "StreamingDataWidthConverter", "StreamingFIFO"}
        m = ModelWrapper(hw_model_path)
        hw, bad = {}, {}
        for n in m.graph.node:
            is_hw = (n.op_type in HW_OPS) or (n.domain == DOMAIN)
            d = hw if is_hw else bad
            d[n.op_type] = d.get(n.op_type, 0) + 1
        print(f"  Total: {len(m.graph.node)}  HW: {sum(hw.values())}  Non-HW: {sum(bad.values())}")
        print(f"  HW ops : {dict(sorted(hw.items()))}")
        if bad:
            print(f"  BAD ops: {dict(sorted(bad.items()))}")
        else:
            print("  All ops converted to HW!")

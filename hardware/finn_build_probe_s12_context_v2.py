"""Generalized FINN build (through real stitched IP + automatic OOC synthesis)
for the S12 dense-vs-separable context-block probe pair, parameterized by
bit width and folding scheme -- generalizes finn_build_probe_s12_context_int4.py
(kept as-is/untouched for the original INT4 baseline) to also cover INT6/INT8
variants, plus an explicit PE=MH,SIMD=1 ("fully unrolled output channels,
fully serial input channels") forced-folding mode -- the specific folding
shape identified (see repo memory finn_gotchas.md's "KEY FINDING" 2026-09-15
entries) as the strongest real-vs-analytical-estimate LUT blowup correlate in
the production dense_relu_calibrated build.

Byte-for-byte clone of finn_build_probe_s12_context_int4.py's step-list
skeleton otherwise (same tidy/streamline/convert_to_hw pattern), with two
differences:
  1. --bitwidth {6,8} selects the matching probe_s12_context_{variant}_int{bw}.onnx
     export (must already exist -- run the matching
     finn_export_probe_s12_context_{variant}_int{bw}.py first).
  2. --force-pe-mh-simd1 inserts step_force_pe_eq_mh_simd_1 right after
     step_apply_folding_config (a no-op here either way, since target_fps=None
     and no folding_config_file is passed) -- always divisibility-safe
     (PE=MH trivially divides MH, SIMD=1 trivially divides MW), unlike
     arbitrary PE/SIMD values which need the divisor-clamping dance documented
     in finn_gotchas.md.

Also automates what finn_ooc_probe_s12_context_synth.py used to require a
separate manual invocation for: right after build.build_dataflow_cfg()
returns (i.e. stitching has finished), this script calls run_ooc_synth()
directly on the same OUTPUT_DIR -- no manual second step needed.

Run inside the FINN container, e.g.:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_build_probe_s12_context_v2.py \\
        dense --bitwidth 8 --force-pe-mh-simd1
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/brevitas/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/pyverilator")
sys.path.insert(0, "/home/thelegendiv/finn/deps/finn-experimental")

_XILINX_BIN_DIRS = [
    "/tools/Xilinx/Vitis_HLS/2022.2/bin",
    "/tools/Xilinx/Vivado/2022.2/bin",
]
os.environ["PATH"] = os.pathsep.join(_XILINX_BIN_DIRS + [os.environ.get("PATH", "")])
os.environ.setdefault("XILINX_VIVADO", "/tools/Xilinx/Vivado/2022.2")
os.environ.setdefault("XILINX_HLS", "/tools/Xilinx/Vitis_HLS/2022.2")

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
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
    MoveMulPastMaxPool,
    MoveScalarLinearPastInvariants,
    MoveMaxPoolPastMultiThreshold,
    MoveLinearPastEltwiseAdd,
    MoveLinearPastFork,
    MakeMaxPoolNHWC,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.sign_to_thres import ConvertSignToThres
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig

from finn_ooc_probe_s12_context_synth import run_ooc_synth

ap = argparse.ArgumentParser()
ap.add_argument("variant", choices=["dense", "separable"])
ap.add_argument("--bitwidth", type=int, choices=[6, 8], required=True)
ap.add_argument("--force-pe-mh-simd1", action="store_true",
                 help="Force PE=MH, SIMD=1 (fully unrolled output channels) on every "
                      "MVAU node instead of leaving FINN's node-default PE=SIMD=1.")
args = ap.parse_args()

VARIANT = args.variant
BITWIDTH = args.bitwidth
FORCE_PE_MH_SIMD1 = args.force_pe_mh_simd1

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
MODEL_NAME = f"probe_s12_context_{VARIANT}_int{BITWIDTH}"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
suffix = "_pemh_simd1" if FORCE_PE_MH_SIMD1 else ""
OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"{MODEL_NAME}{suffix}_{timestamp}")

# Real xczu7ev part directly -- no board= (no PS/DDR/DMA needed for a
# stitched-IP-only probe), same convention as the existing probes.
FPGA_PART = "xczu7ev-ffvc1156-2-e"


def step_reapply_unique_names(model: ModelWrapper, cfg: DataflowBuildConfig):
    return model.transform(GiveUniqueNodeNames())


def step_force_dsp(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Force resType=dsp on every MVAU/VVAU node (dense variant has MVAU
    only, separable variant also has VVAU); ram_style left at default (auto)."""
    n_dsp = 0
    for node in model.graph.node:
        if "MVAU" in node.op_type or "VVAU" in node.op_type:
            getCustomOp(node).set_nodeattr("resType", "dsp")
            n_dsp += 1
    print(f"[step_force_dsp] forced resType=dsp on {n_dsp} MVAU/VVAU node(s), ram_style left at default (auto)")
    return model


def step_force_pe_eq_mh_simd_1(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Force PE=MH (fully unrolled output channels), SIMD=1 on every MVAU
    node -- always divisibility-safe (PE=MH trivially divides MH, SIMD=1
    trivially divides MW), unlike arbitrary PE/SIMD choices."""
    n = 0
    for node in model.graph.node:
        if "MVAU" in node.op_type:
            inst = getCustomOp(node)
            mh = inst.get_nodeattr("MH")
            inst.set_nodeattr("PE", mh)
            inst.set_nodeattr("SIMD", 1)
            n += 1
    print(f"[step_force_pe_eq_mh_simd_1] forced PE=MH, SIMD=1 on {n} MVAU node(s)")
    return model


def step_fix_signed_thresholds(model: ModelWrapper, cfg: DataflowBuildConfig):
    """QuantIdentityHandler (qonnx_activation_handlers.py) REQUIRES signed=1 on
    any 'identity' Quant node it converts to MultiThreshold (hard-fails
    otherwise) -- but this probe's QuantRegularBottleneck.input_quant is an
    identity Quant chained directly onto the PRECEDING block's already-
    unsigned/non-negative ReLU output (no arithmetic in between), so its real
    values never go negative even though FINN forced it signed at conversion
    time. That mismatch then trips a LATER assertion in
    convert_to_hw_layers.py's InferThresholdingLayer: "assert (not odt.signed())
    or (actval < 0)" -- where actval is the MultiThreshold node's own
    'out_bias' attribute (NOT the raw threshold values -- checked and ruled
    out first). Mirror FINN's own condition exactly: any MultiThreshold node
    annotated signed whose out_bias is >= 0 gets its output datatype
    downgraded to the equivalent unsigned type (matches its real,
    always-non-negative value range: out_bias >= 0 means the comparator
    never needed to represent a negative result)."""
    n_fixed = 0
    for node in model.graph.node:
        if node.op_type != "MultiThreshold":
            continue
        out_name = node.output[0]
        dt = model.get_tensor_datatype(out_name)
        if not dt.signed():
            continue
        out_bias = getCustomOp(node).get_nodeattr("out_bias")
        if out_bias < 0:
            continue  # genuinely needs a signed output -- leave it alone
        model.set_tensor_datatype(out_name, DataType[f"UINT{dt.bitwidth()}"])
        n_fixed += 1
    print(f"[step_fix_signed_thresholds] downgraded {n_fixed} MultiThreshold node(s) signed->unsigned (out_bias >= 0)")
    return model


def step_probe_tidy(model: ModelWrapper, cfg: DataflowBuildConfig):
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


def _streamline_linear(model: ModelWrapper, cfg: DataflowBuildConfig):
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
        MoveMulPastMaxPool(),
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


def _streamline_nonlinear(model: ModelWrapper, cfg: DataflowBuildConfig):
    for trn in [
        MoveLinearPastFork(),
        MoveLinearPastEltwiseAdd(),
    ]:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


def step_probe_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    for _iter in range(4):
        model = _streamline_linear(model, cfg)
        model = _streamline_nonlinear(model, cfg)
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(InferDataTypes())
        model = model.transform(SortGraph())

    model = model.transform(DoubleToSingleFloat())

    if len(model.get_nodes_by_op_type("Conv")) > 0:
        model = model.transform(LowerConvsToMatMul())
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(AbsorbConsecutiveTransposes())

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataLayouts())
    model = model.transform(InferDataTypes())
    return model


def step_probe_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig):
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model = model.transform(InferDataLayouts())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())

    for trn in [
        to_hw.InferAddStreamsLayer,
        to_hw.InferChannelwiseLinearLayer,
        to_hw.InferStreamingMaxPool,
        RoundAndClipThresholds,
        to_hw.InferBinaryMatrixVectorActivation,
        to_hw.InferQuantizedMatrixVectorActivation,
        to_hw.InferVectorVectorActivation,
        to_hw.InferThresholdingLayer,
        AbsorbConsecutiveTransposes,
        to_hw.InferConvInpGen,
        to_hw.InferDuplicateStreamsLayer,
    ]:
        if trn is to_hw.InferThresholdingLayer:
            # InferDataTypes() re-runs after every transform in this loop and
            # re-derives (signed) datatypes straight from the still-signed
            # Quant-derived annotations -- undoing step_fix_signed_thresholds
            # if it only ran once, earlier. Re-apply right before the transform
            # that actually asserts on it.
            model = step_fix_signed_thresholds(model, cfg)
        model = model.transform(trn())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())

    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())
    return model


probe_steps = [
    "step_qonnx_to_finn",
    step_fix_signed_thresholds,
    step_probe_tidy,
    step_probe_streamline,
    step_probe_convert_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    step_reapply_unique_names,
    step_force_dsp,
    "step_target_fps_parallelization",
    "step_apply_folding_config",
]
if FORCE_PE_MH_SIMD1:
    probe_steps.append(step_force_pe_eq_mh_simd_1)
probe_steps += [
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_measure_rtlsim_performance",
]

cfg_probe = DataflowBuildConfig(
    output_dir          = OUTPUT_DIR,
    mvau_wwidth_max     = 80,
    target_fps          = None,   # PE=SIMD=1 unless overridden -- see step_force_pe_eq_mh_simd_1 above
    synth_clk_period_ns = 10.0,
    split_large_fifos   = True,
    fpga_part           = FPGA_PART,
    steps               = probe_steps,
    generate_outputs    = [
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
    ],
    save_intermediate_models = True,
)

if __name__ == "__main__":
    print(f"Variant: {VARIANT}")
    print(f"Bitwidth: {BITWIDTH}")
    print(f"Force PE=MH,SIMD=1: {FORCE_PE_MH_SIMD1}")
    print(f"Model : {MODEL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Part  : {FPGA_PART}")
    print(f"Steps : {[s if isinstance(s, str) else s.__name__ for s in probe_steps]}")
    print(flush=True)

    build.build_dataflow_cfg(MODEL_FILE, cfg_probe)

    print("Stitched IP / reports in:", OUTPUT_DIR)
    print("OUTPUT_DIR=", OUTPUT_DIR)

    print(f"\n[auto-ooc-synth] stitching finished -- starting OOC synthesis automatically", flush=True)
    run_ooc_synth(OUTPUT_DIR)
    print("Done (build + OOC synth).")

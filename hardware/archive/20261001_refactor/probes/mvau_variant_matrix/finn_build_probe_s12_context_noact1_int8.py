"""FINN build (through real stitched IP + automatic OOC synthesis) for the
noActivation=1 (UNFUSED, standalone Thresholding_rtl) side of the MVAU-type/
resType/noAct/PE-SIMD combination matrix -- see finn_build_probe_lutmult_
noact0.py for the noActivation=0 (fused) side, and the plan this was built
from, C:\\Users\\win32\\.claude\\plans\\the-current-ilp-inherited-lynx.md.

Uses probe_noact1_single_d16_int8.onnx (finn_export_probe_noact1_single_
int8.py) -- a single QuantRegularBottleneck (channels=32, dilation=16),
residual join intact, deliberately matching finn_export_probe_lutmult_
noact0.py's geometry exactly (in/out=32ch, internal=8, kernel=3, dilation=16,
W8A8) so the noAct=0 vs noAct=1 arms of the full matrix differ ONLY in the
axis under test. QuantRegularBottleneck's residual join reliably lands
noActivation=1 on every layer (that's how the original INT6 7-row
mvau_lut_calibration_dataset_s12_context_dense_int6_pemh_simd1.csv reference
dataset was built, on the older/larger 2-block probe_s12_context_dense_int8.
onnx -- see finn_export_probe_lutmult_noact0.py's own docstring for why a
residual join specifically is what blocks the automatic fusion the noAct=0
sibling needs; that older export is left untouched, this uses a new, smaller
one instead).

Byte-for-byte clone of finn_build_probe_s12_context_v2.py's step-list
skeleton (tidy/streamline/convert_to_hw verbatim), with THREE differences:
  1. --impl-style {auto,hls} -- "auto" reproduces the ORIGINAL v2 script's
     behavior exactly (FINN's own _mvu_rtl_possible() picks RTL automatically
     whenever noActivation=1 + signed weights + bit-width<=8, which this INT8
     network satisfies) via step_force_res_type only (resType is pinned to
     "dsp" regardless of --res-type in this mode -- RTL structurally rejects
     resType="lut", see matrixvectoractivation_rtl.py's own assertion, and
     RTL's dsp_estimation()/lut_estimation() don't read resType at all beyond
     that rejection check, so there is nothing for --res-type=lut to mean
     here). "hls" adds step_force_impl_style, setting preferred_impl_style=
     "hls" on every MVAU node BEFORE step_specialize_layers -- confirmed via
     direct FINN source read (specialize_layers.py:47-49,113-131:
     _determine_impl_style() checks node_inst.get_nodeattr("preferred_impl_
     style") FIRST and takes it verbatim when hls_variant exists, bypassing
     _mvu_rtl_possible() entirely) -- forcing the SAME noActivation=1 nodes
     onto the HLS backend instead, where --res-type={dsp,lut} is then a real,
     independent choice (mirrors finn_build_probe_lutmult_noact0.py's own
     step_force_res_type).
  2. --res-type {dsp,lut} -- only a real choice under --impl-style=hls (see
     above); under --impl-style=auto it's accepted for CLI-shape consistency
     with the noAct=0 build script but always resolves to "dsp".
  3. --fold {pemh_simd1,balanced,pe1_simdmw} -- same three folding-forcing
     steps as finn_build_probe_lutmult_noact0.py (duplicated here rather than
     imported, matching this directory's own established clone-per-script
     convention -- see e.g. finn_build_probe_s12_context_v2.py's own
     docstring calling itself a "byte-for-byte clone" of the int4 script).

Legal combination count for this script: --impl-style=auto x 3 folds (3
builds, res-type ignored) + --impl-style=hls x {dsp,lut} x 3 folds (6
builds) = 9 real Vivado OOC synthesis builds. Together with finn_build_
probe_lutmult_noact0.py's 6 builds, that's the full 15-combination matrix
(MVAU type x forcedsp x noAct x PE/SIMD, minus the 9 illegal rtl+lut /
rtl+noact0 cells) at matched INT8/geometry.

Run inside the FINN container, e.g.:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_build_probe_s12_context_noact1_int8.py \\
        --impl-style hls --res-type lut --fold balanced
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_build_probe_s12_context_noact1_int8.py \\
        --impl-style auto --fold pe1_simdmw
Then extract real per-node numbers the same way as every other probe (see
build_probe_calibration_csv.py's own docstring) -- one label per combination,
e.g. probe_noact1_hls_lut_balanced_int8 / probe_noact1_auto_pe1simdmw_int8.
"""
import argparse
import math
import os
import sys
from datetime import datetime

# hardware/probes/ (parent dir) -- finn_ooc_probe_s12_context_synth.py lives
# there, shared with the other probe families, not duplicated into this
# subfolder. Irrelevant once deployed to the FINN container (everything gets
# docker cp-ed into one flat directory regardless of local folder structure,
# see this repo's own established convention), but needed for any run
# directly against this repo's own folder layout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
ap.add_argument("--impl-style", choices=["auto", "hls"], required=True,
                 help="auto: let FINN's own _mvu_rtl_possible() pick RTL (matches the original v2 script's "
                      "behavior exactly -- this INT8/noActivation=1 network qualifies). hls: force "
                      "preferred_impl_style=hls on every MVAU node, overriding RTL eligibility.")
ap.add_argument("--res-type", choices=["dsp", "lut"], default="dsp",
                 help="Only a real choice under --impl-style=hls. Ignored (always resolves to dsp) under "
                      "--impl-style=auto -- RTL structurally rejects resType=lut.")
ap.add_argument("--fold", choices=["pemh_simd1", "balanced", "pe1_simdmw"], required=True,
                 help="Folding point: pemh_simd1 (PE=MH,SIMD=1, degenerate high-PE), "
                      "pe1_simdmw (PE=1,SIMD=MW, degenerate high-SIMD), or "
                      "balanced (PE=SIMD=gcd(MH,MW) per node).")
ap.add_argument("--thresh-mem", choices=["auto", "distributed"], default="auto",
                 help="auto: leave Thresholding_rtl's depth_trigger_bram/uram at their default 0 (Vivado's "
                      "own 'auto' inference picks BRAM for deep threshold tables -- this is the pre-existing "
                      "behavior). distributed: force depth_trigger_bram to a large sentinel (no real table "
                      "in this probe is ever that deep) on every Thresholding_rtl node, which -- per "
                      "finn-rtllib/thresholding/hdl/thresholding.sv's own RAM_STYLE ternary (DEPTH_TRIGGER_URAM "
                      "&& DEPTH>=that ? ultra : DEPTH_TRIGGER_BRAM && DEPTH>=that ? block : DEPTH_TRIGGER_BRAM "
                      "&& DEPTH>=64 ? distributed : auto) -- resolves every DEPTH>=64 memory to explicit "
                      "'distributed' (LUTRAM) instead of leaving it to Vivado's auto heuristic. "
                      "depth_trigger_uram is left at 0 always (nonzero URAM triggers are what crashed Vivado "
                      "2022.2 in the depth_trigger_uram=1 regression documented in hardware/results.csv / "
                      "hardware/temp/revert_thresh_uram*.sh -- that bug is unrelated to this BRAM-trigger path).")
args = ap.parse_args()

IMPL_STYLE = args.impl_style
RES_TYPE = args.res_type if IMPL_STYLE == "hls" else "dsp"
FOLD = args.fold
THRESH_MEM = args.thresh_mem
THRESH_MEM_BRAM_TRIGGER_SENTINEL = 999999  # any real DEPTH is far smaller; only gates the >=64 distributed branch

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
MODEL_NAME = "probe_noact1_single_d16_int8"  # finn_export_probe_noact1_single_int8.py -- single bottleneck, matches the noAct=0 sibling's geometry exactly
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(
    ENET_DIR, "finn_deployment_outputs",
    f"probe_noact1_{IMPL_STYLE}_{RES_TYPE}_{FOLD}_thresh{THRESH_MEM}_int8_{timestamp}",
)

FPGA_PART = "xczu7ev-ffvc1156-2-e"


def step_reapply_unique_names(model: ModelWrapper, cfg: DataflowBuildConfig):
    return model.transform(GiveUniqueNodeNames())


def step_force_impl_style(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Sets preferred_impl_style=hls on every MVAU/VVAU node, overriding
    FINN's own RTL-eligibility auto-detection (specialize_layers.py's
    _determine_impl_style() reads this nodeattr FIRST -- see module
    docstring point 1). MUST run before "step_specialize_layers". No-op
    (not inserted at all) when --impl-style=auto -- see probe_steps below."""
    n = 0
    for node in model.graph.node:
        if node.op_type in ("MVAU", "VVAU"):
            getCustomOp(node).set_nodeattr("preferred_impl_style", "hls")
            n += 1
    print(f"[step_force_impl_style] forced preferred_impl_style=hls on {n} MVAU/VVAU node(s)")
    return model


def step_print_noactivation_diagnostic(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Verification, not a transform -- confirms every MVAU/VVAU node landed
    noActivation=1 (this network's own premise, inherited from the existing
    s12_context probe topology) and which backend it actually got, right
    after step_specialize_layers."""
    print("[step_print_noactivation_diagnostic] per-node backend/noActivation after specialize_layers:")
    for node in model.graph.node:
        if "MVAU" in node.op_type or "VVAU" in node.op_type:
            inst = getCustomOp(node)
            no_act = inst.get_nodeattr("noActivation")
            print(f"    {node.name} ({node.op_type}): noActivation={no_act}"
                  + ("  <-- UNEXPECTED: this network should be noActivation=1 everywhere" if no_act == 0 else ""))
    return model


def step_force_res_type(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Force resType={RES_TYPE} on every MVAU/VVAU node; ram_style left at
    default (auto). RES_TYPE is pinned to "dsp" under --impl-style=auto (see
    module-level RES_TYPE derivation above) -- harmless/correct either way
    since RTL nodes only ever accept resType != "lut"."""
    n = 0
    for node in model.graph.node:
        if "MVAU" in node.op_type or "VVAU" in node.op_type:
            getCustomOp(node).set_nodeattr("resType", RES_TYPE)
            n += 1
    print(f"[step_force_res_type] forced resType={RES_TYPE} on {n} MVAU/VVAU node(s), ram_style left at default (auto)")
    return model


def step_force_thresh_ram_style(model: ModelWrapper, cfg: DataflowBuildConfig):
    """No-op unless --thresh-mem=distributed. Forces depth_trigger_bram to a
    large sentinel on every Thresholding_rtl node -- per finn-rtllib/
    thresholding/hdl/thresholding.sv's RAM_STYLE ternary (read directly from
    the container's source, see conversation for the exact line numbers),
    this makes every per-stage threshold memory of DEPTH>=64 resolve to
    explicit "distributed" (LUTRAM) instead of Vivado's "auto" heuristic
    (which tends to pick BRAM for deeper tables -- the pre-existing
    behavior, confirmed by "ram_style_chosen": "block" entries in this
    matrix's own analytical-estimate JSONs). depth_trigger_uram is left
    untouched at its default 0 -- forcing that one to a small nonzero value
    (depth_trigger_uram=1) is what crashed Vivado 2022.2 in an earlier,
    unrelated regression (see hardware/results.csv /
    hardware/temp/revert_thresh_uram*.sh); this step never touches it.
    MUST run after step_specialize_layers (node op_type must already be
    Thresholding_rtl, not the pre-specialization generic Thresholding) and
    before step_hw_ipgen (so the forced attr is baked into the generated
    Verilog's DEPTH_TRIGGER_BRAM parameter)."""
    if THRESH_MEM != "distributed":
        print("[step_force_thresh_ram_style] --thresh-mem=auto, no-op (depth_trigger_bram left at default 0)")
        return model
    n = 0
    for node in model.graph.node:
        if node.op_type == "Thresholding_rtl":
            getCustomOp(node).set_nodeattr("depth_trigger_bram", THRESH_MEM_BRAM_TRIGGER_SENTINEL)
            n += 1
    print(f"[step_force_thresh_ram_style] forced depth_trigger_bram={THRESH_MEM_BRAM_TRIGGER_SENTINEL} "
          f"(-> RAM_STYLE=distributed for DEPTH>=64) on {n} Thresholding_rtl node(s), depth_trigger_uram left at 0")
    return model


def step_force_pe_mh_simd1(model: ModelWrapper, cfg: DataflowBuildConfig):
    """PE=MH, SIMD=1 -- degenerate high-PE extreme. Always divisibility-safe."""
    n = 0
    for node in model.graph.node:
        if "MVAU" in node.op_type:
            inst = getCustomOp(node)
            mh = inst.get_nodeattr("MH")
            inst.set_nodeattr("PE", mh)
            inst.set_nodeattr("SIMD", 1)
            n += 1
    print(f"[step_force_pe_mh_simd1] forced PE=MH, SIMD=1 on {n} MVAU node(s)")
    return model


def step_force_pe1_simd_mw(model: ModelWrapper, cfg: DataflowBuildConfig):
    """PE=1, SIMD=MW -- mirror-image degenerate high-SIMD extreme. Always
    divisibility-safe."""
    n = 0
    for node in model.graph.node:
        if "MVAU" in node.op_type:
            inst = getCustomOp(node)
            mw = inst.get_nodeattr("MW")
            inst.set_nodeattr("PE", 1)
            inst.set_nodeattr("SIMD", mw)
            n += 1
    print(f"[step_force_pe1_simd_mw] forced PE=1, SIMD=MW on {n} MVAU node(s)")
    return model


def step_force_balanced_fold(model: ModelWrapper, cfg: DataflowBuildConfig):
    """PE=SIMD=gcd(MH,MW), computed per node -- see finn_build_probe_
    lutmult_noact0.py's own docstring for why this (not PE=SIMD=1) is the
    genuinely-intermediate point. Always divisibility-safe (gcd is always a
    common divisor of both)."""
    n = 0
    for node in model.graph.node:
        if "MVAU" in node.op_type:
            inst = getCustomOp(node)
            mh, mw = inst.get_nodeattr("MH"), inst.get_nodeattr("MW")
            d = math.gcd(mh, mw)
            inst.set_nodeattr("PE", d)
            inst.set_nodeattr("SIMD", d)
            n += 1
    print(f"[step_force_balanced_fold] forced PE=SIMD=gcd(MH,MW) on {n} MVAU node(s)")
    return model


_FOLD_STEPS = {
    "pemh_simd1": step_force_pe_mh_simd1,
    "pe1_simdmw": step_force_pe1_simd_mw,
    "balanced": step_force_balanced_fold,
}


def step_fix_signed_thresholds(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Verbatim from finn_build_probe_s12_context_v2.py -- see that script's
    own docstring for the exact assertion this works around."""
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
            continue
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
]
if IMPL_STYLE == "hls":
    probe_steps.append(step_force_impl_style)  # MUST run before step_specialize_layers
probe_steps += [
    "step_specialize_layers",
    step_reapply_unique_names,
    step_print_noactivation_diagnostic,
    step_force_res_type,
    step_force_thresh_ram_style,
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    _FOLD_STEPS[FOLD],
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
    target_fps          = None,   # PE=SIMD=1 unless overridden -- see _FOLD_STEPS above
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
    print(f"Impl style: {IMPL_STYLE}")
    print(f"resType   : {RES_TYPE}" + ("  (pinned -- --impl-style=auto always uses dsp)" if IMPL_STYLE == "auto" else ""))
    print(f"Fold      : {FOLD}")
    print(f"Model     : {MODEL_FILE}")
    print(f"Output    : {OUTPUT_DIR}")
    print(f"Part      : {FPGA_PART}")
    print(f"Steps     : {[s if isinstance(s, str) else s.__name__ for s in probe_steps]}")
    print(flush=True)

    build.build_dataflow_cfg(MODEL_FILE, cfg_probe)

    print("Stitched IP / reports in:", OUTPUT_DIR)
    print("OUTPUT_DIR=", OUTPUT_DIR)

    print(f"\n[auto-ooc-synth] stitching finished -- starting OOC synthesis automatically", flush=True)
    run_ooc_synth(OUTPUT_DIR)
    print("Done (build + OOC synth).")

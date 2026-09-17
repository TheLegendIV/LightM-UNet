"""FINN build (through real stitched IP + automatic OOC synthesis) for the
hls_lut_noact0/hls_dsp_noact0 ILP-variant probe (finn_export_probe_lutmult_
noact0.py's sequential, no-residual network) -- see that script's own
docstring for WHY this network exists (fusion-guaranteed, unlike the
QuantRegularBottleneck-based s12_context probe) and the plan this was built
from, C:\\Users\\win32\\.claude\\plans\\the-current-ilp-inherited-lynx.md.

Byte-for-byte clone of finn_build_probe_s12_context_v2.py's step-list
skeleton (same tidy/streamline/convert_to_hw pattern, verbatim -- none of
those steps are network-specific), with THREE differences:
  1. --res-type {dsp,lut} replaces the old hardcoded step_force_dsp with a
     parameterized step_force_res_type -- this is the axis the whole probe
     exists to isolate (does resType=lut cost what finn_cost_model.py's
     mult_luts formula predicts, at fixed noActivation=0?).
  2. --fold {pemh_simd1,balanced,pe1_simdmw} selects ONE of three folding-
     forcing steps instead of v2's single --force-pe-mh-simd1 flag -- see
     the mvau_lut_correlation_report.txt finding that real LUT for the
     noActivation=0/resType=dsp regime is driven by PE (not the mult/
     addertree/acc structural formula) and NEGATIVELY correlated with SIMD:
     pemh_simd1 (PE=MH,SIMD=1) is the degenerate high-PE extreme that
     finding condemns; pe1_simdmw (PE=1,SIMD=MW) is the mirror-image safe
     extreme; balanced (PE=SIMD=gcd(MH,MW), computed per-node) anchors a
     genuinely intermediate, non-degenerate point. 2x3 = 6 real Vivado OOC
     synthesis builds total for the full sweep.
  3. A new step_print_noactivation_diagnostic, inserted right after
     step_reapply_unique_names (i.e. right after step_specialize_layers) --
     THIS SESSION CANNOT RUN FINN/Vivado, so this print is the first real
     confirmation that every MVAU node actually landed noActivation=0 (the
     whole premise finn_export_probe_lutmult_noact0.py's sequential topology
     is designed to guarantee). If any node prints noActivation=1 here, the
     network's structural assumption needs revisiting for that node BEFORE
     trusting anything downstream.

Run inside the FINN container, once per (--res-type, --fold) pair, e.g.:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_build_probe_lutmult_noact0.py \\
        --res-type lut --fold balanced
Then extract real per-node numbers the same way as every other probe in this
directory (see build_probe_calibration_csv.py's own docstring): a Vivado
`report_utilization -hierarchical -hierarchical_depth 4` on the routed .dcp,
plus dump_node_attrs.py's landed-nodeattrs JSON, both saved as
_tmp_hier_<label>.rpt / _tmp_attrs_<label>.json, then:
    python3 build_probe_calibration_csv.py probe_lutmult_noact0_lut_balanced \\
        mvau_lut_calibration_dataset_lutmult_noact0_lut_balanced.csv
(one <label>/output CSV per (--res-type, --fold) combination -- 6 total for
the full sweep).
"""
import argparse
import math
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
ap.add_argument("--res-type", choices=["dsp", "lut"], required=True,
                 help="resType forced on every MVAU/VVAU node -- the axis this probe exists to isolate.")
ap.add_argument("--fold", choices=["pemh_simd1", "balanced", "pe1_simdmw"], required=True,
                 help="Folding point: pemh_simd1 (PE=MH,SIMD=1, degenerate high-PE), "
                      "pe1_simdmw (PE=1,SIMD=MW, degenerate high-SIMD), or "
                      "balanced (PE=SIMD=gcd(MH,MW) per node).")
args = ap.parse_args()

RES_TYPE = args.res_type
FOLD = args.fold

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
MODEL_NAME = "probe_lutmult_noact0_int8"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"{MODEL_NAME}_{RES_TYPE}_{FOLD}_{timestamp}")

FPGA_PART = "xczu7ev-ffvc1156-2-e"


def step_reapply_unique_names(model: ModelWrapper, cfg: DataflowBuildConfig):
    return model.transform(GiveUniqueNodeNames())


def step_print_noactivation_diagnostic(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Verification, not a transform (model unchanged) -- confirms every
    MVAU/VVAU node actually landed noActivation=0 (this probe's whole premise)
    right after step_specialize_layers has picked hls/rtl per node. See
    module docstring's point 3."""
    print("[step_print_noactivation_diagnostic] per-node backend/noActivation after specialize_layers:")
    for node in model.graph.node:
        if "MVAU" in node.op_type or "VVAU" in node.op_type:
            inst = getCustomOp(node)
            no_act = inst.get_nodeattr("noActivation")
            mh = inst.get_nodeattr("MH") if "MVAU" in node.op_type else inst.get_nodeattr("Channels")
            print(f"    {node.name} ({node.op_type}): noActivation={no_act}, MH/Channels={mh}"
                  + ("  <-- UNEXPECTED: this probe's sequential topology should force noActivation=0 everywhere"
                     if no_act == 1 else ""))
    return model


def step_force_res_type(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Force resType={RES_TYPE} on every MVAU/VVAU node; ram_style left at
    default (auto) -- parameterized generalization of finn_build_probe_
    s12_context_v2.py's step_force_dsp."""
    n = 0
    for node in model.graph.node:
        if "MVAU" in node.op_type or "VVAU" in node.op_type:
            getCustomOp(node).set_nodeattr("resType", RES_TYPE)
            n += 1
    print(f"[step_force_res_type] forced resType={RES_TYPE} on {n} MVAU/VVAU node(s), ram_style left at default (auto)")
    return model


def step_force_pe_mh_simd1(model: ModelWrapper, cfg: DataflowBuildConfig):
    """PE=MH, SIMD=1 -- degenerate high-PE extreme. Always divisibility-safe
    (PE=MH trivially divides MH, SIMD=1 trivially divides MW)."""
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
    """PE=1, SIMD=MW -- mirror-image degenerate high-SIMD extreme (the
    direction mvau_lut_correlation_report.txt's real data calls safer:
    SIMD negatively correlated with real_LUT, PE positively). Always
    divisibility-safe (PE=1 trivially divides MH, SIMD=MW trivially divides
    MW)."""
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
    """PE=SIMD=gcd(MH,MW), computed per node -- a genuinely intermediate,
    non-degenerate folding point (NOT PE=SIMD=1, which is just the OTHER
    degenerate corner: minimal parallelism entirely, not a balanced HIGH-
    parallelism split). gcd(MH,MW) is a real common divisor of both by
    construction, so this is always divisibility-safe; for this probe's own
    geometry (channels=32, internal_channels=8) it happens to land on 8 for
    every real layer (MH,MW in {(8,32),(8,72),(32,8)}, gcd=8 in all three
    cases) -- computed generically here, not hardcoded, so it still does the
    right thing if the exported network's geometry ever changes."""
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
    own docstring for the exact assertion this works around
    (convert_to_hw_layers.py's "Signed output requires actval < 0"). Kept
    here for safety/consistency even though this probe's sequential network
    has no QuantEltwiseAdd input_quant identity (the specific site that
    needed it there) -- harmless no-op if it finds nothing to fix."""
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
    "step_specialize_layers",
    step_reapply_unique_names,
    step_print_noactivation_diagnostic,
    step_force_res_type,
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
    print(f"resType: {RES_TYPE}")
    print(f"Fold   : {FOLD}")
    print(f"Model  : {MODEL_FILE}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Part   : {FPGA_PART}")
    print(f"Steps  : {[s if isinstance(s, str) else s.__name__ for s in probe_steps]}")
    print(flush=True)

    build.build_dataflow_cfg(MODEL_FILE, cfg_probe)

    print("Stitched IP / reports in:", OUTPUT_DIR)
    print("OUTPUT_DIR=", OUTPUT_DIR)

    print(f"\n[auto-ooc-synth] stitching finished -- starting OOC synthesis automatically", flush=True)
    run_ooc_synth(OUTPUT_DIR)
    print("Done (build + OOC synth).")

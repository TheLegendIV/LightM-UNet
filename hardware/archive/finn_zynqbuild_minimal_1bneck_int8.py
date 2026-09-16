"""FINN full ZynqBuild (single partition) for the minimal 1-bottleneck
derisking model -- quantEnet_minimal_1bneck_int8.onnx (see
finn_export_minimal_1bneck_int8.py).

Unlike finn_enet_ip_build.py (stitched IP + OOC synth only, no board/DMA),
this script runs FINN's REAL board-integration flow: board="ZCU104",
shell_flow_type=ShellFlowType.VIVADO_ZYNQ, generate_outputs includes BITFILE,
PYNQ_DRIVER, and DEPLOYMENT_PACKAGE. FINN's ZynqBuild handles PS/DDR/DMA
wiring, constraints, full Vivado synthesis + implementation, and bitstream
generation automatically for this single-partition model.

Custom tidy/streamline/convert_to_hw steps copied verbatim from
finn_enet_ip_build.py (not imported, so this script has no import-time side
effects tied to sys.argv) -- same rationale as that file's own docstring.

Board name is "ZCU104" (the real FINN board-file name for this chip), NOT the
"ZCU7EV" placeholder used in the never-run notebook reference
(finn_enet_deploy_xczu7ev.ipynb) -- ZCU7EV is the chip, ZCU104 is the board
FINN's board_repo actually ships a definition for. fpga_part is intentionally
NOT set here -- FINN derives it from the board's own board.json when board=
is given (matches the notebook's own reference pattern for cfg_bitstream).

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_zynqbuild_minimal_1bneck_int8.py
"""
import os
import sys
from datetime import datetime

# ── FINN source paths (source install, not pip) ──────────────────────────────
sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/brevitas/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/pyverilator")
sys.path.insert(0, "/home/thelegendiv/finn/deps/finn-experimental")

# ── Xilinx tool PATH (vitis_hls / vivado) ─────────────────────────────────────
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
    MoveScalarMulPastConvTranspose,
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
from finn.transformation.fpgadataflow.infer_pixel_padding_deconv import InferPixelPaddingDeconv
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig


# ── paths ────────────────────────────────────────────────────────────────────
ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
MODEL_NAME = "quantEnet_minimal_1bneck_int8"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"zynqbuild_{MODEL_NAME}_{timestamp}")

BOARD = "ZCU104"


# ── Custom step: tidy (copied verbatim from finn_enet_ip_build.py) ───────────
def step_reapply_unique_names(model: ModelWrapper, cfg: DataflowBuildConfig):
    # step_specialize_layers leaves every new HLS/RTL node's .name == "",
    # which crashes HLSSynthIP's set_top with an empty top-level function name.
    return model.transform(GiveUniqueNodeNames())


def step_enet_tidy(model: ModelWrapper, cfg: DataflowBuildConfig):
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
        MoveScalarMulPastConvTranspose(),
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


def step_enet_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
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

    if len(model.get_nodes_by_op_type("ConvTranspose")) > 0:
        model = model.transform(InferPixelPaddingDeconv())
        model = model.transform(AbsorbConsecutiveTransposes())

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataLayouts())
    model = model.transform(InferDataTypes())
    return model


def step_enet_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig):
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model = model.transform(InferDataLayouts())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())

    for n in model.graph.node:
        if n.op_type == "ConvTranspose":
            idt = model.get_tensor_datatype(n.input[0])
            wdt = model.get_tensor_datatype(n.input[1])
            if idt.is_integer() and wdt.is_integer():
                model.set_tensor_datatype(n.output[0], DataType["INT32"])
    model = model.transform(InferDataTypes())

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
        model = model.transform(trn())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())

    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())
    return model


# ── Build config ───────────────────────────────────────────────────────────
# Same custom preamble as finn_enet_ip_build.py, but the ENDING steps switch
# from OOC-synth-only to the real board flow (step_synthesize_bitfile,
# step_make_pynq_driver, step_deployment_package) -- these are no-ops inside
# FINN itself unless board=/generate_outputs request them, so it's safe/
# idiomatic to just list them (matches FINN's own default step ordering).
enet_zynq_steps = [
    "step_qonnx_to_finn",
    step_enet_tidy,
    step_enet_streamline,
    step_enet_convert_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    step_reapply_unique_names,
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_measure_rtlsim_performance",
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]

cfg_zynq = DataflowBuildConfig(
    output_dir          = OUTPUT_DIR,
    mvau_wwidth_max     = 80,
    target_fps          = None,   # fully unfolded (PE=SIMD=1) -- simplest/basic, per derisking goal
    synth_clk_period_ns = 10.0,   # 100 MHz, same conservative clock as rest of repo
    split_large_fifos   = True,   # auto-sized FIFOs can exceed Vivado FIFO IP's 32768-depth limit
    board               = BOARD,
    shell_flow_type     = build_cfg.ShellFlowType.VIVADO_ZYNQ,
    steps               = enet_zynq_steps,
    generate_outputs    = [
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ],
    save_intermediate_models = True,
)

# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Model : {MODEL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Board : {BOARD}")
    print(f"Steps : {[s if isinstance(s, str) else s.__name__ for s in enet_zynq_steps]}")
    print()

    build.build_dataflow_cfg(MODEL_FILE, cfg_zynq)

    import json

    report_dir = os.path.join(OUTPUT_DIR, "report")
    bitfile_dir = os.path.join(OUTPUT_DIR, "bitfile")
    deploy_dir = os.path.join(OUTPUT_DIR, "deploy")

    perf_json = os.path.join(report_dir, "estimate_network_performance.json")
    if os.path.exists(perf_json):
        print("\n" + "=" * 60)
        print("NETWORK PERFORMANCE ESTIMATES (analytical)")
        print("=" * 60)
        with open(perf_json) as f:
            for k, v in json.load(f).items():
                print(f"  {k}: {v}")

    rtlsim_json = os.path.join(report_dir, "rtlsim_performance.json")
    if os.path.exists(rtlsim_json):
        print("\n" + "=" * 60)
        print("RTLSIM PERFORMANCE (cycle-accurate, Verilator)")
        print("=" * 60)
        with open(rtlsim_json) as f:
            for k, v in json.load(f).items():
                print(f"  {k}: {v}")

    if os.path.exists(bitfile_dir):
        print(f"\nBitfile outputs in {bitfile_dir}:")
        for f in os.listdir(bitfile_dir):
            print(f"  - {f}")
    else:
        print(f"\n[WARN] No bitfile dir at {bitfile_dir}")

    if os.path.exists(deploy_dir):
        print(f"\nDeployment package in {deploy_dir}:")
        for f in os.listdir(deploy_dir):
            print(f"  - {f}")
    else:
        print(f"\n[WARN] No deploy dir at {deploy_dir}")

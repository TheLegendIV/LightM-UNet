"""FINN stitched-IP-only build (NO ZynqBuild/bitfile) for the minimal
"Approach 2" bilinear-upsampling decomposition probe (nearest-neighbor
upsample + depthwise conv) -- see finn_export_probe_upsample_nearest_
depthwise_int8.py. Goal: verify `InferUpsample`/`UpsampleNearestNeighbour`
+ depthwise `VVAU` lower correctly and reach a real stitched IP, as fast/
lightweight as possible (no PS/DDR/DMA, no full Vivado impl+bitstream) so it
can run IN PARALLEL with the S12 partition-0 real ZynqBuild without
competing for hours of Vivado time.

Smallest-footprint build settings: target_fps=None (PE=SIMD=1, fully
unfolded/minimal parallelism -- matches the model's own minimal channel
count already).

Custom tidy/streamline/convert_to_hw copied from finn_zynqbuild_minimal_
1bneck_int8.py's own verbatim-copied steps, with `to_hw.InferUpsample`
added to the convert_to_hw transform list (not present in that file since
ENet itself has no Upsample node -- only ConvTranspose).

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_build_probe_upsample_nearest_depthwise_int8.py
"""
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
    AbsorbTransposeIntoResize,
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


ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
MODEL_NAME = "quant_probe_upsample_nearest_depthwise_int8"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"probe_{MODEL_NAME}_{timestamp}")

# Real xczu7ev part directly -- no board= (no PS/DDR/DMA needed for a
# stitched-IP-only probe, keeps this fast/independent of the ZynqBuild).
FPGA_PART = "xczu7ev-ffvc1156-2-e"


def step_reapply_unique_names(model: ModelWrapper, cfg: DataflowBuildConfig):
    return model.transform(GiveUniqueNodeNames())


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

    # LowerConvsToMatMul (in streamline) locally wraps each lowered conv in
    # Transpose(NHWC->NCHW)/Transpose(NCHW->NHWC) pairs, restoring the
    # original NCHW convention in between -- which leaves the Resize
    # (Upsample) node sitting in "NCHW space" behind a MultiThreshold, so
    # InferUpsample's NHWC requirement silently fails to match and it's
    # skipped. First move the Transpose past the intervening MultiThreshold
    # (so it becomes directly adjacent to Resize), then absorb it into the
    # Resize node itself (reordering its scales to NHWC) so Resize becomes
    # genuinely NHWC-native, then cancel any leftover redundant Transposes.
    model = model.transform(AbsorbTransposeIntoMultiThreshold())
    model = model.transform(AbsorbTransposeIntoResize())
    model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())

    for trn in [
        to_hw.InferUpsample,
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


probe_steps = [
    "step_qonnx_to_finn",
    step_probe_tidy,
    step_probe_streamline,
    step_probe_convert_to_hw,
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
]

cfg_probe = DataflowBuildConfig(
    output_dir          = OUTPUT_DIR,
    mvau_wwidth_max     = 80,
    target_fps          = None,   # fully unfolded (PE=SIMD=1) -- smallest-footprint parallelism
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
    print(f"Model : {MODEL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Part  : {FPGA_PART}")
    print(f"Steps : {[s if isinstance(s, str) else s.__name__ for s in probe_steps]}")
    print(flush=True)

    build.build_dataflow_cfg(MODEL_FILE, cfg_probe)

    print("Done. Stitched IP / reports in:", OUTPUT_DIR)
    print("OUTPUT_DIR=", OUTPUT_DIR)

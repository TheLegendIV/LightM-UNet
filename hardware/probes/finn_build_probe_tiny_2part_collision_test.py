"""FINN stitched-IP-only build for TinyResidualProbe (see
finn_export_probe_tiny_2part_collision_test.py). Run this TWICE with
different --tag values (p0/p1) to produce two INDEPENDENT stitched-IP
projects from the SAME model -- each independent build's child HLS IPs
(DuplicateStreams_hls_0, AddStreams_hls_0, MVAU_hls_0/1/2) restart numbering
from 0, exactly reproducing the real 8-way build's cross-partition VLNV
collision class, cheaply and fast (tiny 4ch/8x8 model, stitched-IP only, no
OOC synth/bitstream).

Usage (inside the FINN container):
    python3 finn_build_probe_tiny_2part_collision_test.py p0
    python3 finn_build_probe_tiny_2part_collision_test.py p1
"""
import os
import sys

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
MODEL_NAME = "quant_probe_tiny_2part_collision_test"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")
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
    model = model.transform(GiveUniqueNodeNames())

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


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("p0", "p1"):
        print("Usage: finn_build_probe_tiny_2part_collision_test.py {p0|p1}")
        sys.exit(1)
    tag = sys.argv[1]

    output_dir = os.path.join(ENET_DIR, "finn_deployment_outputs", f"tiny_2part_collision_test_{tag}")

    cfg_probe = DataflowBuildConfig(
        output_dir          = output_dir,
        mvau_wwidth_max     = 80,
        target_fps          = None,   # fully unfolded (PE=SIMD=1)
        synth_clk_period_ns = 10.0,
        split_large_fifos   = True,
        fpga_part           = FPGA_PART,
        steps               = probe_steps,
        generate_outputs    = [
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
        ],
        save_intermediate_models = True,
    )

    print(f"Model : {MODEL_FILE}")
    print(f"Tag   : {tag}")
    print(f"Output: {output_dir}")
    print(f"Part  : {FPGA_PART}")
    print(flush=True)

    build.build_dataflow_cfg(MODEL_FILE, cfg_probe)

    print("Done. Stitched IP / reports in:", output_dir)
    print("OUTPUT_DIR=", output_dir)


if __name__ == "__main__":
    main()

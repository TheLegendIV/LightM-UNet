"""Runs the REAL, unmodified production multi-partition pipeline
(finn_partition_build_steps.py's step_create_dataflow_partition_multi /
step_build_all_partitions / step_combine_partitions / step_generate_
estimate_reports_multi / step_measure_rtlsim_performance_multi -- the
exact same functions every real *_8way_full.py script uses) against the
tiny 2-partition collision-test toy network, splitting it into 2 real
partitions via partition_id (mirroring finn_stage_partition.py's
assign_stage_partition_ids, just index-halved instead of stage-boundary
based).

Purpose: directly verify -- for THIS toy network, end-to-end, through
the real production code path -- that step_combine_partitions'
_rename_partition_verilog_sources fix actually avoids the RTL/HLS
module-name collision this session investigated, rather than relying
only on inference from a real 8-way build's log (see finn_gotchas.md,
2026-09-20 entries). No monkeypatch here: this deliberately exercises
the pipeline AS-IS.

Usage (inside the FINN container):
    python3 finn_build_probe_tiny_2part_real_pipeline_test.py
"""
import os
import sys
import dataclasses

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
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
from qonnx.util.basic import get_by_name

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
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType

# the REAL, unmodified production multi-partition step functions
from finn_partition_build_steps import (
    step_create_dataflow_partition_multi,
    step_build_all_partitions,
    step_combine_partitions,
    step_generate_estimate_reports_multi,
    step_measure_rtlsim_performance_multi,
)


ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
MODEL_NAME = "quant_probe_tiny_2part_collision_test"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")
FPGA_PART = "xczu7ev-ffvc1156-2-e"


def step_probe_tidy(model):
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


def _streamline_linear(model):
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


def _streamline_nonlinear(model):
    for trn in [
        MoveLinearPastFork(),
        MoveLinearPastEltwiseAdd(),
    ]:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


def step_probe_streamline(model):
    for _iter in range(4):
        model = _streamline_linear(model)
        model = _streamline_nonlinear(model)
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


def step_probe_convert_to_hw(model):
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


def _is_fpgadataflow_node(node):
    backend = get_by_name(node.attribute, "backend")
    return backend is not None and backend.s.decode("UTF-8") == "fpgadataflow"


def assign_2part_partition_ids(model):
    """Splits the toy network's HW nodes into 2 partitions by index
    (first half -> partition_id 0, second half -> partition_id 1),
    mirroring finn_stage_partition.py's assign_stage_partition_ids
    pattern but with a simple midpoint cut instead of stage boundaries."""
    model = model.transform(SortGraph())
    hw_node_indices = [
        idx for idx, node in enumerate(model.graph.node) if _is_fpgadataflow_node(node)
    ]
    assert len(hw_node_indices) >= 2, "Need at least 2 HW nodes to split into 2 partitions"
    mid = hw_node_indices[len(hw_node_indices) // 2]
    n0 = n1 = 0
    for idx, node in enumerate(model.graph.node):
        if not _is_fpgadataflow_node(node):
            continue
        inst = getCustomOp(node)
        if idx < mid:
            inst.set_nodeattr("partition_id", 0)
            n0 += 1
        else:
            inst.set_nodeattr("partition_id", 1)
            n1 += 1
    print(f"[assign_2part_partition_ids] partition 0: {n0} HW nodes, partition 1: {n1} HW nodes")
    return model


def main():
    output_dir = os.path.join(ENET_DIR, "finn_deployment_outputs", "tiny_2part_real_pipeline_test")
    os.makedirs(output_dir, exist_ok=True)

    cfg = DataflowBuildConfig(
        output_dir          = output_dir,
        mvau_wwidth_max     = 80,
        target_fps          = None,
        synth_clk_period_ns = 10.0,
        split_large_fifos   = True,
        fpga_part           = FPGA_PART,
        steps               = [],  # driven manually below, not via build_dataflow_cfg initially
        generate_outputs    = [
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        ],
        save_intermediate_models = True,
    )

    print(f"Model : {MODEL_FILE}")
    print(f"Output: {output_dir}")
    print(f"Part  : {FPGA_PART}", flush=True)

    model = ModelWrapper(MODEL_FILE)
    model = model.transform(ConvertQONNXtoFINN())
    model = step_probe_tidy(model)
    model = step_probe_streamline(model)
    model = step_probe_convert_to_hw(model)
    model = assign_2part_partition_ids(model)

    parent_model = step_create_dataflow_partition_multi(model, cfg)
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    print(f"Got {len(sdp_nodes)} partitions: {[n.name for n in sdp_nodes]}")
    assert len(sdp_nodes) == 2, f"expected 2 partitions, got {len(sdp_nodes)}"

    # real, unmodified production per-partition build (sequential for easy debugging)
    parent_model = step_build_all_partitions(parent_model, cfg, parallel=False)

    parent_ckpt = os.path.join(output_dir, "intermediate_models", "dataflow_parent_built.onnx")
    os.makedirs(os.path.dirname(parent_ckpt), exist_ok=True)
    parent_model.save(parent_ckpt)

    cfg2 = dataclasses.replace(
        cfg,
        steps=[
            step_combine_partitions,
            step_generate_estimate_reports_multi,
            step_measure_rtlsim_performance_multi,
        ],
    )
    print("Proceeding to step_combine_partitions -> estimate reports -> rtlsim...", flush=True)
    build.build_dataflow_cfg(parent_ckpt, cfg2)

    print("Done. Reports in:", os.path.join(output_dir, "report"))
    print("OUTPUT_DIR=", output_dir)


if __name__ == "__main__":
    main()

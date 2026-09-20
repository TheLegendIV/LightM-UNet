"""Validates (or refutes) the *actual* per-partition naming-prefix mechanism
used by the real 8-way builds (`_build_one_partition_with_folding_and_dsp` /
`_build_one_partition` in finn_partition_build_steps.py and the
`finn_ooc_*_8way_full.py` scripts), NOT the Tcl VLNV-rename fix documented
separately in finn_gotchas.md's 2026-09-20 entry.

Those scripts do, in order:
    step_specialize_layers
    model.transform(GiveUniqueNodeNames(prefix))      # e.g. prefix="p0_"
    model.transform(GiveReadableTensorNames())
    step_target_fps_parallelization
    step_apply_folding_config(model, cfg)             # cfg.folding_config_file is ALWAYS set in the real
                                                       # 8-way builds (per-partition HAWQ folding config)
    step_minimize_bit_width
    step_hw_codegen
    step_hw_ipgen
    step_set_fifo_depths
    model.transform(CreateStitchedIP(part, clk, ip_name=prefix.rstrip("_"), False))

BUT finn's own step_apply_folding_config (build_dataflow_steps.py) does:
    if cfg.folding_config_file is not None:
        model = model.transform(GiveUniqueNodeNames())   # <-- bare, NO prefix!
        model = model.transform(ApplyConfig(cfg.folding_config_file))

i.e. whenever a folding_config_file is used (always, in the real per-partition
builds), the earlier prefix gets silently wiped right before step_hw_codegen/
step_hw_ipgen -- meaning the HLS-generated child IP names (which derive from
node.name at ipgen time) come out UNPREFIXED regardless of the prefix passed
to CreateStitchedIP's ip_name (which only renames the *top-level* stitched
BD's own packaged IP, not its child HLS IP catalog entries).

This script builds the tiny 2-part collision probe through that exact
sequence twice (tag=p0/p1), in two modes:
    mode=nofix : reproduces the production step order exactly (expected BUG:
                 child HLS IP names collide across p0/p1 despite prefix)
    mode=fix   : re-applies GiveUniqueNodeNames(prefix) AGAIN right after
                 step_apply_folding_config, before step_minimize_bit_width /
                 step_hw_codegen / step_hw_ipgen (expected FIX: child HLS IP
                 names come out uniquely prefixed per partition)

Usage (inside the FINN container):
    python3 finn_build_probe_tiny_2part_prefix_test.py p0 nofix
    python3 finn_build_probe_tiny_2part_prefix_test.py p1 nofix
    python3 finn_build_probe_tiny_2part_prefix_test.py p0 fix
    python3 finn_build_probe_tiny_2part_prefix_test.py p1 fix

After each pair of runs, inspect e.g.:
    ls finn_deployment_outputs/tiny_2part_prefix_test_nofix_p0/stitched_ip/ip/*/component.xml
    ls finn_deployment_outputs/tiny_2part_prefix_test_nofix_p1/stitched_ip/ip/*/component.xml
to compare child HLS IP catalog directory names between p0 and p1.
"""
import json
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
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP

from finn.builder.build_dataflow_steps import (
    step_qonnx_to_finn,
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
)
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig


ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
MODEL_NAME = "quant_probe_tiny_2part_collision_test"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")
FPGA_PART = "xczu7ev-ffvc1156-2-e"


def probe_tidy(model):
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


def probe_streamline(model):
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


def probe_convert_to_hw(model):
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


def build_one_partition(model, cfg, prefix, apply_fix):
    """Mirrors finn_partition_build_steps.py's _build_one_partition exactly,
    with an optional extra GiveUniqueNodeNames(prefix) re-application after
    step_apply_folding_config to test the candidate fix."""
    model = step_specialize_layers(model, cfg)
    model = model.transform(GiveUniqueNodeNames(prefix))
    model = model.transform(GiveReadableTensorNames())
    print(f"  [after initial prefix rename] node names: {[n.name for n in model.graph.node]}")

    model = step_target_fps_parallelization(model, cfg)
    model = step_apply_folding_config(model, cfg)
    print(f"  [after step_apply_folding_config] node names: {[n.name for n in model.graph.node]}")

    if apply_fix:
        model = model.transform(GiveUniqueNodeNames(prefix))
        print(f"  [after FIX re-apply] node names: {[n.name for n in model.graph.node]}")

    model = step_minimize_bit_width(model, cfg)
    model = step_hw_codegen(model, cfg)
    model = step_hw_ipgen(model, cfg)
    print(f"  [after step_hw_ipgen] node names: {[n.name for n in model.graph.node]}")

    model = step_set_fifo_depths(model, cfg)
    print(f"  [after step_set_fifo_depths] node names: {[n.name for n in model.graph.node]}")

    if apply_fix:
        # step_set_fifo_depths (InsertAndSetFIFODepths) inserts NEW
        # StreamingFIFO_rtl_*/StreamingDataWidthConverter_rtl_* nodes that
        # never got the prefix -- re-apply once more so these newly-created
        # nodes are also uniquely named before HLSSynthIP/CreateStitchedIP
        # bakes their names into the packaged IP-XACT catalog.
        model = model.transform(GiveUniqueNodeNames(prefix))
        model = step_hw_ipgen(model, cfg)
        print(f"  [after 2nd FIX re-apply + re-ipgen] node names: {[n.name for n in model.graph.node]}")

    model = model.transform(CreateStitchedIP(cfg._resolve_fpga_part(), cfg.synth_clk_period_ns, prefix.rstrip("_"), False))
    return model


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("p0", "p1") or sys.argv[2] not in ("nofix", "fix"):
        print("Usage: finn_build_probe_tiny_2part_prefix_test.py {p0|p1} {nofix|fix}")
        sys.exit(1)
    tag = sys.argv[1]
    mode = sys.argv[2]
    prefix = f"{tag}_"
    apply_fix = mode == "fix"

    output_dir = os.path.join(ENET_DIR, "finn_deployment_outputs", f"tiny_2part_prefix_test_{mode}_{tag}")
    os.makedirs(output_dir, exist_ok=True)

    # trivial folding config file -- just needs to be non-None to trigger
    # step_apply_folding_config's internal bare GiveUniqueNodeNames() call,
    # matching the real 8-way builds (which always pass a real per-partition
    # HAWQ folding config here). An empty "Defaults" is a no-op otherwise.
    folding_config_file = os.path.join(output_dir, "trivial_folding_config.json")
    with open(folding_config_file, "w") as f:
        json.dump({"Defaults": {}}, f)

    cfg = DataflowBuildConfig(
        output_dir          = output_dir,
        mvau_wwidth_max     = 80,
        target_fps          = None,
        synth_clk_period_ns = 10.0,
        split_large_fifos   = True,
        fpga_part           = FPGA_PART,
        folding_config_file = folding_config_file,
        steps               = [],  # unused -- we drive steps manually below
        generate_outputs    = [
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
        ],
        save_intermediate_models = True,
    )

    print(f"Model : {MODEL_FILE}")
    print(f"Tag   : {tag}  mode: {mode}  prefix: {prefix!r}  apply_fix: {apply_fix}")
    print(f"Output: {output_dir}")
    print(flush=True)

    model = ModelWrapper(MODEL_FILE)
    model = step_qonnx_to_finn(model, cfg)
    model = model.transform(InferShapes())
    model = probe_tidy(model)
    model = probe_streamline(model)
    model = probe_convert_to_hw(model)

    model = model.transform(CreateDataflowPartition(partition_model_dir=output_dir + "/intermediate_models/supported_op_partitions"))
    sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 1, f"expected 1 partition for this tiny probe, got {len(sdp_nodes)}"
    from qonnx.custom_op.registry import getCustomOp
    kernel_model_fn = getCustomOp(sdp_nodes[0]).get_nodeattr("model")
    kernel_model = ModelWrapper(kernel_model_fn)

    kernel_model = build_one_partition(kernel_model, cfg, prefix, apply_fix)
    kernel_model.save(os.path.join(output_dir, "final_partition_model.onnx"))

    print("Done.", output_dir)
    print("OUTPUT_DIR=", output_dir)


if __name__ == "__main__":
    main()

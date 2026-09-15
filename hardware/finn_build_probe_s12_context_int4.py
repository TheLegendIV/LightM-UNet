"""FINN build (through real stitched IP) for the S12 dense-vs-separable
context-block probe pair -- see finn_export_probe_s12_context_dense_int4.py /
_separable_int4.py for the matched-pair ONNX exports this consumes, and the
plan this implements: C:\\Users\\win32\\.claude\\plans\\i-am-trying-to-jazzy-llama.md.

Byte-for-byte clone of finn_build_probe_upsample_nearest_depthwise_int8.py's
step-list skeleton (same tidy/streamline/convert_to_hw pattern -- this probe
has ordinary full-rank Conv2d nodes only, no Upsample/depthwise, so
`to_hw.InferUpsample` is dropped and no depthwise-specific handling is
needed), parameterized by --variant so the SAME script builds either half of
the pair with identical settings (only the input .onnx differs).

target_fps=None matches this repo's own established convention for "PE=SIMD=1
unless overridden by folding_config_file" (see e.g.
finn_zynqbuild_12_dense_relu_alpha025_partition0.py's own comment) -- no
folding_config_file is passed here, so every node gets FINN's baseline
node-default folding, which for this probe's mvau_wwidth_max=80-bounded plain
convs is PE=SIMD=1, matching the raw analytical estimate's own PE=SIMD=1
assumption exactly (see finn_export_probe_s12_context_common.py's
raw_estimate). If a real build's own auto_folding_config.json (produced by
step_target_fps_parallelization / step_apply_folding_config, see
report/estimate_layer_resources.json alongside it) turns out to show
different PE/SIMD than 1/1, use that ACTUAL folding when comparing against
the analytical estimate rather than assuming it held.

Stops at step_create_stitched_ip / step_measure_rtlsim_performance, same as
the existing probe -- OOC synthesis is a SEPARATE step
(finn_ooc_probe_s12_context_synth.py), applying FINN's own unmodified
SynthOutOfContext directly to the stitched-IP model, mirroring this repo's
own proven per-partition-synth pattern (see e.g.
finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_partition_synth.py)
rather than the documented-buggy combined-design `step_out_of_context_
synthesis_multi` path -- irrelevant here anyway since this probe is a single,
non-partitioned model, but keeping the SAME two-script split this repo
already uses everywhere avoids re-deriving a new pattern.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_build_probe_s12_context_int4.py dense
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_build_probe_s12_context_int4.py separable

Then, for OOC synthesis on either build's stitched IP:
    docker exec -e HOME=/tmp/home_dir <container> python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_ooc_probe_s12_context_synth.py <OUTPUT_DIR>
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


VARIANT = sys.argv[1] if len(sys.argv) >= 2 else "dense"
assert VARIANT in ("dense", "separable"), f"--variant must be 'dense' or 'separable', got {VARIANT!r}"

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
MODEL_NAME = f"probe_s12_context_{VARIANT}_int4"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"{MODEL_NAME}_{timestamp}")

# Real xczu7ev part directly -- no board= (no PS/DDR/DMA needed for a
# stitched-IP-only probe), same convention as the existing upsample/depthwise
# probe and the real S12 per-partition OOC builds.
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
    target_fps          = None,   # PE=SIMD=1 unless overridden -- matches the analytical estimate's own assumption
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
    print(f"Model : {MODEL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Part  : {FPGA_PART}")
    print(f"Steps : {[s if isinstance(s, str) else s.__name__ for s in probe_steps]}")
    print(flush=True)

    build.build_dataflow_cfg(MODEL_FILE, cfg_probe)

    print("Done. Stitched IP / reports in:", OUTPUT_DIR)
    print("OUTPUT_DIR=", OUTPUT_DIR)
    print(f"\nNext: OOC synthesis via "
          f"finn_ooc_probe_s12_context_synth.py {OUTPUT_DIR}")

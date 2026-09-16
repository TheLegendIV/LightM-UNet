"""Variant of finn_enet_build.step_enet_convert_to_hw that forces every
MVAU/VVAU node to noActivation=1 (a prerequisite for RTL specialization --
see finn/transformation/fpgadataflow/specialize_layers.py's
_mvu_rtl_possible/_vvu_rtl_possible: noActivation==0 unconditionally
disqualifies RTL, regardless of datatype/signedness).

Root cause (confirmed via direct FINN source read, 2026-09-16):
InferBinaryMatrixVectorActivation/InferQuantizedMatrixVectorActivation/
InferVectorVectorActivation each check whether the MatMul's immediate
consumer is a MultiThreshold node -- if so, they FUSE it directly into the
new MVAU/VVAU (noActivation=0, threshold table embedded); only when no such
MultiThreshold immediately follows do they emit a standalone node
(noActivation=1). FINN's own DataflowBuildConfig has a built-in
`standalone_thresholds` flag that does exactly this for the STANDARD
step_convert_to_hw (prepends InferThresholdingLayer() so it converts every
MultiThreshold to a standalone Thresholding node before the MVAU/VVAU
transforms ever run, so they never find a fusable MultiThreshold) -- but
this repo's builds use a CUSTOM step_enet_convert_to_hw
(finn_enet_build.py), which the cfg flag has no effect on. This file
replicates the same trick manually, but INSIDE the main HW-conversion loop
(right before InferBinaryMatrixVectorActivation), not as a separate pass
before the whole loop -- doing it before InferUpsample/InferAddStreamsLayer/
etc. broke NHWC layout propagation onto Resize's input for up4/up5's
upsample branch (InferDataLayouts doesn't know how to propagate NHWC
through a HW "Thresholding" node the way it does for plain "MultiThreshold",
so InferUpsample silently skipped both Resize nodes -- confirmed
2026-09-16). InferThresholdingLayer() converts ANY MultiThreshold it finds
unconditionally (does not inspect its producer), so running it once, right
before the MVAU/VVAU-fusion transforms, is sufficient -- the loop's later,
still-present InferThresholdingLayer() call becomes a no-op (no
MultiThreshold nodes left).

Everything else is a byte-for-byte copy of finn_enet_build.step_enet_convert_to_hw.
"""
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from finn_enet_build import (  # noqa: E402
    DataType,
    to_hw,
    InferShapes,
    InferDataLayouts,
    InferDataTypes,
    GiveUniqueNodeNames,
    GiveReadableTensorNames,
    SortGraph,
    DoubleToSingleFloat,
    RoundAndClipThresholds,
    AbsorbTransposeIntoMultiThreshold,
    AbsorbTransposeIntoResize,
    AbsorbConsecutiveTransposes,
    MakeMaxPoolNHWC,
    MoveTransposePastJoinConcat,
    RemoveCNVtoFCFlatten,
    RemoveUnusedTensors,
    ModelWrapper,
    DataflowBuildConfig,
)


def _force_signed_weight_datatypes(model):
    """Promote unsigned-integer MatMul weight tensors (e.g. UINT3) to the
    smallest signed datatype that represents the same non-negative values
    (UINT3 -> INT4, one more bit, values unchanged) -- a metadata-only
    retag, not a re-encoding, since e.g. UINT3's 0..7 range is a subset of
    INT4's -8..7. Required because _mvu_rtl_possible (specialize_layers.py)
    unconditionally disqualifies RTL when wdt.signed() is False, regardless
    of noActivation. Skips promotion if it would push the weight bitwidth
    above 8 (FINN's own RTL weight-bitwidth eligibility limit), since that
    would disqualify RTL anyway. Only affects MatMul (MVAU/VVAU-bound);
    VVAU stays HLS-only regardless on this (non-Versal) target part."""
    for n in model.graph.node:
        if n.op_type != "MatMul":
            continue
        w_name = n.input[1]
        wdt = model.get_tensor_datatype(w_name)
        if wdt.is_integer() and not wdt.signed():
            new_bits = wdt.bitwidth() + 1
            if new_bits <= 8:
                model.set_tensor_datatype(w_name, DataType[f"INT{new_bits}"])
    return model


def step_enet_convert_to_hw_rtl_mvau(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Same as finn_enet_build.step_enet_convert_to_hw, but forces
    noActivation=1 on every MVAU/VVAU node by converting all MultiThreshold
    nodes to standalone Thresholding HW nodes right before the
    Binary/Quantized-MatMul/VVAU conversions get a chance to fuse them
    (NOT any earlier -- InferUpsample/InferDataLayouts rely on the
    standard "producer is MultiThreshold" layout-propagation heuristic;
    converting MultiThreshold to a HW Thresholding node earlier than this
    lost the NHWC tag on up4/up5's Resize input, breaking InferUpsample --
    confirmed 2026-09-16), and forces signed weight datatypes wherever
    numerically safe."""
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model = model.transform(InferDataLayouts())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())
    model = _force_signed_weight_datatypes(model)

    for n in model.graph.node:
        if n.op_type == "ConvTranspose":
            idt = model.get_tensor_datatype(n.input[0])
            wdt = model.get_tensor_datatype(n.input[1])
            if idt.is_integer() and wdt.is_integer():
                model.set_tensor_datatype(n.output[0], DataType["INT32"])
    model = model.transform(InferDataTypes())

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
        to_hw.InferThresholdingLayer,  # forced standalone HERE, before MVAU/VVAU fusion below
        to_hw.InferBinaryMatrixVectorActivation,
        to_hw.InferQuantizedMatrixVectorActivation,
        to_hw.InferVectorVectorActivation,
        to_hw.InferThresholdingLayer,  # no-op now, no MultiThreshold left
        AbsorbConsecutiveTransposes,
        to_hw.InferConvInpGen,
        to_hw.InferDuplicateStreamsLayer,
    ]:
        model = model.transform(trn())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())

    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(MoveTransposePastJoinConcat())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    for trn in [to_hw.InferStreamingMaxPool, to_hw.InferConcatLayer, AbsorbConsecutiveTransposes]:
        model = model.transform(trn())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())

    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())
    return model

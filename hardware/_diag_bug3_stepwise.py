"""Diagnostic (not part of the build pipeline): replicate step_enet_convert_to_hw's
transform list one sub-transform at a time on the pre-convert checkpoint, checking
after each for newly-orphaned (no-producer, non-initializer, non-graph-input) tensors,
to pinpoint exactly which to_hw transform introduces the dangling sicCfu/YWqgvl-style
tensors (Bug #3, see memories/repo/finn_12_dense_relu_alpha025_perlayer.md)."""
import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.general import GiveUniqueNodeNames, SortGraph
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes

PRE = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_preamble_20260908_223941/intermediate_models/_fixup_degenerate_signed_bias.onnx"


def orphans(model):
    graph = model.graph
    produced = set()
    for n in graph.node:
        produced.update(n.output)
    initializers = {t.name for t in graph.initializer}
    graph_inputs = {t.name for t in graph.input}
    bad = []
    for n in graph.node:
        for inp in n.input:
            if inp and inp not in produced and inp not in initializers and inp not in graph_inputs:
                bad.append((inp, n.op_type, n.name))
    return bad


model = ModelWrapper(PRE)
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

print("Orphans before any to_hw transform:", orphans(model))

steps = [
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
]

for trn_cls in steps:
    if trn_cls is None:
        continue
    before = set(orphans(model))
    model = model.transform(trn_cls())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataTypes())
    after = set(orphans(model))
    new = after - before
    print(f"After {trn_cls.__name__}: total_orphans={len(after)} NEW={sorted(new)}")

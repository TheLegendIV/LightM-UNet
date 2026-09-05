"""Real dataflow-graph predecessor tracing -- answers "which conv/pool
layer's OUTPUT actually feeds into this conv/pool layer's OWN INPUT", via
torch.fx symbolic tracing of a PLAIN FP32 model (never a Brevitas-quantized
one -- see compute_predecessor_map's own docstring for why).

WHY THIS EXISTS: compression/hawq/layer_sensitivity.py measures a layer's
sensitivity from a forward hook on that SAME layer's own module, i.e. on its
OWN OUTPUT tensor. But finn_cost_model.py's per-layer cost formula (see
BRAM_swu's `ceil(C*A/36)` line-buffer-sizing term, a real citation from
finn_cost_formulae.md) uses that same layer's own `act_bits` as its INPUT
stream's bit-width -- the tensor arriving at its own SWU, produced by
whatever sits immediately upstream. For a plain serial chain these would be
adjacent, near-enough tensors, but they are never the SAME tensor, and at a
residual join or a downsampling/upsampling block's own dual-path structure
they can be several hops apart. compression/hawq/joint_bits_folding_
ilp_perlayer.py's own module docstring documents this mismatch as a known,
pre-existing limitation (inherited from block_sensitivity.py's identical
per-block convention, just newly VISIBLE at per-layer resolution) -- this
file is the fix: a real predecessor map, so a layer's act-sensitivity
COEFFICIENT in the ILP's objective can be drawn from whichever layer(s)
ACTUALLY produced its input, not from its own (physically different)
output.

WHY TRACE THE PLAIN FP32 MODEL, NOT THE BREVITAS-QUANTIZED ONE: dataflow
TOPOLOGY (which module's output feeds which module's input) is identical
between an FP32 architecture and its Brevitas-quantized mirror -- this is
already an established, verified property of this codebase (QuantENet.py's
own topology-parity self-test against ENet.py; this session's own exact
key-set match between block_utils.block_weight_targets on a plain ENet and
both layer_sensitivity.py's report and LayerQuantENet's own per-layer
naming). Tracing the plain FP32 model sidesteps two real risks a Brevitas
model would introduce for NO benefit: (1) Brevitas modules can return a
QuantTensor rather than a plain torch.Tensor, and this codebase's own
forward() methods conditionally unwrap it via `hasattr(x, "value")` --
tracing that against an FX Proxy is untested and not needed here; (2) FX's
default symbolic tracer treats every nn.Module as either fully traced
through or a single opaque "call_module" leaf depending on its type
registration -- a plain nn.Conv2d/nn.BatchNorm2d/nn.ReLU model traces
cleanly and predictably, which is all the topology this file needs.

ALGORITHM: torch.fx.symbolic_trace(model) gives a real graph of `Node`s
(call_module/call_function/call_method), each with `.all_input_nodes` --
the REAL producer node(s) for every one of its inputs, regardless of how
many non-tracked ops (BatchNorm, activation, Dropout, elementwise add,
torch.cat) sit in between. For every Conv2d/ConvTranspose2d/MaxPool2d
`call_module` node, walk `.all_input_nodes` backward (skipping over
anything that isn't itself one of those three types) until hitting the
nearest tracked ancestor(s) -- plural, because a residual join's Add node
has TWO input branches, and both are real, independent contributors to
whatever wire comes out the other side (e.g. a DownsamplingBottleneck's
`main` branch, literally `MaxPool2d(x)`, and its `expand` branch each
reach the residual join independently -- the layer immediately downstream
of that join has BOTH as real predecessors, not just one).
"""
from __future__ import annotations

import torch.fx as fx
from torch import nn

TRACKED_MODULE_TYPES = (nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d)


class _LeafAtTypes(fx.Tracer):
    """Plain fx.Tracer, except any module whose type is in `leaf_types` is
    treated as opaque (traced as ONE call_module node, never descended into)
    regardless of FX's own default is_leaf_module rule. Needed for any
    submodule whose forward() branches on a tensor's runtime SHAPE (e.g.
    ENet.py's DownsamplingBottleneck pads/truncates its pooled branch via
    `if main.shape[1] < self.out_channels`) -- plain symbolic_trace's static
    Proxy can't evaluate that as a bool and raises TraceError. Stopping at
    the containing module's own boundary sidesteps the conditional entirely
    (real channel counts are fixed per built model instance -- the branch
    always resolves the same way for a given architecture regardless of
    input VALUES, symbolic tracing just can't see that statically) at the
    real cost of losing predecessor visibility for whatever lives strictly
    INSIDE that module -- see compute_predecessor_map's own docstring for
    how callers should treat that gap."""

    def __init__(self, leaf_types: tuple[type, ...]):
        super().__init__()
        self.leaf_types = leaf_types

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, self.leaf_types):
            return True
        return super().is_leaf_module(m, module_qualified_name)


def compute_predecessor_map(model: nn.Module, leaf_module_types: tuple[type, ...] = ()) -> dict[str, list[str]]:
    """Returns {layer_name: [predecessor_layer_names]} for every Conv2d/
    ConvTranspose2d/MaxPool2d module in `model` (`layer_name` is the exact
    dotted module path -- FX's own call_module `target` string, which is
    `named_modules()`'s own qualified name, so these keys already match
    block_utils.block_weight_targets's/layer_sensitivity.py's/
    finn_block_costs.dump_block_layer_geometry's own naming with no
    translation needed).

    A layer with NO real predecessor (the network's very first tracked
    module -- its input is the raw network input, never itself measured by
    anything in this pipeline) maps to an empty list; callers must decide
    their own fallback (see joint_bits_folding_ilp_perlayer.py's own choice:
    fall back to the layer's OWN sensitivity there, since no better signal
    exists for "how sensitive is the raw input image to quantization").

    A layer with MULTIPLE real predecessors (a residual-join point) gets all
    of them, in the order discovered -- deduplicated, order-preserving.

    leaf_module_types: passed straight to _LeafAtTypes -- any module of one
    of these types is traced as ONE opaque node instead of being descended
    into (see that class's own docstring for why ENet.py's
    DownsamplingBottleneck specifically needs this). Any Conv2d/
    ConvTranspose2d/MaxPool2d strictly INSIDE such a module is INVISIBLE to
    this function entirely -- it will not appear as a key in the returned
    dict at all (not even with an empty predecessor list), since FX never
    visits it as a separate node. Callers iterating a known, larger set of
    expected layer names (e.g. every real geometry from dump_block_layer_
    geometry) must handle keys missing from this map explicitly -- do not
    assume `name in predecessor_map` implies "no predecessor", it may
    instead mean "inside an opaque leaf, no information available"."""
    tracer = _LeafAtTypes(leaf_module_types) if leaf_module_types else fx.Tracer()
    graph = tracer.trace(model)
    traced = fx.GraphModule(model, graph)
    named_modules = dict(traced.named_modules())

    tracked_name_of_node: dict[fx.Node, str] = {
        node: node.target for node in traced.graph.nodes
        if node.op == "call_module" and isinstance(named_modules.get(node.target), TRACKED_MODULE_TYPES)
    }

    memo: dict[fx.Node, list[str]] = {}

    def nearest_tracked_ancestors(node: fx.Node) -> list[str]:
        if node in memo:
            return memo[node]
        memo[node] = []  # cycle guard -- FX graphs are DAGs, stays defensive/harmless either way
        found: list[str] = []
        for inp in node.all_input_nodes:
            if inp in tracked_name_of_node:
                found.append(tracked_name_of_node[inp])
            else:
                found.extend(nearest_tracked_ancestors(inp))
        seen: set[str] = set()
        result = [n for n in found if not (n in seen or seen.add(n))]
        memo[node] = result
        return result

    return {name: nearest_tracked_ancestors(node) for node, name in tracked_name_of_node.items()}

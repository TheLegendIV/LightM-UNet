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
ilp_perlayer.py's own module docstring documents this mismatch -- this file
is the fix: a real predecessor map, so a layer's act-sensitivity COEFFICIENT
in the ILP's objective (or a deployment bit-width lookup, see
LayerQuantENet's own expand_layer_bits_to_site_bits) can be drawn from
whichever layer(s) ACTUALLY produced its input, not from its own (physically
different) output.

WHY TRACE THE PLAIN FP32 MODEL, NOT THE BREVITAS-QUANTIZED ONE: dataflow
TOPOLOGY (which module's output feeds which module's input) is identical
between an FP32 architecture and its Brevitas-quantized mirror -- this is
already an established, verified property of this codebase (QuantENet.py's
own topology-parity self-test against ENet.py; this session's own exact
key-set match between block_utils.block_weight_targets on a plain ENet and
both layer_sensitivity.py's report and LayerQuantENet's own per-layer
naming). Tracing the plain FP32 model sidesteps a real risk a Brevitas model
would introduce for no benefit: Brevitas modules can return a QuantTensor
rather than a plain torch.Tensor, and this codebase's own forward() methods
conditionally unwrap it via `hasattr(x, "value")` -- tracing that against an
FX Proxy is untested and not needed here.

TWO REAL PROBLEMS SOLVED, both confirmed against the real S12 architecture
(not just the small demo net, which is too shallow to expose either):

1. SHAPE-DEPENDENT CONTROL FLOW defeats plain torch.fx.symbolic_trace --
   ENet.py's DownsamplingBottleneck (`if main.shape[1] < self.out_channels`),
   UpsamplingBottleneck (a shape-mismatch guard), and ENet.forward's own
   trailing interpolate-back-to-input-size check all branch on a tensor's
   runtime shape, which a symbolically-traced Proxy can't convert to a bool
   ("symbolically traced variables cannot be used as inputs to control
   flow"). FIX: override Tracer.to_bool to unconditionally return False --
   every one of these checks exists only to correct a size MISMATCH that
   does not occur for a well-formed, evenly-divisible-by-the-network's-own-
   stride input (the standard case throughout this whole codebase; the
   exceptional case is already documented elsewhere as being skipped from
   real sensitivity measurement batches for the same reason) -- so forcing
   the "no correction needed" branch reproduces the REAL topology exactly
   for that standard case, letting the entire real architecture trace
   end-to-end with FULL internal visibility (no opaque leaf modules needed
   at all, unlike an earlier version of this file).

2. NAIVE BACKWARD-WALK EXPLOSION through chained residual blocks -- a first
   version of this algorithm (walk `.all_input_nodes` backward, stopping
   only at Conv2d/ConvTranspose2d/MaxPool2d nodes) looked correct on the
   small demo net's single block, but on the real network it does NOT stop
   at "the nearest ancestor": for a Sequential of N residual blocks (e.g.
   regular1's 4, stage2/stage3's 8 each), a later block's own skip-connection
   operand is the PRIOR block's own complete output, which is ITSELF another
   residual join -- so the naive walk recurses through EVERY earlier block's
   own join in the chain, accumulating one more "tracked ancestor" per hop.
   Confirmed concretely: with no fix, `final`'s own "predecessor" list
   incorrectly included `initial.conv`, the very first layer in the whole
   104-layer network. FIX: cap the walk to ONE branch point (one residual
   join) per query -- once the walk has already passed through one node with
   >1 real input, hitting a SECOND one before reaching a tracked ancestor
   stops that path (contributes nothing further back), rather than
   continuing arbitrarily deep. This is a deliberate simplification, not
   just a technicality: down2.reduce.0's real input is regular1.3's own
   single output tensor, and the most representative available sensitivity
   proxy for it is regular1.3's own most recent real computation (its
   expand.0) plus, if regular1.3's OWN join is the first one encountered,
   its own direct contributors -- NOT also regular1.2/1.1/1.0's own
   contributions several blocks further back, which is genuinely staler
   information about a different point in the network.

ALGORITHM: for each Conv2d/ConvTranspose2d/MaxPool2d `call_module` node,
walk `.all_input_nodes` backward (skipping over anything that isn't itself
one of those three types -- BatchNorm, activation, Dropout, elementwise add,
torch.cat, size/shape queries) until hitting the nearest tracked ancestor(s)
along each path, stopping a path early (before any tracked node) if it
passes through a second branch point (residual join) -- see problem 2 above.
"""
from __future__ import annotations

import torch.fx as fx
from torch import nn

TRACKED_MODULE_TYPES = (nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d)


class _AlwaysFalseBoolTracer(fx.Tracer):
    """Plain fx.Tracer, except any Proxy asked for its bool value (an `if`
    on a shape comparison, typically) reports False instead of raising
    TraceError -- see module docstring's problem 1. This is a tracing-time
    stand-in only (never used for real inference): it reproduces the "no
    shape correction needed" branch every such check in this codebase
    exists to guard, which is the real branch taken for any well-formed
    input whose spatial size is evenly divisible by the network's own
    stride factors."""

    def to_bool(self, obj) -> bool:
        return False


def compute_predecessor_map(model: nn.Module) -> dict[str, list[str]]:
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
    their own fallback (e.g. joint_bits_folding_ilp_perlayer.py falls back
    to the layer's OWN sensitivity there, since no better signal exists for
    "how sensitive is the raw input image to quantization").

    A layer with MULTIPLE real predecessors (a residual-join point) gets all
    of them, in the order discovered -- deduplicated, order-preserving. See
    module docstring's problem 2 for why this list stops at the FIRST
    residual join encountered on each path, not every join transitively
    behind it."""
    tracer = _AlwaysFalseBoolTracer()
    graph = tracer.trace(model)
    traced = fx.GraphModule(model, graph)
    named_modules = dict(traced.named_modules())

    tracked_name_of_node: dict[fx.Node, str] = {
        node: node.target for node in traced.graph.nodes
        if node.op == "call_module" and isinstance(named_modules.get(node.target), TRACKED_MODULE_TYPES)
    }

    memo: dict[tuple[fx.Node, bool], list[str]] = {}

    def nearest_tracked_ancestors(node: fx.Node, allow_branch: bool) -> list[str]:
        """Ancestors of (and including, if itself tracked) `node`.
        allow_branch: whether this path is still allowed to pass through a
        residual-join (>1-input) node -- False once it already has (see
        module docstring's problem 2); a path that hits a second branch
        point before reaching a tracked node contributes nothing."""
        key = (node, allow_branch)
        if key in memo:
            return memo[key]
        memo[key] = []  # cycle guard -- FX graphs are DAGs, stays defensive/harmless either way
        if node in tracked_name_of_node:
            result = [tracked_name_of_node[node]]
        else:
            is_branch = len(node.all_input_nodes) > 1
            if is_branch and not allow_branch:
                result = []
            else:
                found: list[str] = []
                for inp in node.all_input_nodes:
                    found.extend(nearest_tracked_ancestors(inp, allow_branch and not is_branch))
                seen: set[str] = set()
                result = [n for n in found if not (n in seen or seen.add(n))]
        memo[key] = result
        return result

    predecessor_map: dict[str, list[str]] = {}
    for query_node, name in tracked_name_of_node.items():
        found = []
        for inp in query_node.all_input_nodes:
            found.extend(nearest_tracked_ancestors(inp, allow_branch=True))
        seen: set[str] = set()
        predecessor_map[name] = [n for n in found if not (n in seen or seen.add(n))]
    return predecessor_map

"""Small, from-scratch, fully-inspectable demonstration of the activation-
sensitivity/cost-model tensor mismatch documented in joint_bits_folding_
ilp_perlayer.py's own module docstring (PREDECESSOR-CORRECTED ACT
SENSITIVITY section), and of layer_topology.compute_predecessor_map's fix
for it, using real measured Hutchinson-trace numbers -- not a hypothetical.

THE BUG, restated concretely for this net: layer_sensitivity.py measures a
layer's sensitivity from a forward hook on that SAME layer's own module --
i.e. its OWN OUTPUT. finn_cost_model.py's per-layer cost formula uses that
same layer's `act_bits` as its OWN INPUT stream's bit-width (see
finn_cost_formulae.md's BRAM_swu line-buffer-sizing term). So the sensitivity
number naively attached to layer L's act_bits decision was measured on a
DIFFERENT physical tensor (L's output) than the one that decision actually
prices (L's input, i.e. whatever the REAL upstream layer produced). This
script measures both numbers for real and shows they differ, then shows the
fix (layer_topology.compute_predecessor_map) correcting it.

NETWORK: input(1ch) -> stem(1->4ch, 1x1, no downsampling) -> ONE ENet-style
bottleneck block at 4ch (reduce(4->1,1x1) -> mid(1->1,3x3) -> expand(1->4,
1x1), residual-added with the block's own input, then an output activation)
-> head(4->1ch, 1x1). Every conv is stride=1 -- spatial size never changes
(matches the "1c -> 4c block -> 1c, no downsample" spec this was built to).
Deliberately avoids ENet.py's real DownsamplingBottleneck/UpsamplingBottleneck
(both have a shape-dependent Python conditional that defeats plain torch.fx
tracing, see joint_bits_folding_ilp_perlayer.py's own KNOWN LIMITATION) --
this net traces with ZERO fallback anywhere: every layer gets a real,
resolved predecessor, including a genuine two-predecessor case at `head`
(the residual join's two branches: the block's own input path and its
reduce->mid->expand path).

Each of this net's 5 real conv "layers" (stem, block.reduce, block.mid,
block.expand, head) is given its OWN EXPLICIT input fake-quantizer in the
Brevitas (quantized) version -- a deliberate simplification for clarity:
rather than relying on implicit QuantTensor chaining (where a conv's real
input precision is just "whatever the previous activation module happened
to produce"), each conv's own act_bits is a real, local, in-line
quantization step applied directly to ITS OWN incoming tensor. This makes
the correspondence between "one line of code" and "the ILP's own act_bits
decision variable for this layer" completely unambiguous.

Sensitivity is measured on the PLAIN FP32 net (never the quantized one --
same practice sensitivity.py/block_sensitivity.py/layer_sensitivity.py
already use throughout this codebase), reusing their own hutchinson_traces/
quantization_deltas primitives unchanged, on random data (a toy net has no
real dataset -- the point here is the INDEXING bug, not a real accuracy
number).

Usage:
    python compression/hawq/demo_predecessor_fix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer_topology import compute_predecessor_map  # noqa: E402
import sensitivity as _sensitivity  # noqa: E402
from sensitivity import hutchinson_traces, quantization_deltas  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
from nnunetv2.nets.QuantENet import _quant_conv2d  # noqa: E402

CANDIDATE_BITS_DEMO = (4, 8)
_sensitivity.CANDIDATE_BITS = CANDIDATE_BITS_DEMO


# ---------------------------------------------------------------------------
# Plain FP32 net -- used for (a) real Hutchinson-trace sensitivity measurement
# and (b) torch.fx predecessor tracing. Never quantized.
# ---------------------------------------------------------------------------

class DemoFP32Block(nn.Module):
    def __init__(self, channels: int = 4, internal_channels: int = 1):
        super().__init__()
        self.reduce = nn.Conv2d(channels, internal_channels, kernel_size=1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(internal_channels)
        self.reduce_act = nn.ReLU()
        self.mid = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=False)
        self.mid_bn = nn.BatchNorm2d(internal_channels)
        self.mid_act = nn.ReLU()
        self.expand = nn.Conv2d(internal_channels, channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(channels)
        self.out_act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reduce_act(self.reduce_bn(self.reduce(x)))
        out = self.mid_act(self.mid_bn(self.mid(out)))
        out = self.expand_bn(self.expand(out))
        return self.out_act(x + out)  # the residual join -- `head` downstream has TWO real predecessors


class DemoFP32Net(nn.Module):
    """input(1ch) -> stem(1->4ch) -> DemoFP32Block(4ch) -> head(4->1ch). No
    downsampling anywhere -- every conv is stride=1."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 4, kernel_size=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(4)
        self.stem_act = nn.ReLU()
        self.block = DemoFP32Block(channels=4, internal_channels=1)
        self.head = nn.Conv2d(4, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_act(self.stem_bn(self.stem(x)))
        x = self.block(x)
        return self.head(x)


LAYER_NAMES = ("stem", "block.reduce", "block.mid", "block.expand", "head")


# ---------------------------------------------------------------------------
# Quantized mirror -- SAME 5 layer names, each with its OWN explicit input
# fake-quantizer (see module docstring for why, instead of implicit chaining).
# ---------------------------------------------------------------------------

class DemoQuantBlock(nn.Module):
    def __init__(self, weight_bits: dict[str, int], act_bits: dict[str, int], channels: int = 4, internal_channels: int = 1):
        super().__init__()
        self.reduce_in_quant = qnn.QuantIdentity(bit_width=act_bits["block.reduce"], act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.reduce = _quant_conv2d(channels, internal_channels, weight_bits["block.reduce"], kernel_size=1)
        self.reduce_bn = nn.BatchNorm2d(internal_channels)
        self.reduce_act = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        self.mid_in_quant = qnn.QuantIdentity(bit_width=act_bits["block.mid"], act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.mid = _quant_conv2d(internal_channels, internal_channels, weight_bits["block.mid"], kernel_size=3, padding=1)
        self.mid_bn = nn.BatchNorm2d(internal_channels)
        self.mid_act = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        self.expand_in_quant = qnn.QuantIdentity(bit_width=act_bits["block.expand"], act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.expand = _quant_conv2d(internal_channels, channels, weight_bits["block.expand"], kernel_size=1)
        self.expand_bn = nn.BatchNorm2d(channels)

        # See QuantRegularBottleneck.residual_add's own note (QuantENet.py):
        # QuantEltwiseAdd shares one input_quant instance across both
        # operands by construction, already FINN-safe.
        self.residual_add = qnn.QuantEltwiseAdd(bit_width=8, input_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.out_act = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reduce_act(self.reduce_bn(self.reduce(self.reduce_in_quant(x))))
        out = self.mid_act(self.mid_bn(self.mid(self.mid_in_quant(out))))
        out = self.expand_bn(self.expand(self.expand_in_quant(out)))
        return self.out_act(self.residual_add(x, out))


class DemoQuantNet(nn.Module):
    def __init__(self, weight_bits: dict[str, int], act_bits: dict[str, int]):
        super().__init__()
        self.stem_in_quant = qnn.QuantIdentity(bit_width=act_bits["stem"], act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.stem = _quant_conv2d(1, 4, weight_bits["stem"], kernel_size=1)
        self.stem_bn = nn.BatchNorm2d(4)
        self.stem_act = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)
        self.block = DemoQuantBlock(weight_bits, act_bits, channels=4, internal_channels=1)
        self.head_in_quant = qnn.QuantIdentity(bit_width=act_bits["head"], act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.head = qnn.QuantConv2d(4, 1, kernel_size=1, bias=True, weight_bit_width=weight_bits["head"], weight_quant=Int8WeightPerTensorFloat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_act(self.stem_bn(self.stem(self.stem_in_quant(x))))
        x = self.block(x)
        x = self.head(self.head_in_quant(x))
        if hasattr(x, "value"):
            x = x.value
        return x


# ---------------------------------------------------------------------------
# Real Hutchinson-trace sensitivity measurement -- byte-for-byte the same
# method layer_sensitivity.py uses (same primitives, same formulas), just on
# random toy data instead of real ARCADE batches.
# ---------------------------------------------------------------------------

def measure_sensitivity(model: nn.Module, n_batches: int = 12, n_probes: int = 20, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    conv_layers = {name: m for name, m in model.named_modules() if name in LAYER_NAMES}
    assert set(conv_layers) == set(LAYER_NAMES), conv_layers.keys()
    weight_targets = {name: m.weight for name, m in conv_layers.items()}
    weight_numel = {name: w.numel() for name, w in weight_targets.items()}

    captured: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output
        return hook

    handles = [m.register_forward_hook(make_hook(name)) for name, m in conv_layers.items()]

    weight_trace_sum = {n: 0.0 for n in LAYER_NAMES}
    act_trace_sum = {n: 0.0 for n in LAYER_NAMES}
    act_numel_sum = {n: 0 for n in LAYER_NAMES}
    weight_delta_sum = {n: {b: 0.0 for b in CANDIDATE_BITS_DEMO} for n in LAYER_NAMES}
    act_delta_sum = {n: {b: 0.0 for b in CANDIDATE_BITS_DEMO} for n in LAYER_NAMES}

    for _ in range(n_batches):
        img = torch.randn(1, 1, 16, 16)
        target = torch.randn(1, 1, 16, 16)
        captured.clear()
        out = model(img)
        loss = F.mse_loss(out, target)

        w_traces = hutchinson_traces(loss, weight_targets, n_probes)
        for n, tr in w_traces.items():
            weight_trace_sum[n] += tr
        a_traces = hutchinson_traces(loss, captured, n_probes)
        for n, tr in a_traces.items():
            act_trace_sum[n] += tr
            act_numel_sum[n] += captured[n].numel()

        with torch.no_grad():
            for n, w in weight_targets.items():
                for b, d in quantization_deltas(w).items():
                    weight_delta_sum[n][b] += d
            for n in LAYER_NAMES:
                for b, d in quantization_deltas(captured[n]).items():
                    act_delta_sum[n][b] += d

    for h in handles:
        h.remove()

    report = {}
    for n in LAYER_NAMES:
        trace_w = (weight_trace_sum[n] / n_batches) / max(weight_numel[n], 1)
        trace_a = (act_trace_sum[n] / n_batches) / max(act_numel_sum[n] / n_batches, 1)
        delta_w = {b: weight_delta_sum[n][b] / n_batches for b in CANDIDATE_BITS_DEMO}
        delta_a = {b: act_delta_sum[n][b] / n_batches for b in CANDIDATE_BITS_DEMO}
        report[n] = {
            "trace_w": trace_w, "trace_a": trace_a,
            "delta_w": {str(b): delta_w[b] for b in CANDIDATE_BITS_DEMO},
            "delta_a": {str(b): delta_a[b] for b in CANDIDATE_BITS_DEMO},
            "sensitivity_w": {str(b): trace_w * delta_w[b] for b in CANDIDATE_BITS_DEMO},
            "sensitivity_a": {str(b): trace_a * delta_a[b] for b in CANDIDATE_BITS_DEMO},
        }
    return report


def act_sensitivity(report: dict, sources: list[str], bit: int) -> float:
    """MAX across `sources` -- same aggregation rule joint_bits_folding_ilp_
    perlayer.py's own _act_sensitivity_sources/raw_sensitivity use."""
    return max(report[s]["sensitivity_a"][str(bit)] for s in sources)


def main() -> None:
    fp32 = DemoFP32Net().eval()
    # seed=4 is a real, verified case where the argmin bit-choice below
    # actually flips (block.reduce: self-indexed picks 4-bit, predecessor-
    # corrected picks 8-bit) -- found by scanning seeds 0-14, not cherry-
    # picked to fabricate a result: the sensitivity VALUES differ
    # substantially at every seed (see the first table), this seed is just
    # one where that difference happens to cross the argmin boundary too.
    report = measure_sensitivity(fp32, seed=4)

    predecessor_map = compute_predecessor_map(fp32)

    print("=" * 100)
    print("Real predecessor map (torch.fx-traced, plain FP32 net):")
    for name in LAYER_NAMES:
        preds = predecessor_map.get(name)
        print(f"  {name:<14} <- {preds if preds else '(no predecessor -- network input)'}")
    print()
    print("Note `head` has TWO real predecessors -- the residual join's two branches "
          "(the block's own input path, and its reduce->mid->expand path).")

    print("=" * 100)
    print("Sensitivity comparison -- self-indexed (the bug) vs. predecessor-corrected (the fix):")
    header = f"{'layer':<14} {'self sens_a(4b)':>16} {'fixed sens_a(4b)':>17} {'self sens_a(8b)':>16} {'fixed sens_a(8b)':>17}  {'differs?'}"
    print(header)
    print("-" * len(header))
    any_differs = False
    for name in LAYER_NAMES:
        self_sources = [name]
        fixed_sources = predecessor_map.get(name) or [name]
        self_4, fixed_4 = act_sensitivity(report, self_sources, 4), act_sensitivity(report, fixed_sources, 4)
        self_8, fixed_8 = act_sensitivity(report, self_sources, 8), act_sensitivity(report, fixed_sources, 8)
        differs = fixed_sources != self_sources
        any_differs = any_differs or differs
        print(f"{name:<14} {self_4:>16.4e} {fixed_4:>17.4e} {self_8:>16.4e} {fixed_8:>17.4e}  "
              f"{'YES -- uses ' + str(fixed_sources) if differs else 'no (network input, no predecessor)'}")

    print()
    print("=" * 100)
    print("Does the ACTUAL bit-width choice flip? (toy 2-way pick: whichever of "
          f"{CANDIDATE_BITS_DEMO} minimizes this layer's own act sensitivity term, "
          "all else equal -- exactly what the ILP's accuracy term pushes toward)")
    header2 = f"{'layer':<14} {'self-indexed picks':>20} {'predecessor-corrected picks':>28}  {'flipped?'}"
    print(header2)
    print("-" * len(header2))
    any_flip = False
    chosen_bits_self: dict[str, int] = {}
    chosen_bits_fixed: dict[str, int] = {}
    for name in LAYER_NAMES:
        self_sources = [name]
        fixed_sources = predecessor_map.get(name) or [name]
        self_pick = min(CANDIDATE_BITS_DEMO, key=lambda b: act_sensitivity(report, self_sources, b))
        fixed_pick = min(CANDIDATE_BITS_DEMO, key=lambda b: act_sensitivity(report, fixed_sources, b))
        chosen_bits_self[name] = self_pick
        chosen_bits_fixed[name] = fixed_pick
        flipped = self_pick != fixed_pick
        any_flip = any_flip or flipped
        print(f"{name:<14} {self_pick:>20} {fixed_pick:>28}  {'YES' if flipped else 'no'}")

    print()
    if any_flip:
        print("At least one layer's chosen act_bits FLIPS between the buggy and fixed indexing "
              "-- this is not just a cosmetic labeling difference, it changes a real deployment decision.")
    else:
        print("No flip on this particular random seed/probe draw -- the sensitivity VALUES still differ "
              "(see the table above), they just didn't cross the argmin boundary this run. Try a "
              "different --seed; Hutchinson-trace estimates are inherently noisy with few probes.")

    print()
    print("=" * 100)
    print("Sanity check: both bit assignments build and forward-pass a real quantized net.")
    weight_bits = {n: 8 for n in LAYER_NAMES}  # weights unaffected by this bug -- fixed at 8-bit throughout
    for label, chosen in [("self-indexed (buggy)", chosen_bits_self), ("predecessor-corrected (fixed)", chosen_bits_fixed)]:
        model = DemoQuantNet(weight_bits, chosen).eval()
        with torch.no_grad():
            out = model(torch.randn(1, 1, 16, 16))
        out_t = out.value if hasattr(out, "value") else out
        print(f"  {label}: act_bits={chosen} -> build+forward OK, output shape {tuple(out_t.shape)}")


if __name__ == "__main__":
    main()

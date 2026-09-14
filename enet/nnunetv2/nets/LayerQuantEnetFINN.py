"""FINN-compatible mirror of LayerQuantENet (see LayerQuantENet.py) --
structurally identical wherever FINN can lower the real op directly
(regular1/regular4/regular5/stage2/stage3 reuse LayerQuantRegularBottleneck
UNMODIFIED), with 4 substitutions where the real op has no FINN hardware
lowering path at all. Originally developed in hardware/finn_export_
12_dense_relu_warmstart150ep_alpha025_dummy.py; moved here (2026-09-14) to
live alongside the other nnunetv2/nets/*.py architectures rather than as an
export-script-local definition, since it's now warmstarted/fine-tuned as a
real network in its own right, not just a one-shot export helper.

4 substitutions vs. the real LayerQuantENet -- see memories/repo/
finn_12_dense_relu_alpha025_perlayer.md and memories/repo/finn_gotchas.md
for the full derivation/verification of each:

0. FINNInitialBlockConcat: real LayerQuantInitialBlock does
   `torch.cat([self.conv(x), self.pool(x)], dim=1)` with NO shared quantizer
   forcing the conv branch and the pool branch onto the same (scale,
   zero_point) pair first. Brevitas's QuantTensor-aware `cat` requires
   coherent quant params across all its inputs to stay in the QuantTensor
   representation, and silently falls back to a plain float tensor when
   they differ -- confirmed against a real preamble build's
   step_enet_convert_to_hw.onnx: Concat_0's inputs are annotated FLOAT32,
   not integer, which is why Concat/the BN-affine Mul+Add/MaxPool never
   lower to FINN HW nodes (every Infer*Layer HW-conversion pass hard-
   requires an integer input DataType). Fix, and ONLY departure from the
   real block: one new shared `branch_quant` (qnn.QuantIdentity instance,
   no HAWQ site -> bit-width matched to the chain) applied to BOTH
   self.conv(x) and self.pool(x) right before the cat, forcing them onto
   one common scale so the cat stays integer-annotated all the way
   through. conv/pool/bn/act are otherwise identical (same modules, same
   real per-site weights, same site names) to LayerQuantInitialBlock.
1. FINNDownsamplingBottleneck: real down1/down2 shortcut is a
   parameter-free MaxPool + zero-pad-to-out_channels (see
   LayerQuantDownsamplingBottleneck.forward). Concat-with-a-zeros-tensor
   isn't confirmed FINN-exportable, so this substitutes a 1x1 conv with a
   FIXED (frozen, requires_grad=False) padded-identity weight
   (W[c,c]=1 for c<in_ch, 0 elsewhere) + zero bias -- mathematically
   IDENTICAL to the real op, not an approximation. No new weight/act site:
   `shortcut_proj` has no HAWQ-searched bit-width and needs none (frozen).
   Built as qnn.QuantConv2d (INT8 weight_quant), NOT plain nn.Conv2d: a
   plain conv's weight exports as a bare FLOAT32 initializer, and FINN does
   NOT infer an integer DataType for it even though every value is a
   literal 0.0/1.0, so an explicit Quant wrapper is required -- same fix
   already applied to `main_up` below.
2. FINNUpsamplingBottleneck: real up4/up5 main path is main_proj (real,
   trained, transferred as-is) -> F.interpolate(bilinear,
   align_corners=False) (LayerQuantENet always passes indices=None in
   decoder_type="upsample_conv" mode -- confirmed NEVER MaxUnpool2d).
   Resize/interpolate isn't FINN-synthesizable, so this substitutes a fixed
   (frozen) nn.Upsample(mode="nearest", scale_factor=2) followed by a fixed
   (frozen) depthwise 3x3 conv with the exact bilinear-equivalent "tent"
   kernel outer([1,2,1]/4, [1,2,1]/4) -- derived and verified (both on
   synthetic tensors and on real trained up4/up5 activations, see
   hardware/testbench_bilinear_vs_nearest_depthwise_up4up5.py) to match
   F.interpolate EXACTLY (float-rounding-level, ~5e-7) on every interior
   pixel; only the outermost 1px border ring differs (zero-pad vs. the real
   op's edge-replicate/clamp assumption at the image border, <1% relative
   error overall). An edge-replicate Pad fix was considered and REJECTED --
   confirmed via direct source read (2026-09-14) that BOTH of FINN's
   padding HW ops (fmpadding.py, fmpadding_pixel.py) hardcode zero-fill
   only (`InferConvInpGen` asserts `pad_val == 0`; FMPadding_Pixel's own
   `execute_node` is a literal `np.zeros(...)` fill) -- there is no
   edge/replicate/reflect padding primitive anywhere in FINN's op set, so
   full bit-exactness at the border is not achievable in real hardware
   without a brand-new custom FINN op (out of scope). main_up stays
   FROZEN (requires_grad=False, no HAWQ site): the real op it approximates,
   F.interpolate, has zero trainable parameters, so this substitute must
   not introduce a new learnable layer either. Built DEPTHWISE
   (groups=channels), matching the real math exactly (each output channel
   only ever mixes its own input channel's spatial neighborhood) -- this
   is this architecture's first VVAU (depthwise) node, but the
   Upsample(nearest)->depthwise-conv HW lowering path (Resize ->
   UpsampleNearestNeighbour_hls, depthwise conv -> VVAU_hls) was already
   independently built and verified end-to-end in finn_build_probe_
   upsample_nearest_depthwise_int8.py (see memories/repo/finn_gotchas.md,
   2026-09-14 entry) -- not an untested path.
3. `final` bias: real LayerQuantENet's `final` is bias=True with a plain
   (unquantized) float bias Add -- fine on GPU/CPU but not FINN-HW-
   convertible (MVAU has no bias input at all, and InferChannelwiseLinearLayer
   requires the constant to be exactly integer-valued). Fix:
   bias_quant=Int32Bias, which quantizes the bias into the MatMul
   accumulator's own implicit scale (input_scale * weight_scale), making it
   exact-integer and letting the resulting Add lower to a real
   ChannelwiseOp HW node instead of a CPU-side post-processing step.
"""
from __future__ import annotations

import torch
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int32Bias

from nnunetv2.nets.QuantENet import _quant_conv2d, _quant_act, _quant_block_act
from nnunetv2.nets.LayerQuantENet import _make_layer_shallow_stage, _make_layer_context_stage, _local_single


# ---------------------------------------------------------------------------
# FINN-safe block substitutes -- initial, down1/down2 and up4/up5 need one
# (see module docstring). regular1/regular4/regular5/stage2/stage3 reuse
# LayerQuantRegularBottleneck unmodified.
# ---------------------------------------------------------------------------

class FINNInitialBlockConcat(nn.Module):
    """FINN-safe substitute for LayerQuantInitialBlock -- see module
    docstring point 0. conv/pool/bn/act are the real per-site ops, real
    weights transfer directly (same site names: "conv", "input_quant",
    "act"); `branch_quant` is the one new, no-HAWQ-site rounding point that
    this substitute adds, bit-width chosen to match the chain (see below)
    rather than a flat fallback."""

    def __init__(self, in_channels: int, out_channels: int, weight_bits: dict[str, int], act_bits: dict[str, int]):
        super().__init__()
        if out_channels <= in_channels:
            raise ValueError("FINNInitialBlockConcat out_channels must exceed in_channels.")
        self.input_quant = qnn.QuantIdentity(
            bit_width=act_bits["input_quant"], act_quant=Int8ActPerTensorFloat, return_quant_tensor=True,
        )
        self.conv = _quant_conv2d(in_channels, out_channels - in_channels, weight_bits["conv"], kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # bit-width matches the wider of this block's own "prior" (input_quant,
        # pool's true source precision) and "post" (act, its eventual consumer)
        # so this new rounding point never becomes an information bottleneck.
        branch_bits = max(act_bits["input_quant"], act_bits["act"])
        self.branch_quant = qnn.QuantIdentity(bit_width=branch_bits, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = _quant_block_act(out_channels, act_bits["act"], None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        branches = [self.branch_quant(self.conv(x)), self.branch_quant(self.pool(x))]
        return self.act(self.bn(torch.cat(branches, dim=1)))


def _padded_identity_conv(in_channels: int, out_channels: int) -> qnn.QuantConv2d:
    """Fixed (frozen, non-learned) 1x1 conv reproducing "zero-pad channels
    to out_channels" EXACTLY -- see module docstring point 1.

    A plain (non-quantized) nn.Conv2d here exports its weight as a bare
    FLOAT32 ONNX initializer: even though every value is a literal 0.0/1.0,
    FINN's tidy-up does NOT infer an integer DataType for it, so
    InferQuantizedMatrixVectorActivation's `wdt.is_integer()` gate fails and
    the conv never lowers to an MVAU -- same failure mode already found and
    fixed for `main_up`'s bilinear kernel (see
    _nearest_depthwise_bilinear_kernel below). Fix: wrap in qnn.QuantConv2d
    (INT8 weight quant) so an explicit Quant node forces a real integer
    weight DataType, then overwrite+freeze the weight exactly as before."""
    conv = qnn.QuantConv2d(
        in_channels, out_channels, kernel_size=1, bias=False,
        weight_bit_width=8, weight_quant=Int8WeightPerTensorFloat,
    )
    weight = torch.zeros(out_channels, in_channels, 1, 1)
    for c in range(min(in_channels, out_channels)):
        weight[c, c, 0, 0] = 1.0
    with torch.no_grad():
        conv.weight.copy_(weight)
    conv.weight.requires_grad_(False)
    return conv


def _nearest_depthwise_bilinear_kernel(channels: int) -> nn.Sequential:
    """Fixed (frozen, non-learned) nn.Upsample(mode="nearest") followed by a
    fixed depthwise conv, reproducing 2x bilinear upsampling
    (align_corners=False) exactly on interior pixels -- see module
    docstring point 2 and hardware/testbench_bilinear_vs_nearest_depthwise_
    up4up5.py (verified against real trained up4/up5 activations: interior
    max abs err ~5e-7, error only at the 1px border ring -- edge-replicate
    padding is NOT available in FINN, see module docstring, so this border
    mismatch is accepted as-is).

    Stays frozen (requires_grad=False): the real op this approximates,
    F.interpolate(bilinear), has ZERO trainable parameters, so this must
    not introduce a new learnable layer either. Built DEPTHWISE
    (groups=channels), matching the real math exactly.

    A plain (non-quantized) nn.Conv2d here would export as a bare float
    ONNX Conv node and get silently excluded from the HW partition. Fix:
    wrap in qnn.QuantConv2d (INT8 weight quant), overwrite the weight with
    the exact fixed tent kernel, and freeze it."""
    dw = qnn.QuantConv2d(
        channels, channels, kernel_size=3, padding=1, groups=channels, bias=False,
        weight_bit_width=8, weight_quant=Int8WeightPerTensorFloat,
    )
    k1d = torch.tensor([1.0, 2.0, 1.0]) / 4.0
    k2d = torch.outer(k1d, k1d)
    weight = k2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    with torch.no_grad():
        dw.weight.copy_(weight)
    dw.weight.requires_grad_(False)
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), dw)


class FINNDownsamplingBottleneck(nn.Module):
    """FINN-safe substitute for LayerQuantDownsamplingBottleneck -- only the
    shortcut ("main") branch differs; reduce/conv/expand/residual_add/
    out_act are the real per-site ops, real weights transfer directly (same
    site names as the real block: "reduce.0"/"conv.0"/"expand.0"/
    "residual_add"/"out_act", assuming use_strided=True)."""

    def __init__(
        self, in_channels: int, out_channels: int, weight_bits: dict[str, int], act_bits: dict[str, int],
        internal_ratio: int = 4, dropout_p: float = 0.01,
    ):
        super().__init__()
        internal_channels = max(1, out_channels // internal_ratio)

        self.shortcut_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.shortcut_proj = _padded_identity_conv(in_channels, out_channels)

        self.reduce = nn.Sequential(
            _quant_conv2d(in_channels, internal_channels, weight_bits["reduce.0"], kernel_size=2, stride=2),
            nn.BatchNorm2d(internal_channels), _quant_act(act_bits["reduce.2"]),
        )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_channels, internal_channels, weight_bits["conv.0"], kernel_size=3, padding=1),
            nn.BatchNorm2d(internal_channels), _quant_act(act_bits["conv.2"]),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_channels, out_channels, weight_bits["expand.0"], kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.residual_add = qnn.QuantEltwiseAdd(
            bit_width=act_bits["residual_add"], input_quant=Int8ActPerTensorFloat, return_quant_tensor=True,
        )
        self.out_act = _quant_act(act_bits["out_act"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.shortcut_proj(self.shortcut_pool(x))
        out = self.dropout(self.expand(self.conv(self.reduce(x))))
        return self.out_act(self.residual_add(main, out))


class FINNUpsamplingBottleneck(nn.Module):
    """FINN-safe substitute for LayerQuantUpsamplingBottleneck -- main_proj
    (real, transferred as-is) is unchanged; only the spatial-upsample step
    after it is substituted (fixed, frozen nearest+depthwise bilinear-
    equivalent kernel, see module docstring point 2). reduce/up/expand/
    residual_add/out_act are the real per-site ops (same site names as the
    real block)."""

    def __init__(
        self, in_channels: int, out_channels: int, weight_bits: dict[str, int], act_bits: dict[str, int],
        internal_ratio: int = 4,
    ):
        super().__init__()
        internal_channels = max(1, in_channels // internal_ratio)

        self.main_proj = nn.Sequential(
            _quant_conv2d(in_channels, out_channels, weight_bits["main_proj.0"], kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        # real main_proj has no activation quantizer here (BN feeds straight into
        # F.interpolate) -- but main_up (see below) is now a real Quant-wrapped HW
        # conv, and FINN's Conv->MVAU/VVAU fusion pass requires a properly-quantized
        # (integer-datatype) INPUT to fuse, else it leaves a dangling non-HW
        # Im2Col+MatMul+MultiThreshold triple that breaks partitioning. Add an
        # explicit signed passthrough quantizer (FINN-export-only, no HAWQ site)
        # to bridge this gap, bit-width matched to residual_add -- the one real
        # "post" consumer this branch's output feeds into after main_up.
        self.main_act = qnn.QuantIdentity(bit_width=act_bits["residual_add"], act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.main_up = _nearest_depthwise_bilinear_kernel(out_channels)

        self.reduce = nn.Sequential(
            _quant_conv2d(in_channels, internal_channels, weight_bits["reduce.0"], kernel_size=1),
            nn.BatchNorm2d(internal_channels), _quant_act(act_bits["reduce.2"]),
        )
        self.up = nn.Sequential(
            qnn.QuantConvTranspose2d(
                internal_channels, internal_channels, kernel_size=2, stride=2, bias=False,
                weight_bit_width=weight_bits["up.0"], weight_quant=Int8WeightPerTensorFloat,
            ),
            nn.BatchNorm2d(internal_channels), _quant_act(act_bits["up.2"]),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_channels, out_channels, weight_bits["expand.0"], kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=0.1)
        self.residual_add = qnn.QuantEltwiseAdd(
            bit_width=act_bits["residual_add"], input_quant=Int8ActPerTensorFloat, return_quant_tensor=True,
        )
        self.out_act = _quant_act(act_bits["out_act"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main_up(self.main_act(self.main_proj(x)))
        out = self.dropout(self.expand(self.up(self.reduce(x))))
        return self.out_act(self.residual_add(main, out))


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------

class LayerQuantEnetFINN(nn.Module):
    """FINN-compatible mirror of a LayerQuantENet configuration.
    regular1/regular4/regular5/stage2/stage3 are the REAL, unmodified
    LayerQuantRegularBottleneck; initial/down1/down2/up4/up5 are the
    FINN-safe (but numerically exact except for `initial`'s new
    branch_quant, see module docstring) substitutes above. `final`'s bias
    (bias=True, bias_quant=Int32Bias) quantizes the bias into the MatMul
    accumulator's own scale (input_scale * weight_scale), which makes it an
    exact-integer constant -- letting InferChannelwiseLinearLayer lower the
    resulting Add straight to a real ChannelwiseOp HW node instead of a
    CPU-side post-processing step."""

    def __init__(
        self, layer_weight_bits: dict[str, int], layer_act_bits: dict[str, int], *,
        in_channels: int = 1, out_channels: int = 5,
        channels: tuple[int, int, int, int, int], bottlenecks_per_stage: tuple[int, int, int, int, int],
        context_pattern: str, final_bias: bool = True,
    ):
        super().__init__()
        c0, c1, c23, c4, c5 = channels
        n1, n2, n3, n4, n5 = bottlenecks_per_stage

        self.initial = FINNInitialBlockConcat(
            in_channels, c0, _local_single(layer_weight_bits, "initial"), _local_single(layer_act_bits, "initial"),
        )

        self.down1 = FINNDownsamplingBottleneck(
            c0, c1, _local_single(layer_weight_bits, "down1"), _local_single(layer_act_bits, "down1"), dropout_p=0.01,
        )
        self.regular1 = _make_layer_shallow_stage(c1, n1, layer_weight_bits, layer_act_bits, 0.01, "regular1", {})

        self.down2 = FINNDownsamplingBottleneck(
            c1, c23, _local_single(layer_weight_bits, "down2"), _local_single(layer_act_bits, "down2"), dropout_p=0.1,
        )
        self.stage2 = _make_layer_context_stage(
            c23, n2, layer_weight_bits, layer_act_bits, "stage2", {}, context_pattern, separable_dilated=False,
        )
        self.stage3 = _make_layer_context_stage(
            c23, n3, layer_weight_bits, layer_act_bits, "stage3", {}, context_pattern, separable_dilated=False,
        )

        self.up4 = FINNUpsamplingBottleneck(
            c23, c4, _local_single(layer_weight_bits, "up4"), _local_single(layer_act_bits, "up4"),
        )
        self.regular4 = _make_layer_shallow_stage(c4, n4, layer_weight_bits, layer_act_bits, 0.1, "regular4", {})

        self.up5 = FINNUpsamplingBottleneck(
            c4, c5, _local_single(layer_weight_bits, "up5"), _local_single(layer_act_bits, "up5"),
        )
        self.regular5 = _make_layer_shallow_stage(c5, n5, layer_weight_bits, layer_act_bits, 0.1, "regular5", {})

        self.final = qnn.QuantConvTranspose2d(
            c5, out_channels, kernel_size=2, stride=2, bias=final_bias,
            bias_quant=Int32Bias if final_bias else None,
            weight_bit_width=layer_weight_bits["final"], weight_quant=Int8WeightPerTensorFloat,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.regular1(self.down1(x))
        x = self.stage2(self.down2(x))
        x = self.stage3(x)
        x = self.regular4(self.up4(x))
        x = self.regular5(self.up5(x))
        out = self.final(x)
        return out.value if hasattr(out, "value") else out

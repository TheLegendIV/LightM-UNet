"""Per-INDIVIDUAL-QUANTIZER-SITE clone of CombinedQuantENet.py -- instead of
one shared (weight_bit_width, act_bit_width) pair per whole bottleneck BLOCK,
every conv weight site (reduce/conv/expand, and any internal split-conv site)
and every activation site (reduce's act, the post-conv act, the residual-join
act, out_act, ...) inside a block gets its OWN independent bit-width.

WHY A NEW FILE, NOT AN EDIT TO QuantENet.py/CombinedQuantENet.py: Brevitas
bakes weight_bit_width/bit_width into a fixed quantizer proxy at construction
time, so per-site granularity requires a signature change (scalar int ->
dict[str, int]) to every block primitive -- QuantENet.py's own module
docstring says it's deliberately kept separate from ENet.py "so the FP32
architecture... can't be perturbed by speculative Brevitas wiring," and it is
now load-bearing for 10 real, currently-training CombinedQuantENet-based
trainers. This file defines its own sibling primitive classes
(LayerQuantInitialBlock, LayerQuantRegularBottleneck,
LayerQuantDSCNoProjectionBottleneck, LayerQuantDownsamplingBottleneck,
LayerQuantUpsamplingBottleneck) -- structural clones of QuantENet.py's own,
generalized only in how they look up bit-widths -- rather than touching
anything that already has a real trained checkpoint against it.

SITE NAMING: a site's key is its exact named_modules() dotted path relative
to the whole LayerQuantENet instance (e.g. "regular1.0.reduce.0",
"stage2.3.conv.3", "down1.residual_add", "final") -- not an invented scheme,
the literal path PyTorch's own module-tree walk produces. Confirmed
(brevitas==0.12.1) that qnn.QuantConv2d IS-A nn.Conv2d and
qnn.QuantConvTranspose2d IS-A nn.ConvTranspose2d, so this is mechanically
guaranteed to match compression/hawq/block_utils.block_weight_targets's own
naming for the WEIGHT half (verified: running block_weight_targets against a
built LayerQuantENet instance's underlying blocks produces the identical key
set layer_names_for's own weight half does -- see this file's own __main__
self-test and compression/hawq/layer_sensitivity.py, whose report keys this
was built to line up with directly). The ACTIVATION half has no external
naming contract to satisfy (FP32 ENet.py has no activation-quantizer-
equivalent site at all) -- it uses this same dotted-path convention purely
for its own internal self-consistency.

layer_names_for(...) discovers the expected site-name set by CONSTRUCTING a
throwaway, uniformly-8-bit instance and introspecting it (not by hand-deriving
a second copy of _make_layer_context_stage's pattern-branching logic) --
site NAMES are a pure function of the architecture-shape/pattern flags, never
of which bit-width is chosen at a site, so an 8-bit throwaway's names are
identical to the real model's, and the name list can never drift out of sync
with the real assembly code because it *is* the real assembly code
(_build_layer_enet_modules is the single shared body both the real build and
the discovery build call).

Usage (not yet wired into a real trainer -- see the sibling
nnUNetTrainerLayerQuantENet_12_separable_dense_relu_perlayer.py):
    from nnunetv2.nets.LayerQuantENet import LayerQuantENet, layer_names_for, expand_uniform_layer_bits

    weight_names, act_names = layer_names_for(
        out_channels=5, channels=(4, 16, 32, 16, 4), bottlenecks_per_stage=(4, 8, 8, 2, 1),
        context_pattern="dense_dilation", separable_dilated=True,
    )
    layer_weight_bits, layer_act_bits = expand_uniform_layer_bits(8, 8, weight_names, act_names)
    model = LayerQuantENet(
        layer_weight_bits, layer_act_bits, out_channels=5, channels=(4, 16, 32, 16, 4),
        bottlenecks_per_stage=(4, 8, 8, 2, 1), context_pattern="dense_dilation", separable_dilated=True,
    )
"""
from __future__ import annotations

import collections
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

from nnunetv2.nets.ENet import (
    CONTEXT_STAGE_PATTERN,
    DENSE_DILATION_PATTERN,
    DENSE_DILATION_REG_INTERLEAVED_PATTERN,
    DENSE_DILATION_REG_INTERLEAVED_DOUBLE_MID_PATTERN,
    DENSE_DILATION_D2_PROJECTED_PATTERN,
    DENSE_DILATION_D8_D16_PROJECTED_PATTERN,
    DENSE_DILATION_D2_REGULAR_PATTERN,
    DENSE_DILATION_REG_TRAILING_PATTERN,
)
from nnunetv2.nets.QuantENet import (
    _quant_act,
    _quant_block_act,
    _quant_conv2d,
    QuantDecomposedLeakyAct,
    QuantFusedLeakyAct,
)

VALID_CONTEXT_PATTERNS = (
    "default", "dense_dilation", "dense_dilation_reg_interleaved",
    "dense_dilation_reg_interleaved_double_mid",
    "dense_dilation_d2_projected", "dense_dilation_d8_d16_projected",
    "dense_dilation_d2_regular", "dense_dilation_reg_trailing",
)

# QuantDecomposedLeakyAct/QuantFusedLeakyAct contain their OWN internal
# QuantIdentity/QuantReLU submodules (pre_quant/act_pos/out_quant) -- a flat
# isinstance-filtered named_modules() walk would wrongly also match those,
# inflating one real site into 3-4 spurious ones. _first_match_modules below
# stops descending the instant a module itself matches one of these types.
ACT_SITE_TYPES = (qnn.QuantReLU, qnn.QuantIdentity, QuantDecomposedLeakyAct, QuantFusedLeakyAct, qnn.QuantEltwiseAdd)


def _first_match_modules(
    module: nn.Module, types: tuple[type, ...], prefix: str = "", _memo: set[int] | None = None,
) -> dict[str, nn.Module]:
    """Like a type-filtered named_modules() walk, but does NOT descend into a
    module's children once that module itself matched `types` -- required so
    a QuantDecomposedLeakyAct/QuantFusedLeakyAct instance counts as ONE site,
    not one plus its internal quantizers. Safe for the non-leaky case too
    (plain QuantReLU/QuantIdentity/QuantEltwiseAdd have no matching
    descendants of their own).

    _memo replicates named_modules()'s own remove_duplicate=True behavior --
    REQUIRED here, not optional: LayerQuantRegularBottleneck's conv_bn_act is
    `nn.Sequential(self.conv, BN, act)`, i.e. conv_bn_act[0] is the SAME
    object as self.conv (registered under "conv" first). Without tracking
    visited module identity, this walker would revisit that shared conv
    Sequential a second time via "conv_bn_act.0" and find its real "conv.2"
    intermediate-act instance again under the bogus alias "conv_bn_act.0.2"
    -- a real bug caught by this file's own H self-test (a naive first
    version reported 6 act sites instead of 5 for the separable_dilated,
    dilation!=1 branch)."""
    if _memo is None:
        _memo = set()
    found: dict[str, nn.Module] = {}
    for name, child in module.named_children():
        if id(child) in _memo:
            continue
        _memo.add(id(child))
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, types):
            found[full_name] = child
            continue
        found.update(_first_match_modules(child, types, full_name, _memo))
    return found


def _local(bits: dict[str, int], name_prefix: str, i: int, placeholder: int | None = None) -> dict[str, int]:
    """Extracts the local per-site sub-dict for one loop-index instance (e.g.
    "regular1.0.reduce.0" -> "reduce.0" when name_prefix="regular1", i=0). A
    plain dict in the real build (placeholder=None -- a genuinely missing key
    raises KeyError as a correctness backstop, even though LayerQuantENet's
    own validation should already have caught it earlier); a defaultdict
    wrapping the already-filtered local dict in discovery mode (placeholder=8)
    -- the defaultdict can't be the ITERATION source itself, since its own
    prefix-filtering scan would find nothing to iterate until a key had
    already been looked up once."""
    prefix = f"{name_prefix}.{i}."
    local = {k[len(prefix):]: v for k, v in bits.items() if k.startswith(prefix)}
    if placeholder is not None:
        return collections.defaultdict(lambda: placeholder, local)
    return local


def _local_single(bits: dict[str, int], attr: str, placeholder: int | None = None) -> dict[str, int]:
    """Same as _local but for a single-instance top-level block (initial/
    down1/down2/up4/up5 -- no ".<i>" loop index in the name)."""
    prefix = f"{attr}."
    local = {k[len(prefix):]: v for k, v in bits.items() if k.startswith(prefix)}
    if placeholder is not None:
        return collections.defaultdict(lambda: placeholder, local)
    return local


class LayerQuantInitialBlock(nn.Module):
    """Per-site clone of QuantENet.QuantInitialBlock. Weight sites: "conv"
    (1). Act sites: "input_quant", "act" (2)."""

    def __init__(
        self, in_channels: int, out_channels: int, weight_bits: dict[str, int], act_bits: dict[str, int],
        negative_slope: float | None = None, trainable_slope: bool = False,
        internal_bit_width: int | None = None, fused_leaky: bool = False, alpha_bit_width: int = 8,
        out_bit_width: int | None = None,
    ):
        super().__init__()
        if out_channels <= in_channels:
            raise ValueError("LayerQuantInitialBlock out_channels must exceed in_channels.")
        self.input_quant = qnn.QuantIdentity(bit_width=act_bits["input_quant"], act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.conv = _quant_conv2d(in_channels, out_channels - in_channels, weight_bits["conv"], kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = _quant_block_act(
            out_channels, act_bits["act"], negative_slope, trainable_slope=trainable_slope,
            internal_bit_width=internal_bit_width, fused_leaky=fused_leaky, alpha_bit_width=alpha_bit_width,
            out_bit_width=out_bit_width,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        return self.act(self.bn(torch.cat([self.conv(x), self.pool(x)], dim=1)))


class LayerQuantRegularBottleneck(nn.Module):
    """Per-site clone of QuantENet.QuantRegularBottleneck. Weight sites:
    "reduce.0", "expand.0" (always), plus "conv" (plain branch, bare -- a
    single non-Sequential conv attribute) OR "conv.0"+"conv.1" (use_dsc) OR
    "conv.0"+"conv.3" (asymmetric, or separable_dilated with dilation!=1).
    Act sites: "reduce.2", "conv_bn_act.2", "out_act", "residual_add"
    (always), plus "conv.2" (asymmetric/separable_dilated-with-dilation!=1
    branch only, the intermediate act between the two split convs)."""

    def __init__(
        self, channels: int, weight_bits: dict[str, int], act_bits: dict[str, int], internal_ratio: int = 4,
        kernel_size: int = 3, padding: int = 1, dilation: int = 1,
        asymmetric: bool = False, dropout_p: float = 0.1, use_dsc: bool = False,
        separable_dilated: bool = False, negative_slope: float | None = None, trainable_slope: bool = False,
        internal_bit_width: int | None = None, fused_leaky: bool = False, alpha_bit_width: int = 8,
        out_bit_width: int | None = None,
    ):
        super().__init__()
        internal_channels = max(1, channels // internal_ratio)

        def _act(ch: int, key: str) -> nn.Module:
            return _quant_block_act(
                ch, act_bits[key], negative_slope, trainable_slope=trainable_slope,
                internal_bit_width=internal_bit_width, fused_leaky=fused_leaky,
                alpha_bit_width=alpha_bit_width, out_bit_width=out_bit_width,
            )

        self.reduce = nn.Sequential(
            _quant_conv2d(channels, internal_channels, weight_bits["reduce.0"], kernel_size=1),
            nn.BatchNorm2d(internal_channels),
            _act(internal_channels, "reduce.2"),
        )
        if asymmetric:
            if use_dsc:
                raise ValueError("use_dsc is not defined for asymmetric bottlenecks -- see ENet.py's RegularBottleneck.")
            self.conv = nn.Sequential(
                _quant_conv2d(internal_channels, internal_channels, weight_bits["conv.0"], kernel_size=(kernel_size, 1), padding=(padding, 0)),
                nn.BatchNorm2d(internal_channels),
                _act(internal_channels, "conv.2"),
                _quant_conv2d(internal_channels, internal_channels, weight_bits["conv.3"], kernel_size=(1, kernel_size), padding=(0, padding)),
            )
        elif use_dsc:
            self.conv = nn.Sequential(
                _quant_conv2d(internal_channels, internal_channels, weight_bits["conv.0"], kernel_size=kernel_size,
                               padding=padding, dilation=dilation, groups=internal_channels),
                _quant_conv2d(internal_channels, internal_channels, weight_bits["conv.1"], kernel_size=1),
            )
        elif separable_dilated and dilation != 1:
            self.conv = nn.Sequential(
                _quant_conv2d(internal_channels, internal_channels, weight_bits["conv.0"], kernel_size=(kernel_size, 1),
                               padding=(padding, 0), dilation=dilation),
                nn.BatchNorm2d(internal_channels),
                _act(internal_channels, "conv.2"),
                _quant_conv2d(internal_channels, internal_channels, weight_bits["conv.3"], kernel_size=(1, kernel_size),
                               padding=(0, padding), dilation=dilation),
            )
        else:
            self.conv = _quant_conv2d(internal_channels, internal_channels, weight_bits["conv"], kernel_size=kernel_size, padding=padding, dilation=dilation)

        self.conv_bn_act = nn.Sequential(self.conv, nn.BatchNorm2d(internal_channels), _act(internal_channels, "conv_bn_act.2"))
        self.expand = nn.Sequential(
            _quant_conv2d(internal_channels, channels, weight_bits["expand.0"], kernel_size=1),
            nn.BatchNorm2d(channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        # See QuantRegularBottleneck.residual_add's own note (QuantENet.py) --
        # QuantEltwiseAdd shares one input_quant instance (learned scale)
        # across both operands by construction, already FINN-safe.
        self.residual_add = qnn.QuantEltwiseAdd(bit_width=act_bits["residual_add"], input_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.out_act = _act(channels, "out_act")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reduce(x)
        out = self.conv_bn_act(out)
        out = self.dropout(self.expand(out))
        return self.out_act(self.residual_add(x, out))


class LayerQuantDSCNoProjectionBottleneck(nn.Module):
    """Per-site clone of QuantENet.QuantDSCNoProjectionBottleneck. Weight
    sites: "conv.0" (depthwise), "conv.3" (pointwise). Act sites: "conv.2"
    (intermediate), "residual_add", "out_act"."""

    def __init__(
        self, channels: int, weight_bits: dict[str, int], act_bits: dict[str, int], kernel_size: int = 3,
        padding: int = 1, dilation: int = 1, dropout_p: float = 0.1, negative_slope: float | None = None,
        trainable_slope: bool = False,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            _quant_conv2d(channels, channels, weight_bits["conv.0"], kernel_size=kernel_size,
                          padding=padding, dilation=dilation, groups=channels),
            nn.BatchNorm2d(channels),
            _quant_block_act(channels, act_bits["conv.2"], negative_slope, trainable_slope=trainable_slope),
            _quant_conv2d(channels, channels, weight_bits["conv.3"], kernel_size=1),
            nn.BatchNorm2d(channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.residual_add = qnn.QuantEltwiseAdd(bit_width=act_bits["residual_add"], input_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.out_act = _quant_block_act(channels, act_bits["out_act"], negative_slope, trainable_slope=trainable_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.conv(x))
        return self.out_act(self.residual_add(x, out))


class LayerQuantDownsamplingBottleneck(nn.Module):
    """Per-site clone of QuantENet.QuantDownsamplingBottleneck. Weight
    sites: "conv.0", "expand.0" (always), plus "reduce.0" (use_strided=True)
    OR "reduce.1" (use_strided=False -- a real index-shift gotcha, not just a
    cosmetic difference: reduce's own Sequential gains a leading
    weight-less MaxPool2d when use_strided=False, shifting every subsequent
    index by one). Act sites: "conv.2", "residual_add", "out_act" (always),
    plus "reduce.2" (use_strided=True) OR "reduce.3" (use_strided=False)."""

    def __init__(
        self, in_channels: int, out_channels: int, weight_bits: dict[str, int], act_bits: dict[str, int],
        internal_ratio: int = 4, dropout_p: float = 0.01, use_strided: bool = True,
        negative_slope: float | None = None, trainable_slope: bool = False, internal_bit_width: int | None = None,
        fused_leaky: bool = False, alpha_bit_width: int = 8, out_bit_width: int | None = None,
    ):
        super().__init__()
        internal_channels = max(1, out_channels // internal_ratio)

        def _act(ch: int, key: str) -> nn.Module:
            return _quant_block_act(
                ch, act_bits[key], negative_slope, trainable_slope=trainable_slope,
                internal_bit_width=internal_bit_width, fused_leaky=fused_leaky,
                alpha_bit_width=alpha_bit_width, out_bit_width=out_bit_width,
            )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        if use_strided:
            self.reduce = nn.Sequential(
                _quant_conv2d(in_channels, internal_channels, weight_bits["reduce.0"], kernel_size=2, stride=2),
                nn.BatchNorm2d(internal_channels), _act(internal_channels, "reduce.2"),
            )
        else:
            self.reduce = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                _quant_conv2d(in_channels, internal_channels, weight_bits["reduce.1"], kernel_size=1),
                nn.BatchNorm2d(internal_channels), _act(internal_channels, "reduce.3"),
            )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_channels, internal_channels, weight_bits["conv.0"], kernel_size=3, padding=1),
            nn.BatchNorm2d(internal_channels), _act(internal_channels, "conv.2"),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_channels, out_channels, weight_bits["expand.0"], kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.residual_add = qnn.QuantEltwiseAdd(bit_width=act_bits["residual_add"], input_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.out_act = _act(out_channels, "out_act")
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
        input_size = x.size()
        main, indices = self.pool(x)
        if main.shape[1] < self.out_channels:
            padding = torch.zeros(
                main.shape[0], self.out_channels - main.shape[1], main.shape[2], main.shape[3],
                dtype=main.dtype, device=main.device,
            )
            main = torch.cat([main, padding], dim=1)
        elif main.shape[1] > self.out_channels:
            main = main[:, : self.out_channels]

        out = self.reduce(x)
        out = self.conv(out)
        out = self.dropout(self.expand(out))
        return self.out_act(self.residual_add(main, out)), indices, input_size


class LayerQuantUpsamplingBottleneck(nn.Module):
    """Per-site clone of QuantENet.QuantUpsamplingBottleneck (no
    negative_slope support at all in the original class -- kept that way).
    Weight sites: "main_proj.0", "reduce.0", "up.0", "expand.0" (always 4).
    Act sites: "reduce.2", "up.2", "residual_add", "out_act" (always 4)."""

    def __init__(self, in_channels: int, out_channels: int, weight_bits: dict[str, int], act_bits: dict[str, int], internal_ratio: int = 4):
        super().__init__()
        internal_channels = max(1, in_channels // internal_ratio)
        self.main_proj = nn.Sequential(
            _quant_conv2d(in_channels, out_channels, weight_bits["main_proj.0"], kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.reduce = nn.Sequential(
            _quant_conv2d(in_channels, internal_channels, weight_bits["reduce.0"], kernel_size=1),
            nn.BatchNorm2d(internal_channels), _quant_act(act_bits["reduce.2"]),
        )
        self.up = nn.Sequential(
            qnn.QuantConvTranspose2d(internal_channels, internal_channels, kernel_size=2, stride=2, bias=False,
                                      weight_bit_width=weight_bits["up.0"], weight_quant=Int8WeightPerTensorFloat),
            nn.BatchNorm2d(internal_channels), _quant_act(act_bits["up.2"]),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_channels, out_channels, weight_bits["expand.0"], kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=0.1)
        self.residual_add = qnn.QuantEltwiseAdd(bit_width=act_bits["residual_add"], input_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.out_act = _quant_act(act_bits["out_act"])

    def forward(self, x: torch.Tensor, output_size: torch.Size, indices: torch.Tensor | None = None) -> torch.Tensor:
        main = self.main_proj(x)
        if indices is None:
            main = F.interpolate(main, size=output_size[2:], mode="bilinear", align_corners=False)
        else:
            main = self.unpool(main, indices, output_size=output_size)
        out = self.reduce(x)
        out = self.up(out)
        out = self.dropout(self.expand(out))
        return self.out_act(self.residual_add(main, out))


def _make_layer_shallow_stage(
    channels: int, n_ops: int, layer_weight_bits: dict[str, int], layer_act_bits: dict[str, int], dropout_p: float,
    name_prefix: str, slope_map: dict[str, float], use_dsc: bool = False, dsc_no_projection: bool = False,
    dsc_no_projection_context_only: bool = False, trainable_slope: bool = True,
    discovery_placeholder: int | None = None,
) -> nn.Sequential:
    """Per-site generalization of CombinedQuantENet._make_block_shallow_stage
    -- looks up a LOCAL per-site dict per loop index via _local(...) instead
    of a single scalar (w, a) pair."""
    slope_map = slope_map or {}
    if dsc_no_projection and not dsc_no_projection_context_only:
        return nn.Sequential(*[
            LayerQuantDSCNoProjectionBottleneck(
                channels, _local(layer_weight_bits, name_prefix, i, discovery_placeholder),
                _local(layer_act_bits, name_prefix, i, discovery_placeholder),
                dropout_p=dropout_p, negative_slope=slope_map.get(f"{name_prefix}.{i}"),
                trainable_slope=trainable_slope,
            )
            for i in range(n_ops)
        ])
    return nn.Sequential(*[
        LayerQuantRegularBottleneck(
            channels, _local(layer_weight_bits, name_prefix, i, discovery_placeholder),
            _local(layer_act_bits, name_prefix, i, discovery_placeholder),
            dropout_p=dropout_p, use_dsc=use_dsc, negative_slope=slope_map.get(f"{name_prefix}.{i}"),
            trainable_slope=trainable_slope,
        )
        for i in range(n_ops)
    ])


def _make_layer_context_stage(
    channels: int, n_ops: int, layer_weight_bits: dict[str, int], layer_act_bits: dict[str, int],
    name_prefix: str, slope_map: dict[str, float], context_pattern: str, use_dilated: bool = True,
    use_asymmetric: bool = False, use_dsc: bool = False, dsc_no_projection: bool = False,
    separable_dilated: bool = True, trainable_slope: bool = True, discovery_placeholder: int | None = None,
) -> nn.Sequential:
    """Per-site generalization of CombinedQuantENet._make_block_context_stage
    -- same dilation-pattern selection / dsc_no_projection bookend handling /
    asymmetric guard, looking up a LOCAL per-site dict per loop index instead
    of a single scalar (w, a) pair."""
    if context_pattern not in VALID_CONTEXT_PATTERNS:
        raise ValueError(f"context_pattern must be one of {VALID_CONTEXT_PATTERNS}, got {context_pattern!r}.")
    if context_pattern == "dense_dilation":
        pattern = DENSE_DILATION_PATTERN
    elif context_pattern == "dense_dilation_reg_interleaved":
        pattern = DENSE_DILATION_REG_INTERLEAVED_PATTERN
    elif context_pattern == "dense_dilation_reg_interleaved_double_mid":
        pattern = DENSE_DILATION_REG_INTERLEAVED_DOUBLE_MID_PATTERN
    elif context_pattern == "dense_dilation_d2_projected":
        pattern = DENSE_DILATION_D2_PROJECTED_PATTERN
    elif context_pattern == "dense_dilation_d8_d16_projected":
        pattern = DENSE_DILATION_D8_D16_PROJECTED_PATTERN
    elif context_pattern == "dense_dilation_d2_regular":
        pattern = DENSE_DILATION_D2_REGULAR_PATTERN
    elif context_pattern == "dense_dilation_reg_trailing":
        pattern = DENSE_DILATION_REG_TRAILING_PATTERN
    else:
        pattern = CONTEXT_STAGE_PATTERN

    slope_map = slope_map or {}

    if dsc_no_projection:
        ops = []
        for i in range(n_ops):
            kwargs = dict(pattern[i % len(pattern)])
            block_name = f"{name_prefix}.{i}"
            w_bits = _local(layer_weight_bits, name_prefix, i, discovery_placeholder)
            a_bits = _local(layer_act_bits, name_prefix, i, discovery_placeholder)
            if kwargs.get("reg_bottleneck", False):
                ops.append(LayerQuantRegularBottleneck(
                    channels, w_bits, a_bits, dropout_p=0.1, use_dsc=False,
                    negative_slope=slope_map.get(block_name), trainable_slope=trainable_slope,
                    dilation=kwargs.get("dilation", 1), padding=kwargs.get("padding", 1),
                ))
                continue
            if kwargs.get("asymmetric", False):
                if use_asymmetric:
                    raise ValueError("dsc_no_projection is not defined for asymmetric bottlenecks -- set use_asymmetric=False.")
                kwargs = {}
            dilation = kwargs.get("dilation", 1)
            if dilation != 1 and not use_dilated:
                dilation = 1
            ops.append(LayerQuantDSCNoProjectionBottleneck(
                channels, w_bits, a_bits, kernel_size=3, padding=dilation, dilation=dilation,
                dropout_p=0.1, negative_slope=slope_map.get(block_name), trainable_slope=trainable_slope,
            ))
        return nn.Sequential(*ops)

    ops = []
    for i in range(n_ops):
        kwargs = dict(pattern[i % len(pattern)])
        is_reg_bookend = kwargs.pop("reg_bottleneck", False)
        use_dsc_here = False if is_reg_bookend else use_dsc
        if kwargs.get("dilation", 1) != 1 and not use_dilated:
            kwargs = {}
        if kwargs.get("asymmetric", False) and not use_asymmetric:
            kwargs = {}
        block_name = f"{name_prefix}.{i}"
        ops.append(LayerQuantRegularBottleneck(
            channels, _local(layer_weight_bits, name_prefix, i, discovery_placeholder),
            _local(layer_act_bits, name_prefix, i, discovery_placeholder),
            dropout_p=0.1, use_dsc=use_dsc_here, separable_dilated=separable_dilated,
            negative_slope=slope_map.get(block_name), trainable_slope=trainable_slope, **kwargs,
        ))
    return nn.Sequential(*ops)


def _build_layer_enet_modules(
    layer_weight_bits: dict[str, int], layer_act_bits: dict[str, int], *, in_channels: int, out_channels: int,
    channels: tuple[int, int, int, int, int], bottlenecks_per_stage: tuple[int, int, int, int, int],
    context_pattern: str, use_dilated: bool, use_asymmetric: bool, use_strided: bool, use_dsc: bool,
    dsc_no_projection: bool, dsc_no_projection_context_only: bool, separable_dilated: bool,
    leaky_slope_map: dict[str, float] | None, trainable_slope: bool,
    discovery_placeholder: int | None = None,
) -> dict[str, nn.Module]:
    """Single source of truth for block assembly -- verbatim port of
    CombinedQuantENet.__init__'s body (11 top-level submodule constructions),
    generalized only in the per-site bit lookups. Returns {"initial": ...,
    "down1": ..., "regular1": ..., "down2": ..., "stage2": ..., "stage3": ...,
    "up4": ..., "regular4": ..., "up5": ..., "regular5": ...} -- NOT "final"
    (built separately by both callers: LayerQuantENet.__init__ and
    layer_names_for, neither needs a per-loop-index sub-dict for it). Used
    with discovery_placeholder=None for the real build (a genuinely missing
    key still raises KeyError as a correctness backstop even though
    LayerQuantENet's own validation should already have caught it) and
    discovery_placeholder=8 for layer_names_for's throwaway discovery build
    -- factored out specifically so the two can never drift the way a
    hand-duplicated name predictor could."""
    slope_map = leaky_slope_map or {}
    initial_ch, stage1_ch, stage23_ch, stage4_ch, stage5_ch = channels
    n_stage1, n_stage2, n_stage3, n_regular4, n_regular5 = bottlenecks_per_stage

    modules: dict[str, nn.Module] = {}

    modules["initial"] = LayerQuantInitialBlock(
        in_channels, initial_ch, _local_single(layer_weight_bits, "initial", discovery_placeholder),
        _local_single(layer_act_bits, "initial", discovery_placeholder),
        negative_slope=slope_map.get("initial"), trainable_slope=trainable_slope,
    )

    modules["down1"] = LayerQuantDownsamplingBottleneck(
        initial_ch, stage1_ch, _local_single(layer_weight_bits, "down1", discovery_placeholder),
        _local_single(layer_act_bits, "down1", discovery_placeholder),
        dropout_p=0.01, use_strided=use_strided, negative_slope=slope_map.get("down1"), trainable_slope=trainable_slope,
    )
    modules["regular1"] = _make_layer_shallow_stage(
        stage1_ch, n_stage1, layer_weight_bits, layer_act_bits, 0.01, "regular1", slope_map,
        use_dsc=use_dsc, dsc_no_projection=dsc_no_projection,
        dsc_no_projection_context_only=dsc_no_projection_context_only, trainable_slope=trainable_slope,
        discovery_placeholder=discovery_placeholder,
    )

    modules["down2"] = LayerQuantDownsamplingBottleneck(
        stage1_ch, stage23_ch, _local_single(layer_weight_bits, "down2", discovery_placeholder),
        _local_single(layer_act_bits, "down2", discovery_placeholder),
        dropout_p=0.1, use_strided=use_strided, negative_slope=slope_map.get("down2"), trainable_slope=trainable_slope,
    )
    modules["stage2"] = _make_layer_context_stage(
        stage23_ch, n_stage2, layer_weight_bits, layer_act_bits, "stage2", slope_map, context_pattern,
        use_dilated=use_dilated, use_asymmetric=use_asymmetric, use_dsc=use_dsc,
        dsc_no_projection=dsc_no_projection, separable_dilated=separable_dilated, trainable_slope=trainable_slope,
        discovery_placeholder=discovery_placeholder,
    )
    modules["stage3"] = _make_layer_context_stage(
        stage23_ch, n_stage3, layer_weight_bits, layer_act_bits, "stage3", slope_map, context_pattern,
        use_dilated=use_dilated, use_asymmetric=use_asymmetric, use_dsc=use_dsc,
        dsc_no_projection=dsc_no_projection, separable_dilated=separable_dilated, trainable_slope=trainable_slope,
        discovery_placeholder=discovery_placeholder,
    )

    # Decoder (regular4/regular5/up4/up5) is always plain QuantReLU,
    # regardless of leaky_slope_map -- same rule CombinedQuantENet/ENet.py
    # already use (decoder hardcodes relu=True, prelu_variant is
    # encoder/context-only). slope_map={} passed below, matching
    # CombinedQuantENet's own regular4/regular5 calls exactly (no explicit
    # trainable_slope either -- irrelevant with no negative_slope to use it).
    modules["up4"] = LayerQuantUpsamplingBottleneck(
        stage23_ch, stage4_ch, _local_single(layer_weight_bits, "up4", discovery_placeholder),
        _local_single(layer_act_bits, "up4", discovery_placeholder),
    )
    modules["regular4"] = _make_layer_shallow_stage(
        stage4_ch, n_regular4, layer_weight_bits, layer_act_bits, 0.1, "regular4", {},
        use_dsc=use_dsc, dsc_no_projection=dsc_no_projection,
        dsc_no_projection_context_only=dsc_no_projection_context_only,
        discovery_placeholder=discovery_placeholder,
    )

    modules["up5"] = LayerQuantUpsamplingBottleneck(
        stage4_ch, stage5_ch, _local_single(layer_weight_bits, "up5", discovery_placeholder),
        _local_single(layer_act_bits, "up5", discovery_placeholder),
    )
    modules["regular5"] = _make_layer_shallow_stage(
        stage5_ch, n_regular5, layer_weight_bits, layer_act_bits, 0.1, "regular5", {},
        use_dsc=use_dsc, dsc_no_projection=dsc_no_projection,
        dsc_no_projection_context_only=dsc_no_projection_context_only,
        discovery_placeholder=discovery_placeholder,
    )

    return modules


def layer_names_for(
    *, in_channels: int = 1, out_channels: int, channels: tuple[int, int, int, int, int],
    bottlenecks_per_stage: tuple[int, int, int, int, int], context_pattern: str,
    use_dilated: bool = True, use_asymmetric: bool = False, use_strided: bool = True,
    use_dsc: bool = False, dsc_no_projection: bool = False, dsc_no_projection_context_only: bool = False,
    separable_dilated: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Returns (layer_weight_names, layer_act_names) -- the exact set of
    per-site dict keys a LayerQuantENet built with this architecture shape
    requires, discovered by constructing a throwaway uniformly-8-bit module
    tree and introspecting it (see module docstring for why: site names are a
    pure function of shape/pattern flags, never of the bit-width chosen at a
    site, so this can never drift out of sync with the real assembly code).
    Signature is LayerQuantENet.__init__'s architecture-shape subset only --
    excludes decoder_type/prelu_variant/leaky_slope_map/trainable_slope, none
    of which affect which sites exist."""
    modules = _build_layer_enet_modules(
        {}, {}, in_channels=in_channels, out_channels=out_channels, channels=channels,
        bottlenecks_per_stage=bottlenecks_per_stage, context_pattern=context_pattern,
        use_dilated=use_dilated, use_asymmetric=use_asymmetric, use_strided=use_strided,
        use_dsc=use_dsc, dsc_no_projection=dsc_no_projection,
        dsc_no_projection_context_only=dsc_no_projection_context_only,
        separable_dilated=separable_dilated, leaky_slope_map=None, trainable_slope=False,
        discovery_placeholder=8,
    )
    dummy = nn.Module()
    for name, module in modules.items():
        setattr(dummy, name, module)
    dummy.final = qnn.QuantConvTranspose2d(
        channels[4], out_channels, kernel_size=2, stride=2, bias=True,
        weight_bit_width=8, weight_quant=Int8WeightPerTensorFloat,
    )
    weight_names = tuple(
        name for name, m in dummy.named_modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))
    )
    act_names = tuple(_first_match_modules(dummy, ACT_SITE_TYPES).keys())
    return weight_names, act_names


def expand_uniform_layer_bits(
    weight_bit_width: int, act_bit_width: int, layer_weight_names: tuple[str, ...], layer_act_names: tuple[str, ...],
) -> tuple[dict[str, int], dict[str, int]]:
    """Broadcasts one (w, a) pair to every layer site -- e.g. for a quick
    uniform-bit smoke test or a parity check against CombinedQuantENet/
    QuantENet at equal bit-width."""
    return (
        {n: weight_bit_width for n in layer_weight_names},
        {n: act_bit_width for n in layer_act_names},
    )


def expand_block_bits_to_layer_bits(
    block_weight_bits: dict[str, int], block_act_bits: dict[str, int],
    layer_weight_names: tuple[str, ...], layer_act_names: tuple[str, ...],
) -> tuple[dict[str, int], dict[str, int]]:
    """Broadcasts an EXISTING per-block dict (e.g. a real block_bits_*.json,
    CombinedQuantENet's own schema) down to every layer inside each block,
    uniformly -- a coarse fallback/warm-start for LayerQuantENet before a
    real per-layer ILP result exists, or a parity check against
    CombinedQuantENet built from the same block_bits_*.json. A layer name's
    owning block is derived by stripping its trailing site suffix: names of
    the form "<attr>.<i>.<site...>" (attr in {regular1, regular4, regular5,
    stage2, stage3}) belong to block "<attr>.<i>"; names of the form
    "<attr>.<site...>" (attr in {initial, down1, down2, up4, up5}) belong to
    block "<attr>"; "final" belongs to block "final". Raises KeyError via the
    normal dict lookup if block_weight_bits/block_act_bits is missing an
    owning block's entry -- not re-validated here beyond that."""
    multi_instance_attrs = ("regular1", "regular4", "regular5", "stage2", "stage3")

    def _owning_block(layer_name: str) -> str:
        if layer_name == "final":
            return "final"
        parts = layer_name.split(".")
        return f"{parts[0]}.{parts[1]}" if parts[0] in multi_instance_attrs else parts[0]

    return (
        {n: block_weight_bits[_owning_block(n)] for n in layer_weight_names},
        {n: block_act_bits[_owning_block(n)] for n in layer_act_names},
    )


class LayerQuantENet(nn.Module):
    """Per-site-parametrized replacement for CombinedQuantENet -- see module
    docstring for scope/naming/rationale. prelu_variant is stored purely as a
    provenance/documentation field, same as CombinedQuantENet's own."""

    def __init__(
        self, layer_weight_bits: dict[str, int], layer_act_bits: dict[str, int], *,
        in_channels: int = 1, out_channels: int, channels: tuple[int, int, int, int, int],
        bottlenecks_per_stage: tuple[int, int, int, int, int], context_pattern: str,
        decoder_type: str = "upsample_conv", use_dilated: bool = True, use_asymmetric: bool = False,
        use_strided: bool = True, use_dsc: bool = False, dsc_no_projection: bool = False,
        dsc_no_projection_context_only: bool = False, separable_dilated: bool = True,
        prelu_variant: str = "standard", leaky_slope_map: dict[str, float] | None = None,
        trainable_slope: bool = True,
    ):
        super().__init__()
        if decoder_type != "upsample_conv":
            raise NotImplementedError(
                "LayerQuantENet's forward() only implements the upsample_conv decoder path "
                "(no pooling-indices plumbing) -- same scope CombinedQuantENet already has."
            )
        if dsc_no_projection_context_only and not dsc_no_projection:
            raise ValueError("dsc_no_projection_context_only narrows dsc_no_projection's scope -- meaningless without dsc_no_projection=True itself.")
        if len(channels) != 5 or len(bottlenecks_per_stage) != 5:
            raise ValueError("channels and bottlenecks_per_stage must each have 5 values (see ENet.py).")

        self.prelu_variant = prelu_variant

        shape_kwargs = dict(
            in_channels=in_channels, out_channels=out_channels, channels=channels,
            bottlenecks_per_stage=bottlenecks_per_stage, context_pattern=context_pattern,
            use_dilated=use_dilated, use_asymmetric=use_asymmetric, use_strided=use_strided,
            use_dsc=use_dsc, dsc_no_projection=dsc_no_projection,
            dsc_no_projection_context_only=dsc_no_projection_context_only, separable_dilated=separable_dilated,
        )
        expected_weight_names, expected_act_names = layer_names_for(**shape_kwargs)

        missing_w = [n for n in expected_weight_names if n not in layer_weight_bits]
        missing_a = [n for n in expected_act_names if n not in layer_act_bits]
        if missing_w or missing_a:
            raise ValueError(
                f"layer_weight_bits/layer_act_bits must have one entry per expected layer site "
                f"({len(expected_weight_names)} weight, {len(expected_act_names)} act) -- "
                f"missing weight keys: {missing_w}, missing act keys: {missing_a}."
            )
        # Deliberately stricter than CombinedQuantENet's own validation: a
        # per-layer dict has 150-300+ keys where a typo (e.g. a stale/mistyped
        # layer_bits_*.json) is much easier to miss than in CombinedQuantENet's
        # ~20-30 block keys, so also reject unrecognized extra keys.
        extra_w = [k for k in layer_weight_bits if k not in expected_weight_names]
        extra_a = [k for k in layer_act_bits if k not in expected_act_names]
        if extra_w or extra_a:
            raise ValueError(
                f"layer_weight_bits/layer_act_bits contain keys that are not real sites in this "
                f"architecture -- extra weight keys: {extra_w}, extra act keys: {extra_a}."
            )

        self.layer_weight_bits = dict(layer_weight_bits)
        self.layer_act_bits = dict(layer_act_bits)

        modules = _build_layer_enet_modules(
            layer_weight_bits, layer_act_bits, leaky_slope_map=leaky_slope_map, trainable_slope=trainable_slope,
            discovery_placeholder=None, **shape_kwargs,
        )
        for name, module in modules.items():
            setattr(self, name, module)

        self.final = qnn.QuantConvTranspose2d(
            channels[4], out_channels, kernel_size=2, stride=2, bias=True,
            weight_bit_width=layer_weight_bits["final"], weight_quant=Int8WeightPerTensorFloat,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        x = self.initial(x)
        x, _indices1, size1 = self.down1(x)
        x = self.regular1(x)
        x, _indices2, size2 = self.down2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.up4(x, size2, None)
        x = self.regular4(x)
        x = self.up5(x, size1, None)
        x = self.regular5(x)
        x = self.final(x)
        if hasattr(x, "value"):
            x = x.value
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x

    @classmethod
    def from_pretrained(
        cls, checkpoint_path: str | Path, layer_weight_bits: dict[str, int], layer_act_bits: dict[str, int], *,
        leaky_slope_map: dict[str, float] | None = None, trainable_slope: bool = True, **kwargs,
    ) -> "LayerQuantENet":
        """Same strict=False name+shape transfer CombinedQuantENet's own
        from_pretrained already uses -- state-dict key/shape matching depends
        only on module-tree structure and tensor shapes, neither of which
        vary with bit-width choice (this repo's quantizers are per-TENSOR,
        not per-channel, so scale-buffer shape never depends on bit-width
        either). **kwargs forwards every other __init__ arg."""
        model = cls(
            layer_weight_bits, layer_act_bits, leaky_slope_map=leaky_slope_map,
            trainable_slope=trainable_slope, **kwargs,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source_state_dict = checkpoint["network_weights"]
        model_state_dict = model.state_dict()
        transferable = {
            key: value for key, value in source_state_dict.items()
            if key in model_state_dict and model_state_dict[key].shape == value.shape
        }
        missing, unexpected = model.load_state_dict(transferable, strict=False)
        assert not unexpected, f"unexpected keys after strict=False load (should be impossible): {unexpected}"
        n_shape_mismatch = sum(
            1 for key, value in source_state_dict.items()
            if key in model_state_dict and model_state_dict[key].shape != value.shape
        )
        print(
            f"LayerQuantENet.from_pretrained({checkpoint_path}): transferred {len(transferable)}/"
            f"{len(model_state_dict)} model keys ({n_shape_mismatch} shape mismatches, "
            f"{len(missing)} left uninitialized -- expected for Brevitas-only quantizer params)."
        )
        return model


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    _REPO_ROOT = _Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(_REPO_ROOT / "compression" / "hawq"))

    torch.manual_seed(0)

    S12_SHAPE = dict(
        out_channels=5, channels=(4, 16, 32, 16, 4), bottlenecks_per_stage=(4, 8, 8, 2, 1),
        context_pattern="dense_dilation", use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=True,
    )

    # A. Step 0 -- Brevitas quant-conv IS-A plain conv (gates the naming-parity claim below).
    assert isinstance(qnn.QuantConv2d(1, 1, 1), nn.Conv2d)
    assert isinstance(qnn.QuantConvTranspose2d(1, 1, 2, stride=2), nn.ConvTranspose2d)
    print("A. Step 0 isinstance checks: OK")

    # B. layer_names_for at S12's exact shape -- spot-check expected keys.
    weight_names, act_names = layer_names_for(**S12_SHAPE)
    assert len(weight_names) == 101, f"expected 101 weight sites for S12, got {len(weight_names)}"
    assert "down1.reduce.0" in weight_names and "down1.conv.0" in weight_names and "down1.expand.0" in weight_names
    assert "regular1.0.conv" in weight_names, "plain (non-split) conv site should be bare 'conv', not 'conv.0'"
    assert "stage2.1.conv.0" in weight_names and "stage2.1.conv.3" in weight_names, "dilated slot should split into conv.0/conv.3"
    assert "stage2.1.conv.2" in act_names, "dilated slot should have an extra intermediate act at conv.2"
    assert "stage2.0.conv" not in weight_names, "stage2.0 (dilation=1 bookend) should be a bare, non-split conv"
    assert "final" in weight_names and "final" not in act_names
    print(f"B. layer_names_for: OK ({len(weight_names)} weight sites, {len(act_names)} act sites)")

    # C. Cross-tool naming-parity assertion vs. block_utils.block_weight_targets on a plain FP32 ENet.
    from block_utils import block_weight_targets, enumerate_blocks  # noqa: E402
    from nnunetv2.nets.ENet import ENet  # noqa: E402

    fp32 = ENet(
        in_channels=1, out_channels=5, channels=(4, 16, 32, 16, 4), bottlenecks_per_stage=(4, 8, 8, 2, 1),
        decoder_type="upsample_conv", use_asymmetric=False, context_pattern="dense_dilation", separable_dilated=True,
    )
    fp32_blocks = enumerate_blocks(fp32)
    fp32_weight_keys = {
        name for block in block_weight_targets(fp32_blocks).values() for name in block
    }
    assert fp32_weight_keys == set(weight_names), (
        f"naming parity FAILED -- only in layer_names_for: {set(weight_names) - fp32_weight_keys}; "
        f"only in block_weight_targets: {fp32_weight_keys - set(weight_names)}"
    )
    print("C. Cross-tool naming parity vs. block_utils.block_weight_targets: OK (exact key-set match)")

    # D. Construct/forward at uniform 8-bit.
    layer_weight_bits, layer_act_bits = expand_uniform_layer_bits(8, 8, weight_names, act_names)
    model = LayerQuantENet(layer_weight_bits, layer_act_bits, decoder_type="upsample_conv", **S12_SHAPE).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 512, 512))
    out_t = out.value if hasattr(out, "value") else out
    assert out_t.shape == (1, 5, 512, 512), f"got {tuple(out_t.shape)}"
    print(f"D. Construct+forward at uniform 8-bit: OK, output shape {tuple(out_t.shape)}")

    # E. Structural parity vs. CombinedQuantENet at equal uniform 8-bit -- same total parameter count.
    from nnunetv2.nets.CombinedQuantENet import CombinedQuantENet, block_names_for, expand_uniform_bits  # noqa: E402

    block_names = block_names_for(S12_SHAPE["bottlenecks_per_stage"])
    block_w, block_a = expand_uniform_bits(8, 8, block_names)
    combined = CombinedQuantENet(block_w, block_a, decoder_type="upsample_conv", **S12_SHAPE)
    n_layerwise = sum(p.numel() for p in model.parameters())
    n_combined = sum(p.numel() for p in combined.parameters())
    assert n_layerwise == n_combined, f"parameter count mismatch: LayerQuantENet={n_layerwise} CombinedQuantENet={n_combined}"
    print(f"E. Structural parity vs. CombinedQuantENet: OK ({n_layerwise} params, both)")

    # F. Validation-error tests: one missing key, one bogus extra key.
    bad_missing = dict(layer_weight_bits)
    del bad_missing["down1.conv.0"]
    try:
        LayerQuantENet(bad_missing, layer_act_bits, decoder_type="upsample_conv", **S12_SHAPE)
        raise AssertionError("expected ValueError for a missing weight key, got none")
    except ValueError as e:
        assert "down1.conv.0" in str(e)
    bad_extra = dict(layer_weight_bits)
    bad_extra["bogus.site.0"] = 8
    try:
        LayerQuantENet(bad_extra, layer_act_bits, decoder_type="upsample_conv", **S12_SHAPE)
        raise AssertionError("expected ValueError for a bogus extra weight key, got none")
    except ValueError as e:
        assert "bogus.site.0" in str(e)
    print("F. Validation-error tests (missing key / bogus extra key): OK")

    # G. from_pretrained sanity -- transfer from a synthetic FP32 ENet checkpoint.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        torch.save({"network_weights": fp32.state_dict()}, tmp.name)
        pretrained = LayerQuantENet.from_pretrained(
            tmp.name, layer_weight_bits, layer_act_bits, decoder_type="upsample_conv", **S12_SHAPE,
        )
    assert torch.allclose(pretrained.down1.reduce[0].weight, fp32.down1.reduce[0].weight)
    print("G. from_pretrained sanity: OK (conv weights transferred elementwise-identical)")

    # H. Per-branch site-count check on LayerQuantRegularBottleneck directly.
    plain = LayerQuantRegularBottleneck(16, {"reduce.0": 8, "conv": 8, "expand.0": 8},
                                         {"reduce.2": 8, "conv_bn_act.2": 8, "out_act": 8, "residual_add": 8})
    dsc = LayerQuantRegularBottleneck(16, {"reduce.0": 8, "conv.0": 8, "conv.1": 8, "expand.0": 8},
                                       {"reduce.2": 8, "conv_bn_act.2": 8, "out_act": 8, "residual_add": 8}, use_dsc=True)
    split = LayerQuantRegularBottleneck(16, {"reduce.0": 8, "conv.0": 8, "conv.3": 8, "expand.0": 8},
                                         {"reduce.2": 8, "conv.2": 8, "conv_bn_act.2": 8, "out_act": 8, "residual_add": 8},
                                         separable_dilated=True, dilation=2)
    for name, block, exp_w, exp_a in [("plain", plain, 3, 4), ("use_dsc", dsc, 4, 4), ("separable_dilated d=2", split, 4, 5)]:
        n_w = sum(1 for _, m in block.named_modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)))
        n_a = len(_first_match_modules(block, ACT_SITE_TYPES))
        assert n_w == exp_w, f"{name}: expected {exp_w} weight sites, got {n_w}"
        assert n_a == exp_a, f"{name}: expected {exp_a} act sites, got {n_a}"
    print("H. Per-branch site-count check (plain 3/4, use_dsc 4/4, separable_dilated d=2 4/5): OK")

    print("\nAll LayerQuantENet self-tests PASSED.")

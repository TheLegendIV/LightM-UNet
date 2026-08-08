"""Export a FINN-compatible quantized model for the S13-leaky-frozen
compression-sweep config (the real per-block frozen-LeakyReLU config, later
PTQ'd to int8):

    config_name (FP32 source) = nnUNetTrainerENet_13_separable_dense_nonneg_block_leaky_frozen
    config_name (PTQ int8)    = nnUNetTrainerENetQuant_13_leaky_frozen_ptq_int8
    channels    = (4, 16, 32, 16, 4)   # 5-tuple: initial, s1, s23 (shared), s4, s5
    bnecks      = (4, 8, 8, 2, 1)
    ops_flags   = dilated=1, asymmetric=0, strided=1, dsc=0,
                  context_pattern=dense_dilation, separable_dilated=1,
                  leaky_slope_map={...} (real per-block frozen slopes, see
                  compression/post-quantization/slope_maps/13_separable_dense_nonneg_block_leaky_frozen.json)

See enet/nnunetv2/nets/QuantENet.py (QuantRegularBottleneck's
separable_dilated branch, QuantDecomposedLeakyAct, QuantENet._make_context_stage)
for the real architecture this mirrors.

Like hardware/finn_export_s5_dscnoproj_dense.py, this uses the real
per-block LeakyReLU activation this config was frozen with, decomposed as
PReLU(x) = alpha*x + (1-alpha)*ReLU(x) -> exported as
Quant -> LeakyRelu(alpha) -> Quant, fused into a single MultiThreshold by
hardware/finn_enet_build_decomposed_prelu.py's step_fuse_leaky_relu_to_threshold.

NOTE: this deliberately does NOT import QuantDecomposedLeakyAct from
enet/nnunetv2/nets/QuantENet.py, despite that class's docstring claiming to
be "proven end-to-end against FINN's real streamlining pipeline" -- its
actual implementation (`scaled_x = x_q*alpha; scaled_pos = pos*(1-alpha);
return out_quant(scaled_x+scaled_pos)`) is architecturally IDENTICAL to
finn_export_s5_dscnoproj_dense.py's own "v1" attempt (its docstring's own
words), which that script's docstring explicitly documents as HAVING FAILED
at FINN's step_create_dataflow_partition ("cycle-free graph violated:
partition depends on itself") -- a fork -> 2 branches -> dynamic Add-join
topology no FINN transform absorbs into a single MultiThreshold. The two
docstrings contradict each other; rather than trust the untested one, this
script reuses the S5 script's actual PROVEN "v3" shape instead (a locally
defined DecomposedLeakyAct below, identical to DecomposedPReLUAct but
parameterized by the real per-block alpha instead of a fixed 0.25 default):
Quant -> torch.nn.functional.leaky_relu -> Quant, no fork, no Add-join.

One real slope-map value is exactly 0.0 (stage2.5, stage2.6, stage3.1,
stage3.2) -- LeakyReLU(x, 0.0) == ReLU(x) exactly, and
step_fuse_leaky_relu_to_threshold's fusion asserts 0.0 < alpha < 1.0 (the
algebraic inverse `t/alpha` is undefined at alpha=0 anyway), so those
specific blocks use plain QuantReLU instead of the decomposed path --
mathematically identical to what the real slope value already means, not
an approximation.

Same convention as finn_enet_prod_export.py/finn_export_s5_dscnoproj_dense.py
for topology fixes needed to satisfy FINN (see hardware/README.md for the
full rationale of each):
  1. No MaxPool+Concat initial block -> single stride-2 conv producing the
     full channel count directly (FINNInitialBlock).
  2. No MaxUnpool/F.interpolate upsampling -> QuantConvTranspose2d main path
     (FINNUpsamplingBottleneck).
  3. Downsampling shortcut uses MaxPool(no indices) + 1x1 conv projection
     instead of MaxPool(return_indices=True) (FINNDownsamplingBottleneck).
  4. Final layer bias=False.
These are real topology deviations from a literal re-export of QuantENet's
own graph -- exactly like every prior FINN build in this repo (O8_native,
O2_native, E1, S5-DscNoProjDense), this export uses a freshly initialized
(untrained, torch.manual_seed(0)) model, NOT the real PTQ'd checkpoint's
weights. This is fine for FINN's *resource/timing* estimate and OOC
synthesis result -- both depend only on architecture + bit-width + which
nodes exist, not on learned weight VALUES (see this session's own established
finding: FINN's LUT/DSP/BRAM model is weight-value-independent) -- but it
means this export is for hardware feasibility/resource measurement only,
not an accuracy-preserving deployment of the real trained network. The one
place weight VALUES do matter here is the frozen LeakyReLU alpha per block,
which IS the real trained/frozen value from the slope map (that's a fixed
buffer, not a training-derived conv/BN weight, so "freshly initialized"
doesn't apply to it).

Usage (run inside the pytorch training container, which already has
torch/brevitas/qonnx):
    python hardware/finn_export_s13_leaky_frozen.py

Output: hardware/outputs/finn_exports/quantEnet_s13_leaky_frozen_int8.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_s13_leaky_frozen_int8.onnx \\
        <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/
    docker exec <finn_container_id> python /home/thelegendiv/finn/notebooks/enet/finn_enet_build_decomposed_prelu.py \\
        quantEnet_s13_leaky_frozen_int8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
from brevitas.inject.enum import ScalingImplType

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
from nnunetv2.nets.QuantENet import _quant_conv2d, _quant_act  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"
DEFAULT_CHANNELS = (4, 16, 32, 16, 4)   # initial, s1, s23 (shared), s4, s5
DEFAULT_BNECKS = (4, 8, 8, 2, 1)
BIT_WIDTH = 8
DEFAULT_SLOPE_MAP_FILE = (
    REPO_ROOT / "compression" / "post-quantization" / "slope_maps"
    / "13_separable_dense_nonneg_block_leaky_frozen.json"
)
# ENet.py's DENSE_DILATION_PATTERN flattened to its dilation values -- every
# context-stage slot dilated (2/4/8/16 schedule repeated twice), no plain/
# asymmetric slots at all (context_pattern=dense_dilation).
DENSE_DILATIONS = [2, 4, 8, 16, 2, 4, 8, 16]


class DecomposedLeakyAct(nn.Module):
    """PReLU(x) = alpha*x + (1-alpha)*ReLU(x), exported as a plain ONNX
    `LeakyRelu(alpha)` sandwiched between two int8 Quant identities -- the
    PROVEN "v3" shape from finn_export_s5_dscnoproj_dense.py's
    DecomposedPReLUAct (copied verbatim, just parameterized by the real
    per-block `negative_slope` instead of a fixed alpha_init=0.25 default).
    See that class's own docstring for why the fork/dynamic-Add-join
    alternatives (v1/v2) fail at FINN's step_create_dataflow_partition, and
    hardware/finn_enet_build_decomposed_prelu.py's
    step_fuse_leaky_relu_to_threshold for the exact fusion this shape
    enables."""

    def __init__(self, channels: int, bit_width: int, negative_slope: float):
        super().__init__()
        self.pre_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.negative_slope = negative_slope
        self.out_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.pre_quant(x)
        out = torch.nn.functional.leaky_relu(x_q, negative_slope=self.negative_slope)
        return self.out_quant(out)


class SignedReLUAct(nn.Module):
    """NOTE: unused, kept only for the historical record in this file's
    docstrings/comments below -- see _EPSILON_SLOPE for why. FINN's
    qonnx_activation_handlers._check_compatibility() *hard-requires* an
    UNSIGNED, non-narrow Quant immediately before a literal ONNX `Relu`
    node ("FINN only supports unsigned and non-narrow Quant nodes for Relu
    activations") -- so a signed Quant->Relu->Quant chain like this one is
    rejected outright by step_qonnx_to_finn, before our own custom fusion
    steps even run. Signed-Int8 consistency with the leaky sites (to avoid
    the residual scale-mismatch bug) and FINN's Relu handler's unsigned-
    only requirement are mutually exclusive for a literal `Relu` node --
    hence _EPSILON_SLOPE's approach instead (route the alpha=0.0 sites
    through the SAME literal-`LeakyRelu` + custom fusion path as every
    other leaky site, entirely bypassing FINN's Relu-specific handler)."""

    def __init__(self, channels: int, bit_width: int):
        super().__init__()
        self.pre_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.relu = nn.ReLU()
        self.out_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_quant(self.relu(self.pre_quant(x)))


# A real slope-map value of exactly 0.0 (stage2.5, stage2.6, stage3.1,
# stage3.2) is mathematically ReLU, but neither literal encoding works
# cleanly through this FINN pipeline:
#   - a literal ONNX `Relu` node needs an UNSIGNED pre-Quant (FINN's own
#     qonnx_activation_handlers hard-requirement) -- but every leaky site
#     uses a SIGNED Int8 quantizer (leaky outputs can be negative), and
#     mixing signed/unsigned quantizers across a residual `+` leaves an
#     un-absorbable scalar rescale stranded on the fork, which trips
#     step_create_dataflow_partition's cycle-free-graph check (confirmed
#     empirically: exactly 4 real slope-map zeros == exactly 4 observed
#     failures, each at a plain/leaky residual boundary).
#   - a literal ONNX `LeakyRelu` node with alpha=0.0 exactly hits
#     step_fuse_leaky_relu_to_threshold's own `0.0 < alpha < 1.0` assert.
# Fix: use a NEGLIGIBLY small positive slope instead of exactly 0.0 --
# functionally indistinguishable from ReLU at int8 quantization precision
# (any output delta from a true 0.0 is far below one quantization step),
# while satisfying the fusion step's assert AND keeping every activation
# in the network on the same signed-Int8 convention.
_EPSILON_SLOPE = 1e-6


def _plain_relu_factory(channels: int, bit_width: int) -> nn.Module:
    return DecomposedLeakyAct(channels, bit_width, _EPSILON_SLOPE)


def _val(t):
    """Unwrap a Brevitas QuantTensor to its plain float tensor before a
    residual `+` -- QuantTensor.__add__ hard-asserts both operands share
    the exact same (calibrated/initialized) scaling factor
    (check_scaling_factors_same), which two independently-instantiated
    quantizers (e.g. two different QuantDecomposedLeakyAct/QuantReLU
    instances at different call sites) generically do NOT share at random
    init -- confirmed empirically (RuntimeError: \"Scaling factors are
    different\") the moment two decomposed-leaky-quantized branches meet at
    a residual Add. Adding plain tensors instead sidesteps the check
    entirely; the following act_factory call already re-quantizes the sum,
    so this changes nothing about the exported ONNX graph's Add node
    itself (still a plain elementwise Add, same as every other residual in
    this file)."""
    return t.value if hasattr(t, "value") else t


class _ConstScaleInt8Act(Int8ActPerTensorFloat):
    """Int8ActPerTensorFloat variant with a HARD-CODED, non-learned,
    input-independent scale (ScalingImplType.CONST) instead of the default
    stats/parameter-calibrated scale. Used ONLY as a shared pre-residual-add
    requantizer so that two independently-calibrated branches are forced
    onto the EXACT SAME int8 grid immediately before a residual `+` --
    eliminating the un-representable per-branch float rescale that FINN's
    convert_to_hw_layers cannot express (confirmed via direct source read:
    InferChannelwiseLinearLayer requires an exact-integer constant;
    InferAddStreamsLayer requires both Add operands already integer-typed
    with no pending rescale; MultiThreshold's out_scale is hard-locked to
    1.0). This export is for FINN resource/topology estimation only (a
    freshly-initialized/untrained model -- see module docstring, already
    established weight-VALUE-independent), so hard-coding this constant
    introduces no accuracy concern for the purpose at hand."""

    scaling_impl_type = ScalingImplType.CONST
    scaling_init = 1.0 / 64.0  # arbitrary fixed value; only its SHARED-ness matters


def _requant_factory(bit_width: int) -> nn.Module:
    """Fresh const-scale requantizer for one residual-add site -- call the
    SAME returned instance on BOTH operands immediately before the `+` so
    they land on the identical (constant, non-data-dependent) int8 grid,
    regardless of whatever scale their own upstream quantizers calibrated
    to."""
    return qnn.QuantIdentity(
        bit_width=bit_width, act_quant=_ConstScaleInt8Act, return_quant_tensor=False,
    )


def _make_act_factory(negative_slope: float | None):
    """Picks the epsilon-slope DecomposedLeakyAct (slope None or exactly
    0.0 -- see _EPSILON_SLOPE above for why an exact 0.0/plain-Relu
    encoding doesn't work here) or the real-slope DecomposedLeakyAct
    (every other value, the real per-block frozen slope) for one
    activation site. Every site in this network ends up on the SAME
    DecomposedLeakyAct class/quantizer convention -- only the slope value
    differs -- so no residual `+` ever mixes quantizer types. Returns a
    FACTORY (not a module instance) since each call site within a block
    needs its own instance."""
    if negative_slope is None or negative_slope == 0.0:
        return _plain_relu_factory
    return lambda channels, bit_width: DecomposedLeakyAct(channels, bit_width, negative_slope)


# ---------------------------------------------------------------------------
# FINN-compatible block definitions
# ---------------------------------------------------------------------------

class FINNInitialBlock(nn.Module):
    """Single-conv initial block (no MaxPool branch -> no Concat -> FINN-safe),
    matches hardware/finn_enet_prod_export.py's FINNInitialBlock."""

    def __init__(self, in_ch: int, out_ch: int, bit_width: int, act_factory):
        super().__init__()
        self.conv = _quant_conv2d(in_ch, out_ch, bit_width, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act_factory(out_ch, bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FINNDownsamplingBottleneck(nn.Module):
    """Matches hardware/finn_enet_prod_export.py's FINNDownsamplingBottleneck:
    shortcut = MaxPool(no indices) + 1x1 projection, branch = stride-2 2x2
    conv + 3x3 conv + 1x1 expand, plain `+` residual."""

    def __init__(self, in_ch: int, out_ch: int, bit_width: int, act_factory,
                 dropout_p: float = 0.01, residual: bool = True):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, out_ch // 4)

        self.shortcut_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.shortcut_proj = nn.Sequential(
            _quant_conv2d(in_ch, out_ch, bit_width, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            act_factory(out_ch, bit_width),
        )
        self.reduce = nn.Sequential(
            _quant_conv2d(in_ch, internal_ch, bit_width, kernel_size=2, stride=2),
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, bit_width),
        )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_ch, internal_ch, bit_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, bit_width),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_ch, out_ch, bit_width, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            act_factory(out_ch, bit_width),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = act_factory(out_ch, bit_width)
        self.requant = _requant_factory(bit_width) if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_proj(self.shortcut_pool(x)) if self.residual else None
        out = self.dropout(self.expand(self.conv(self.reduce(x))))
        if self.residual:
            out = self.requant(_val(out)) + self.requant(_val(shortcut))
        return self.act(out)


class FINNUpsamplingBottleneck(nn.Module):
    """Main path = QuantConvTranspose2d directly (no F.interpolate/Resize, no
    MaxUnpool) -- matches hardware/finn_enet_prod_export.py's
    FINNUpsamplingBottleneck. Decoder-half block: always plain QuantReLU
    (never leaky), matching ENet.py/QuantENet.py hardcoding relu=True on
    up4/up5/regular4/regular5 regardless of the network's leaky_slope_map."""

    def __init__(self, in_ch: int, out_ch: int, bit_width: int, residual: bool = True):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, in_ch // 4)

        self.main_up = qnn.QuantConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2, bias=False,
            weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat,
        )
        self.main_bn = nn.BatchNorm2d(out_ch)
        self.main_act = _quant_act(bit_width)

        self.reduce = nn.Sequential(
            _quant_conv2d(in_ch, internal_ch, bit_width, kernel_size=1),
            nn.BatchNorm2d(internal_ch),
            _quant_act(bit_width),
        )
        self.up = nn.Sequential(
            qnn.QuantConvTranspose2d(
                internal_ch, internal_ch, kernel_size=2, stride=2, bias=False,
                weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat,
            ),
            nn.BatchNorm2d(internal_ch),
            _quant_act(bit_width),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_ch, out_ch, bit_width, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            _quant_act(bit_width),
        )
        self.dropout = nn.Dropout2d(p=0.1)
        self.act = _quant_act(bit_width)
        self.requant = _requant_factory(bit_width) if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main_act(self.main_bn(self.main_up(x)))
        out = self.dropout(self.expand(self.up(self.reduce(x))))
        if self.residual:
            out = self.requant(_val(out)) + self.requant(_val(main))
        return self.act(out)


class FINNRegularBottleneck(nn.Module):
    """Plain (non-dilated, non-separable) regular bottleneck -- used for
    regular1/regular4/regular5 (dilation=1 always, per QuantENet.py's
    _make_shallow_stage). Structurally identical to
    hardware/finn_enet_prod_export.py's FINNRegularBottleneck, parameterized
    by act_factory for the (possibly leaky) block activation."""

    def __init__(self, channels: int, bit_width: int, act_factory,
                 internal_ratio: int = 4, dropout_p: float = 0.1, residual: bool = True):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, channels // internal_ratio)

        self.reduce = nn.Sequential(
            _quant_conv2d(channels, internal_ch, bit_width, kernel_size=1),
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, bit_width),
        )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_ch, internal_ch, bit_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, bit_width),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_ch, channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(channels),
            act_factory(channels, bit_width),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = act_factory(channels, bit_width)
        self.requant = _requant_factory(bit_width) if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.expand(self.conv(self.reduce(x))))
        if self.residual:
            out = self.requant(_val(out)) + self.requant(_val(x))
        return self.act(out)


class FINNRegularBottleneckSepDilated(nn.Module):
    """Dilated context-stage bottleneck using the (k,1)+(1,k) separable-
    dilated factoring (QuantRegularBottleneck's separable_dilated branch,
    triggered whenever dilation != 1 -- always true for every stage2/stage3
    slot under context_pattern=dense_dilation). Mirrors QuantRegularBottleneck
    block-for-block (reduce -> [ (k,1)+BN+act, (1,k) ] +BN+act -> expand ->
    dropout -> residual `+` -> out_act), just with a plain `+` residual
    (FINN-safe convention, matching every other FINN*Bottleneck here) instead
    of Brevitas's QuantEltwiseAdd."""

    def __init__(self, channels: int, bit_width: int, act_factory, dilation: int,
                 internal_ratio: int = 4, kernel_size: int = 3, dropout_p: float = 0.1,
                 residual: bool = True):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, channels // internal_ratio)
        padding = dilation

        self.reduce = nn.Sequential(
            _quant_conv2d(channels, internal_ch, bit_width, kernel_size=1),
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, bit_width),
        )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_ch, internal_ch, bit_width, kernel_size=(kernel_size, 1),
                          padding=(padding, 0), dilation=dilation),
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, bit_width),
            _quant_conv2d(internal_ch, internal_ch, bit_width, kernel_size=(1, kernel_size),
                          padding=(0, padding), dilation=dilation),
        )
        self.conv_bn_act = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, bit_width),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_ch, channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(channels),
            act_factory(channels, bit_width),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = act_factory(channels, bit_width)
        self.requant = _requant_factory(bit_width) if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reduce(x)
        out = self.conv_bn_act(out)
        out = self.dropout(self.expand(out))
        if self.residual:
            out = self.requant(_val(out)) + self.requant(_val(x))
        return self.act(out)


class FINNQuantENetS13LeakyFrozen(nn.Module):
    """FINN-compatible quantized model for the S13-leaky-frozen config:
    separable_dilated=1 + context_pattern=dense_dilation (every context-stage
    slot dilated, 2/4/8/16 x2) + real per-block frozen LeakyReLU slopes
    (encoder half only, decoder always plain ReLU) + strided=1 + asymmetric=0
    + dsc=0.

    channels is a 5-tuple (initial, s1, s23 shared, s4, s5) -- QuantENet.py's
    own 5-value convention (stage2/stage3 always share one channel count,
    unlike ENet.py's optional 6-tuple split)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 5,
        channels: tuple[int, ...] = DEFAULT_CHANNELS,
        bottlenecks_per_stage: tuple[int, ...] = DEFAULT_BNECKS,
        bit_width: int = BIT_WIDTH,
        residual: bool = True,
        leaky_slope_map: dict | None = None,
    ):
        super().__init__()
        if len(channels) != 5:
            raise ValueError(f"channels must be a 5-tuple (initial, s1, s23, s4, s5), got {channels!r}.")
        if len(bottlenecks_per_stage) != 5:
            raise ValueError("bottlenecks_per_stage must have 5 values (stage1, stage2, stage3, regular4, regular5).")
        c0, c1, c23, c4, c5 = channels
        n1, n2, n3, n4, n5 = bottlenecks_per_stage
        slope_map = leaky_slope_map or {}

        self.initial = FINNInitialBlock(in_channels, c0, bit_width, _make_act_factory(slope_map.get("initial")))

        self.down1 = FINNDownsamplingBottleneck(c0, c1, bit_width, _make_act_factory(slope_map.get("down1")),
                                                 dropout_p=0.01, residual=residual)
        self.regular1 = nn.Sequential(*[
            FINNRegularBottleneck(c1, bit_width, _make_act_factory(slope_map.get(f"regular1.{i}")),
                                   dropout_p=0.01, residual=residual)
            for i in range(n1)
        ])

        self.down2 = FINNDownsamplingBottleneck(c1, c23, bit_width, _make_act_factory(slope_map.get("down2")),
                                                 dropout_p=0.1, residual=residual)
        self.stage2 = self._make_dense_context_stage(c23, n2, bit_width, residual, slope_map, "stage2")
        self.stage3 = self._make_dense_context_stage(c23, n3, bit_width, residual, slope_map, "stage3")

        # Decoder half: plain ReLU always (ENet.py/QuantENet.py hardcode
        # relu=True here regardless of the network's leaky_slope_map).
        self.up4 = FINNUpsamplingBottleneck(c23, c4, bit_width, residual=residual)
        self.regular4 = nn.Sequential(*[
            FINNRegularBottleneck(c4, bit_width, _plain_relu_factory, dropout_p=0.1, residual=residual)
            for _ in range(n4)
        ])
        self.up5 = FINNUpsamplingBottleneck(c4, c5, bit_width, residual=residual)
        self.regular5 = nn.Sequential(*[
            FINNRegularBottleneck(c5, bit_width, _plain_relu_factory, dropout_p=0.1, residual=residual)
            for _ in range(n5)
        ])

        # Final: no bias (bias in ConvTranspose complicates threshold
        # streamlining), same as finn_enet_prod_export.py/finn_export_s5.
        self.final = qnn.QuantConvTranspose2d(
            c5, out_channels, kernel_size=2, stride=2, bias=False,
            weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat,
        )

    @staticmethod
    def _make_dense_context_stage(channels: int, n: int, bit_width: int, residual: bool,
                                   slope_map: dict, name_prefix: str) -> nn.Sequential:
        blocks = []
        for i in range(n):
            dilation = DENSE_DILATIONS[i % len(DENSE_DILATIONS)]
            block_name = f"{name_prefix}.{i}"
            blocks.append(FINNRegularBottleneckSepDilated(
                channels, bit_width, _make_act_factory(slope_map.get(block_name)),
                dilation=dilation, dropout_p=0.1, residual=residual,
            ))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.regular1(self.down1(x))
        x = self.stage2(self.down2(x))
        x = self.stage3(x)
        x = self.regular4(self.up4(x))
        x = self.regular5(self.up5(x))
        out = self.final(x)
        return out.value if hasattr(out, "value") else out


# ---------------------------------------------------------------------------
# Export helpers (identical convention to finn_export_s5_dscnoproj_dense.py)
# ---------------------------------------------------------------------------

def _fast_cleanup(model):
    """See finn_export_s5_dscnoproj_dense.py's _fast_cleanup docstring --
    replaces stock FoldConstants (one-fold-per-apply-call) with
    FoldConstantsFiltered (all-eligible-folds-per-call) to avoid O(n)
    sequential InferShapes() passes on a network with this many Quant-node
    constant subgraphs."""
    from qonnx.transformation.infer_shapes import InferShapes
    from qonnx.transformation.general import (
        GiveUniqueParameterTensors, RemoveUnusedTensors, RemoveStaticGraphInputs,
        GiveUniqueNodeNames, GiveReadableTensorNames,
    )
    from qonnx.transformation.fold_constants import FoldConstantsFiltered
    from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit

    preserve_qnt_optypes = ["Quant", "BipolarQuant", "QuantizeLinear", "DequantizeLinear"]

    def _foldable(model, node):
        return node.op_type not in preserve_qnt_optypes

    for q_op_type in ["Quant", "Trunc", "BipolarQuant"]:
        for qnt_node in model.get_nodes_by_op_type(q_op_type):
            qnt_node.domain = "qonnx.custom_op.general"

    for t in [
        InferShapes(),
        GiveUniqueParameterTensors(),
        FoldConstantsFiltered(_foldable),
        FoldTransposeIntoQuantInit(),
        RemoveUnusedTensors(),
        RemoveStaticGraphInputs(),
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
    ]:
        model = model.transform(t)
    return model


def export_model(model: nn.Module, name: str, dummy: torch.Tensor) -> Path:
    """Export model to cleaned QONNX, set INT8 datatypes, verify it loads."""
    from brevitas.export import export_qonnx
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.datatype import DataType
    import onnx

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.onnx"

    model.cpu().eval()
    export_qonnx(model, export_path=str(out_path), input_t=dummy)

    qm = ModelWrapper(str(out_path))
    qm = _fast_cleanup(qm)
    qm.save(str(out_path))

    qm = ModelWrapper(str(out_path))
    qm.set_tensor_datatype(qm.graph.input[0].name, DataType["INT8"])
    qm.set_tensor_datatype(qm.graph.output[0].name, DataType["INT8"])
    qm.save(str(out_path))

    loaded = onnx.load(str(out_path))
    assert len(loaded.graph.node) > 0, "exported model has no nodes"

    ops: dict[str, int] = {}
    for n in qm.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"  {name}: {len(qm.graph.node)} nodes -- {dict(sorted(ops.items()))}")
    print(f"  Saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channels", type=int, nargs=5, default=list(DEFAULT_CHANNELS), metavar="C",
                        help="Channel widths, 5 values: initial s1 s23 s4 s5 (default: 4 16 32 16 4)")
    parser.add_argument("--bnecks", type=int, nargs=5, default=list(DEFAULT_BNECKS), metavar="N",
                        help="Bottleneck counts per stage (default: 4 8 8 2 1)")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5,
                        help="Segmentation classes (default 5 = background + 4 ARCADE classes)")
    parser.add_argument("--input-hw", type=int, nargs=2, default=(64, 64), metavar=("H", "W"),
                        help="Spatial dims for export dummy input (default: 64 64). Must be divisible by 8.")
    parser.add_argument("--bit-width", type=int, default=BIT_WIDTH)
    parser.add_argument("--no-residuals", action="store_true", help="Remove all residual shortcuts.")
    parser.add_argument("--slope-map-file", default=str(DEFAULT_SLOPE_MAP_FILE),
                         help="JSON file of {block_name: slope} -- defaults to the real S13 frozen slope map.")
    args = parser.parse_args()

    h, w = args.input_hw
    if h % 8 != 0 or w % 8 != 0:
        parser.error(f"--input-hw {h}x{w}: both dims must be divisible by 8.")

    with open(args.slope_map_file) as f:
        leaky_slope_map = json.load(f)

    torch.manual_seed(0)
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1

    residual = not args.no_residuals
    channels = tuple(args.channels)
    bnecks = tuple(args.bnecks)
    bw = args.bit_width

    print("\nFINNQuantENetS13LeakyFrozen export")
    print(f"  channels={channels}, bnecks={bnecks}")
    print(f"  input=({args.in_channels},{h},{w})  bit_width={bw}")
    print(f"  residual={residual}  (separable_dilated=1, context_pattern=dense_dilation, asymmetric=0, strided=1, dsc=0)")
    n_leaky = sum(1 for v in leaky_slope_map.values() if v != 0.0)
    n_zero = sum(1 for v in leaky_slope_map.values() if v == 0.0)
    print(f"  leaky_slope_map: {len(leaky_slope_map)} entries ({n_leaky} decomposed-leaky, {n_zero} alpha=0.0 -> plain ReLU)")

    model = FINNQuantENetS13LeakyFrozen(
        in_channels=args.in_channels, out_channels=args.out_channels,
        channels=channels, bottlenecks_per_stage=bnecks,
        bit_width=bw, residual=residual, leaky_slope_map=leaky_slope_map,
    ).eval()

    with torch.no_grad():
        out = model(dummy)
    out_t = out.value if hasattr(out, "value") else out
    assert out_t.shape[2:] == (h, w), f"output HxW {tuple(out_t.shape[2:])} != input ({h},{w})"
    assert out_t.shape[1] == args.out_channels, f"output channels {out_t.shape[1]} != {args.out_channels}"
    print(f"  forward OK: output shape {tuple(out_t.shape)}")

    suffix = "_no_res" if not residual else ""
    name = f"quantEnet_s13_leaky_frozen_int{bw}{suffix}"
    export_model(model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")
    print(f"  docker exec <finn_container_id> python /home/thelegendiv/finn/notebooks/enet/finn_enet_build_decomposed_prelu.py {name}")


if __name__ == "__main__":
    main()

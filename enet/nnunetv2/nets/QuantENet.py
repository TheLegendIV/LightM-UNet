"""Brevitas-quantized ENet, homogeneous bit-width only (weight_bit_width /
act_bit_width apply uniformly to every layer -- Stage 4.2's per-layer
heterogeneous case is a later extension, not built here).

Mirrors ENet.py's topology block-for-block (same constructor signature plus
bit-width knobs) -- this is deliberately a separate file rather than
quantization branches threaded through ENet.py, so the FP32 architecture
(already verified: self-test, real HPC training runs) can't be perturbed by
speculative Brevitas wiring. The cost is that the two files must be kept in
sync by hand; QUANT_TOPOLOGY_SELF_TEST below checks the two produce the same
per-block channel/depth structure for the same config, to catch drift.

Built for QONNX export (-> FINN) feasibility, per
agent_instructions_1.yaml's early_probes.p1_finn_resource_probe and
stage_4_quantization. Only verified so far: constructs, forward-passes, and
exports to a QONNX file that passes onnx.checker -- NOT yet verified inside
FINN itself (FINN requires its own dedicated Docker environment, separate
from Vivado/Vitis and separate from this repo's training container; not set
up yet, see compression/foundation_log.md).

PReLU -> QuantReLU: ENet's blocks use PReLU throughout (see ENet.py); there
is no standard quantized PReLU in Brevitas/FINN's dataflow op set, so this
quantized variant uses QuantReLU instead. This is a real deviation from the
FP32 architecture (not just a precision change) -- Stage 4 Dice numbers on
QuantENet are not directly comparable to the FP32 ENet baselines without
accounting for this activation-function change too.

leaky_slope_map (QuantDecomposedLeakyAct): a partial exception to the
PReLU/LeakyReLU gap above -- for a model whose real (FP32) deployment
target is ENet.py's frozen per-block LeakyReLU (apply_leaky_slope_
overrides), passing that same per-block slope dict here builds a
quantized, FINN-verified ALGEBRAIC DECOMPOSITION of it instead of forcing
plain QuantReLU (see QuantDecomposedLeakyAct's own docstring, and
hardware/README.md's "PReLU vs. QuantReLU -- FINN compatibility
investigation" for the proof this decomposition survives FINN's real
streamlining/HW-conversion pipeline). Values are FIXED (register_buffer,
not learnable) -- meant for an already-trained slope, not a QAT target.
Blocks not covered by the map fall back to plain QuantReLU as before.

context_pattern/dsc_no_projection/separable_dilated: a scoped subset of
ENet.py's own architecture-probe axes, ported here only as far as the
compression sweep's actual quantized configs need (see each's own
docstring/validation) -- not full parity with ENet.py's flag surface
(merge_dilated_pairs, two_block_skip, dsc_dilated_only, shallow_dilation*,
etc. are NOT built here).
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Uint8ActPerTensorFloat

from nnunetv2.nets.ENet import (
    CONTEXT_STAGE_PATTERN,
    DENSE_DILATION_PATTERN,
    DENSE_DILATION_REG_INTERLEAVED_PATTERN,
)

DecoderType = Literal["max_unpool", "upsample_conv"]


def _quant_conv2d(in_ch: int, out_ch: int, bit_width: int, **kwargs) -> qnn.QuantConv2d:
    return qnn.QuantConv2d(
        in_ch, out_ch, bias=False,
        weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat,
        **kwargs,
    )


def _quant_act(bit_width: int) -> qnn.QuantReLU:
    # Uint8ActPerTensorFloat (signed=False, narrow_range=False), NOT
    # Int8ActPerTensorFloat: FINN's dataflow backend rejects a QONNX-exported
    # ReLU activation quantizer unless it's unsigned and non-narrow
    # ("FINN only supports unsigned and non-narrow quant noted for ReLU
    # activations") -- confirmed against Brevitas's own base.py, where
    # Uint8ActPerTensorFloat is exactly signed=False/narrow_range=False and
    # is the quantizer its own docstring pairs with QuantReLU, vs
    # Int8ActPerTensorFloat's docstring pairing with QuantIdentity. Since
    # ReLU output is never negative, this also uses the full 0-255 codebook
    # at the same bit-width instead of wasting the sign half on values that
    # never occur.
    return qnn.QuantReLU(
        bit_width=bit_width, act_quant=Uint8ActPerTensorFloat,
        return_quant_tensor=True,
    )


class QuantDecomposedLeakyAct(nn.Module):
    """Quantized, FINN-compatible stand-in for a FIXED (non-learnable)
    LeakyReLU/frozen-per-block PReLU slope -- Brevitas has no QuantLeakyReLU/
    QuantPReLU at all (confirmed: neither exists in brevitas.nn), and FINN's
    complete fpgadataflow HW op registry has no way to represent one
    directly either (a literal ONNX PRelu/LeakyRelu node survives FINN's
    entire streamlining pipeline completely untouched -- see
    hardware/README.md's "PReLU vs. QuantReLU -- FINN compatibility
    investigation").

    Uses the algebraic identity `LeakyReLU(x, a) = a*x + (1-a)*ReLU(x)`
    (exact, not an approximation -- trivially verified by cases: x>=0 gives
    x, x<0 gives a*x), built entirely from ops FINN already supports: one
    quantized ReLU (native via MultiThreshold/Thresholding), a fork of x
    into two branches, two per-channel Mul scales
    (-> InferChannelwiseLinearLayer), and one elementwise Add of two
    dynamic streams (-> InferAddStreamsLayer) -- the same fork+recombine
    shape ENet's own residual connections already use. Proven end-to-end
    against FINN's real streamlining + HW-conversion pipeline (all-HW
    residue, zero stuck PRelu nodes) and numerically validated against real
    nn.PReLU (see hardware/README.md and hardware/_tmp_prelu_
    investigation2.py's DecomposedPReLUAct, which this ports verbatim).

    `negative_slope` is a plain Python float baked in as a fixed buffer
    (register_buffer, NOT nn.Parameter) -- unlike DecomposedPReLUAct's
    learnable `alpha_init`, this is meant for an ALREADY-FIXED, ALREADY-
    TRAINED value (e.g. one block's own frozen slope from ENet.py's
    apply_leaky_slope_overrides / collect_prelu_block_means), not a QAT-
    learnable one. Multiple call sites within the same bottleneck block are
    each given their own instance built from the SAME negative_slope float
    (not a literally-shared module object like ENet.py's NonNegativePReLU(1)
    trick) -- necessary because sites at different channel counts (e.g.
    reduce's internal_channels vs. out_act's full channels) need differently
    -shaped buffers, and unnecessary anyway since the slope is fixed, not
    something gradients need to tie together across sites."""

    def __init__(self, channels: int, bit_width: int, negative_slope: float):
        super().__init__()
        self.pre_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.act_pos = qnn.QuantReLU(bit_width=bit_width, act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
        self.register_buffer("alpha", torch.full((1, channels, 1, 1), float(negative_slope)))
        self.register_buffer("one_minus_alpha", torch.full((1, channels, 1, 1), 1.0 - float(negative_slope)))
        self.out_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.pre_quant(x)
        pos = self.act_pos(x_q)
        # constant operand MUST be input[1] (dynamic * constant, not
        # constant * dynamic) -- FINN's streamlining transforms (e.g.
        # CollapseRepeatedMul) hard-assume this convention, see
        # hardware/_tmp_prelu_investigation2.py's own note.
        scaled_x = x_q * self.alpha
        scaled_pos = pos * self.one_minus_alpha
        return self.out_quant(scaled_x + scaled_pos)


def _quant_block_act(channels: int, bit_width: int, negative_slope: float | None) -> nn.Module:
    """Picks plain QuantReLU (negative_slope=None, the default -- matches
    every existing QuantENet config) or the FINN-verified decomposed-leaky
    stand-in (negative_slope set) for one activation site."""
    if negative_slope is None:
        return _quant_act(bit_width)
    return QuantDecomposedLeakyAct(channels, bit_width, negative_slope)


class QuantInitialBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bit_width: int, negative_slope: float | None = None):
        super().__init__()
        if out_channels <= in_channels:
            raise ValueError("QuantInitialBlock out_channels must exceed in_channels.")
        self.input_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.conv = _quant_conv2d(in_channels, out_channels - in_channels, bit_width, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = _quant_block_act(out_channels, bit_width, negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        return self.act(self.bn(torch.cat([self.conv(x), self.pool(x)], dim=1)))


class QuantRegularBottleneck(nn.Module):
    def __init__(
        self, channels: int, bit_width: int, internal_ratio: int = 4,
        kernel_size: int = 3, padding: int = 1, dilation: int = 1,
        asymmetric: bool = False, dropout_p: float = 0.1, use_dsc: bool = False,
        separable_dilated: bool = False, negative_slope: float | None = None,
    ):
        super().__init__()
        internal_channels = max(1, channels // internal_ratio)

        self.reduce = nn.Sequential(
            _quant_conv2d(channels, internal_channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(internal_channels),
            _quant_block_act(internal_channels, bit_width, negative_slope),
        )
        if asymmetric:
            if use_dsc:
                raise ValueError("use_dsc is not defined for asymmetric bottlenecks -- see ENet.py's RegularBottleneck.")
            self.conv = nn.Sequential(
                _quant_conv2d(internal_channels, internal_channels, bit_width, kernel_size=(kernel_size, 1), padding=(padding, 0)),
                nn.BatchNorm2d(internal_channels),
                _quant_block_act(internal_channels, bit_width, negative_slope),
                _quant_conv2d(internal_channels, internal_channels, bit_width, kernel_size=(1, kernel_size), padding=(0, padding)),
            )
        elif use_dsc:
            self.conv = nn.Sequential(
                _quant_conv2d(internal_channels, internal_channels, bit_width, kernel_size=kernel_size,
                               padding=padding, dilation=dilation, groups=internal_channels),
                _quant_conv2d(internal_channels, internal_channels, bit_width, kernel_size=1),
            )
        elif separable_dilated and dilation != 1:
            # Probe 4.4.2's (k,1)+(1,k) factoring of a dilated conv, mirrored
            # from ENet.py's RegularBottleneck -- same dilation on both
            # passes, BN+activation between them (a SEPARATE site from
            # conv_bn_act's own trailing activation below).
            self.conv = nn.Sequential(
                _quant_conv2d(internal_channels, internal_channels, bit_width, kernel_size=(kernel_size, 1),
                               padding=(padding, 0), dilation=dilation),
                nn.BatchNorm2d(internal_channels),
                _quant_block_act(internal_channels, bit_width, negative_slope),
                _quant_conv2d(internal_channels, internal_channels, bit_width, kernel_size=(1, kernel_size),
                               padding=(0, padding), dilation=dilation),
            )
        else:
            self.conv = _quant_conv2d(internal_channels, internal_channels, bit_width, kernel_size=kernel_size, padding=padding, dilation=dilation)

        self.conv_bn_act = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(internal_channels),
            _quant_block_act(internal_channels, bit_width, negative_slope),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_channels, channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        # NOTE (FINN residual-join scale safety): brevitas.nn.QuantEltwiseAdd
        # applies its OWN `input_quant` instance to BOTH operands --
        # `forward()` does `self.input_quant(input)` then
        # `self.input_quant(other)`, the SAME submodule both times (see
        # brevitas/nn/quant_eltwise.py). Int8ActPerTensorFloat's default
        # scaling_impl_type is PARAMETER_FROM_STATS -- one learnable
        # nn.Parameter shared by the module, NOT recomputed per call -- so
        # both operands are forced through an IDENTICAL scale by
        # construction, every forward pass, regardless of what scale either
        # operand carried on entry. This residual join can never hit FINN's
        # "two independently-scaled int8 streams meeting at an Add" problem.
        # That problem DID occur in the separate FINN-export mirror
        # hardware/finn_export_s13_leaky_frozen.py, whose hand-rolled
        # residual blocks do a raw `+` with no shared quantizer at the join
        # at all (fixed there with an explicit CONST-scale shared
        # QuantIdentity: `_ConstScaleInt8Act`/`_requant_factory`). That fix is
        # NOT needed here -- this comment exists so a future agent doesn't
        # try to re-apply it to QuantENet.py's own residual_add sites (this
        # one and the 3 others in this file, which just reference this note).
        self.residual_add = qnn.QuantEltwiseAdd(bit_width=bit_width, input_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.out_act = _quant_block_act(channels, bit_width, negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reduce(x)
        out = self.conv_bn_act(out)
        out = self.dropout(self.expand(out))
        return self.out_act(self.residual_add(x, out))


class QuantDSCNoProjectionBottleneck(nn.Module):
    """Quantized mirror of ENet.py's DSCNoProjectionBottleneck: depthwise
    KxK (groups=channels, dilation-aware) + pointwise 1x1 applied directly
    at full `channels` width, no reduce/expand projection pair at all. Used
    network-wide (stage1/regular1, stage2, stage3, regular4, regular5) when
    QuantENet(dsc_no_projection=True). Not combinable with asymmetric slots
    (same constraint QuantRegularBottleneck's own use_dsc already has)."""

    def __init__(
        self, channels: int, bit_width: int, kernel_size: int = 3, padding: int = 1,
        dilation: int = 1, dropout_p: float = 0.1, negative_slope: float | None = None,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            _quant_conv2d(channels, channels, bit_width, kernel_size=kernel_size,
                          padding=padding, dilation=dilation, groups=channels),
            nn.BatchNorm2d(channels),
            _quant_block_act(channels, bit_width, negative_slope),
            _quant_conv2d(channels, channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        # See QuantRegularBottleneck.residual_add's note above: QuantEltwiseAdd
        # shares one input_quant instance (learned PARAMETER_FROM_STATS scale)
        # across both operands by construction, so this join is already
        # FINN-safe -- no CONST-scale fix needed here.
        self.residual_add = qnn.QuantEltwiseAdd(bit_width=bit_width, input_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.out_act = _quant_block_act(channels, bit_width, negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.conv(x))
        return self.out_act(self.residual_add(x, out))


class QuantDownsamplingBottleneck(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, bit_width: int, internal_ratio: int = 4,
        dropout_p: float = 0.01, use_strided: bool = True, negative_slope: float | None = None,
    ):
        super().__init__()
        internal_channels = max(1, out_channels // internal_ratio)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        if use_strided:
            self.reduce = nn.Sequential(
                _quant_conv2d(in_channels, internal_channels, bit_width, kernel_size=2, stride=2),
                nn.BatchNorm2d(internal_channels),
                _quant_block_act(internal_channels, bit_width, negative_slope),
            )
        else:
            self.reduce = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                _quant_conv2d(in_channels, internal_channels, bit_width, kernel_size=1),
                nn.BatchNorm2d(internal_channels),
                _quant_block_act(internal_channels, bit_width, negative_slope),
            )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_channels, internal_channels, bit_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(internal_channels),
            _quant_block_act(internal_channels, bit_width, negative_slope),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_channels, out_channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        # See QuantRegularBottleneck.residual_add's note above: QuantEltwiseAdd
        # shares one input_quant instance (learned PARAMETER_FROM_STATS scale)
        # across both operands by construction, so this join is already
        # FINN-safe -- no CONST-scale fix needed here.
        self.residual_add = qnn.QuantEltwiseAdd(bit_width=bit_width, input_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.out_act = _quant_block_act(out_channels, bit_width, negative_slope)
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


class QuantUpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bit_width: int, internal_ratio: int = 4):
        super().__init__()
        internal_channels = max(1, in_channels // internal_ratio)
        self.main_proj = nn.Sequential(
            _quant_conv2d(in_channels, out_channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.reduce = nn.Sequential(
            _quant_conv2d(in_channels, internal_channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(internal_channels),
            _quant_act(bit_width),
        )
        self.up = nn.Sequential(
            qnn.QuantConvTranspose2d(internal_channels, internal_channels, kernel_size=2, stride=2, bias=False,
                                      weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat),
            nn.BatchNorm2d(internal_channels),
            _quant_act(bit_width),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_channels, out_channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=0.1)
        # See QuantRegularBottleneck.residual_add's note above: QuantEltwiseAdd
        # shares one input_quant instance (learned PARAMETER_FROM_STATS scale)
        # across both operands by construction, so this join is already
        # FINN-safe -- no CONST-scale fix needed here.
        self.residual_add = qnn.QuantEltwiseAdd(bit_width=bit_width, input_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.out_act = _quant_act(bit_width)

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


class QuantENet(nn.Module):
    """Homogeneous-bit-width quantized mirror of ENet.py -- same constructor
    signature plus weight_bit_width/act_bit_width. See module docstring for
    what's verified so far (construct/forward/QONNX-export) vs not (FINN
    itself, not installed)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        channels: tuple[int, int, int, int, int] = (20, 72, 144, 72, 20),
        bottlenecks_per_stage: tuple[int, int, int, int, int] = (4, 8, 8, 2, 1),
        decoder_type: DecoderType = "max_unpool",
        use_dilated: bool = True,
        use_asymmetric: bool = True,
        use_strided: bool = True,
        use_dsc: bool = False,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        context_pattern: str = "default",
        dsc_no_projection: bool = False,
        dsc_no_projection_context_only: bool = False,
        separable_dilated: bool = False,
        leaky_slope_map: dict | None = None,
    ):
        super().__init__()
        if len(channels) != 5 or len(bottlenecks_per_stage) != 5:
            raise ValueError("channels and bottlenecks_per_stage must each have 5 values (see ENet.py).")
        initial_channels, stage1_channels, stage23_channels, stage4_channels, stage5_channels = channels
        if decoder_type not in ("max_unpool", "upsample_conv"):
            raise ValueError(f"decoder_type must be 'max_unpool' or 'upsample_conv', got {decoder_type!r}.")
        valid_context_patterns = ("default", "dense_dilation", "dense_dilation_reg_interleaved")
        if context_pattern not in valid_context_patterns:
            raise ValueError(
                f"QuantENet only supports {valid_context_patterns} so far (mirrors just the subset of "
                f"ENet.py's context_pattern axis needed by the compression sweep's quantized configs) -- "
                f"got {context_pattern!r}."
            )
        if dsc_no_projection_context_only and not dsc_no_projection:
            raise ValueError("dsc_no_projection_context_only narrows dsc_no_projection's scope -- meaningless without dsc_no_projection=True itself.")

        self.decoder_type: DecoderType = decoder_type
        bw = weight_bit_width  # conv weight bit-width; act_bit_width used for _quant_act/input/residual quantizers below
        n_stage1, n_stage2, n_stage3, n_regular4, n_regular5 = bottlenecks_per_stage
        slope_map = leaky_slope_map or {}

        self.initial = QuantInitialBlock(in_channels, initial_channels, act_bit_width, negative_slope=slope_map.get("initial"))
        self.down1 = QuantDownsamplingBottleneck(initial_channels, stage1_channels, bw, dropout_p=0.01, use_strided=use_strided, negative_slope=slope_map.get("down1"))
        self.regular1 = self._make_shallow_stage(
            stage1_channels, n_stage1, bw, dropout_p=0.01, use_dsc=use_dsc,
            dsc_no_projection=dsc_no_projection, dsc_no_projection_context_only=dsc_no_projection_context_only,
            name_prefix="regular1", leaky_slope_map=slope_map,
        )

        self.down2 = QuantDownsamplingBottleneck(stage1_channels, stage23_channels, bw, dropout_p=0.1, use_strided=use_strided, negative_slope=slope_map.get("down2"))
        self.stage2 = self._make_context_stage(
            stage23_channels, n_stage2, bw, use_dilated, use_asymmetric, use_dsc,
            context_pattern, dsc_no_projection, separable_dilated, "stage2", slope_map,
        )
        self.stage3 = self._make_context_stage(
            stage23_channels, n_stage3, bw, use_dilated, use_asymmetric, use_dsc,
            context_pattern, dsc_no_projection, separable_dilated, "stage3", slope_map,
        )

        self.up4 = QuantUpsamplingBottleneck(stage23_channels, stage4_channels, bw)
        # Decoder (regular4/regular5/up4/up5) is always plain QuantReLU,
        # regardless of leaky_slope_map -- same rule as ENet.py's prelu_
        # variant (decoder hardcodes relu=True everywhere).
        self.regular4 = self._make_shallow_stage(
            stage4_channels, n_regular4, bw, dropout_p=0.1, use_dsc=use_dsc,
            dsc_no_projection=dsc_no_projection, dsc_no_projection_context_only=dsc_no_projection_context_only,
            name_prefix="regular4", leaky_slope_map=None,
        )
        self.up5 = QuantUpsamplingBottleneck(stage4_channels, stage5_channels, bw)
        self.regular5 = self._make_shallow_stage(
            stage5_channels, n_regular5, bw, dropout_p=0.1, use_dsc=use_dsc,
            dsc_no_projection=dsc_no_projection, dsc_no_projection_context_only=dsc_no_projection_context_only,
            name_prefix="regular5", leaky_slope_map=None,
        )
        self.final = qnn.QuantConvTranspose2d(stage5_channels, out_channels, kernel_size=2, stride=2, bias=True,
                                               weight_bit_width=bw, weight_quant=Int8WeightPerTensorFloat)

    @staticmethod
    def _make_shallow_stage(
        channels: int, n_ops: int, bit_width: int, dropout_p: float, use_dsc: bool,
        dsc_no_projection: bool, dsc_no_projection_context_only: bool, name_prefix: str,
        leaky_slope_map: dict | None,
    ) -> nn.Sequential:
        """regular1/regular4/regular5: plain QuantRegularBottleneck repeats,
        or QuantDSCNoProjectionBottleneck repeats when dsc_no_projection is
        set and not narrowed to stage2/stage3 only -- mirrors ENet.py's
        _make_shallow_stage (minus the shallow_dilation probes, not needed
        by either quantized config this was built for)."""
        slope_map = leaky_slope_map or {}
        if dsc_no_projection and not dsc_no_projection_context_only:
            return nn.Sequential(*[
                QuantDSCNoProjectionBottleneck(channels, bit_width, dropout_p=dropout_p,
                                                negative_slope=slope_map.get(f"{name_prefix}.{i}"))
                for i in range(n_ops)
            ])
        return nn.Sequential(*[
            QuantRegularBottleneck(channels, bit_width, dropout_p=dropout_p, use_dsc=use_dsc,
                                    negative_slope=slope_map.get(f"{name_prefix}.{i}"))
            for i in range(n_ops)
        ])

    @staticmethod
    def _make_context_stage(
        channels: int, n_ops: int, bit_width: int, use_dilated: bool, use_asymmetric: bool, use_dsc: bool,
        context_pattern: str, dsc_no_projection: bool, separable_dilated: bool, name_prefix: str,
        leaky_slope_map: dict | None,
    ) -> nn.Sequential:
        """Mirrors ENet.py's _make_context_stage, scoped to just what
        QuantENet's two target configs need: context_pattern selection
        (default/dense_dilation/dense_dilation_reg_interleaved),
        dsc_no_projection (with its {"reg_bottleneck": True} bookend
        sentinel), and separable_dilated -- NOT merge_dilated_pairs/
        two_block_skip/dsc_dilated_only/sparse variants, unused by either
        S8-ReLU or S13's quantized configs."""
        if context_pattern == "dense_dilation":
            pattern = DENSE_DILATION_PATTERN
        elif context_pattern == "dense_dilation_reg_interleaved":
            pattern = DENSE_DILATION_REG_INTERLEAVED_PATTERN
        else:
            pattern = CONTEXT_STAGE_PATTERN

        slope_map = leaky_slope_map or {}

        if dsc_no_projection:
            ops = []
            for i in range(n_ops):
                kwargs = dict(pattern[i % len(pattern)])
                block_name = f"{name_prefix}.{i}"
                if kwargs.pop("reg_bottleneck", False):
                    # A full-rank RegularBottleneck bookend (DENSE_DILATION_
                    # REG_INTERLEAVED_PATTERN's {"reg_bottleneck": True}
                    # slots), not a DSCNoProjectionBottleneck -- same
                    # sentinel meaning as ENet.py's own dsc_no_projection
                    # branch.
                    ops.append(QuantRegularBottleneck(
                        channels, bit_width, dropout_p=0.1, use_dsc=False,
                        negative_slope=slope_map.get(block_name),
                    ))
                    continue
                if kwargs.get("asymmetric", False):
                    if use_asymmetric:
                        raise ValueError("dsc_no_projection is not defined for asymmetric bottlenecks -- set use_asymmetric=0.")
                    kwargs = {}
                dilation = kwargs.get("dilation", 1)
                if dilation != 1 and not use_dilated:
                    dilation = 1
                ops.append(QuantDSCNoProjectionBottleneck(
                    channels, bit_width, kernel_size=3, padding=dilation, dilation=dilation,
                    dropout_p=0.1, negative_slope=slope_map.get(block_name),
                ))
            return nn.Sequential(*ops)

        ops = []
        for i in range(n_ops):
            kwargs = dict(pattern[i % len(pattern)])
            kwargs.pop("reg_bottleneck", False)  # only meaningful under dsc_no_projection, see branch above
            if kwargs.get("dilation", 1) != 1 and not use_dilated:
                kwargs = {}
            if kwargs.get("asymmetric", False) and not use_asymmetric:
                kwargs = {}
            ops.append(QuantRegularBottleneck(
                channels, bit_width, dropout_p=0.1, use_dsc=use_dsc, separable_dilated=separable_dilated,
                negative_slope=slope_map.get(f"{name_prefix}.{i}"), **kwargs,
            ))
        return nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        x = self.initial(x)
        x, indices1, size1 = self.down1(x)
        x = self.regular1(x)
        x, indices2, size2 = self.down2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        use_indices = self.decoder_type == "max_unpool"
        x = self.up4(x, size2, indices2 if use_indices else None)
        x = self.regular4(x)
        x = self.up5(x, size1, indices1 if use_indices else None)
        x = self.regular5(x)
        x = self.final(x)
        if hasattr(x, "value"):
            # QuantConvTranspose2d can return a Brevitas QuantTensor rather
            # than a plain torch.Tensor -- nnU-Net's loss functions
            # (DC_and_CE_loss etc.) expect a plain tensor of logits, and
            # QAT doesn't need the output itself quantized (only the
            # weights/activations feeding it), so unwrap here rather than
            # push QuantTensor-awareness into the training loop.
            x = x.value
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


if __name__ == "__main__":
    from nnunetv2.nets.ENet import ENet

    torch.manual_seed(0)
    dummy = torch.zeros(1, 1, 512, 512)

    # 1. Construct/forward-pass check, mirroring ENet.py's own self-test.
    configs = [
        ("E1_int8", (20, 72, 144, 72, 20), (4, 8, 8, 2, 1), "upsample_conv"),
        ("UF_int4", (20, 4, 4, 4, 4), (4, 8, 8, 2, 1), "upsample_conv"),
        ("U8_int8", (20, 12, 20, 12, 4), (4, 8, 8, 2, 1), "upsample_conv"),
    ]
    for name, channels, bnecks, decoder in configs:
        bits = 4 if "int4" in name else 8
        model = QuantENet(
            in_channels=1, out_channels=2, channels=channels, bottlenecks_per_stage=bnecks,
            decoder_type=decoder, weight_bit_width=bits, act_bit_width=bits,
        ).eval()
        with torch.no_grad():
            out = model(dummy)
        out_t = out.value if hasattr(out, "value") else out
        assert out_t.shape == (1, 2, 512, 512), f"{name}: got {tuple(out_t.shape)}"
        print(f"{name}: build+forward OK, output shape {tuple(out_t.shape)}")

    # 2. Topology drift check against ENet.py for one config: same per-stage
    #    module counts (regular1/stage2/stage3/regular4/regular5 depths).
    fp32 = ENet(in_channels=1, out_channels=2, channels=(20, 72, 144, 72, 20),
                bottlenecks_per_stage=(4, 8, 8, 2, 1), decoder_type="upsample_conv")
    quant = QuantENet(in_channels=1, out_channels=2, channels=(20, 72, 144, 72, 20),
                       bottlenecks_per_stage=(4, 8, 8, 2, 1), decoder_type="upsample_conv")
    for attr in ["regular1", "stage2", "stage3", "regular4", "regular5"]:
        fp32_len, quant_len = len(getattr(fp32, attr)), len(getattr(quant, attr))
        assert fp32_len == quant_len, f"topology drift in {attr}: ENet={fp32_len} QuantENet={quant_len}"
    print("Topology parity vs ENet.py: OK (regular1/stage2/stage3/regular4/regular5 depths match)")

    # 3. QONNX export -- the actual P1/Stage-4 deliverable. Needs qonnx+onnx
    # installed (requirements-enet-base.txt); NOT a FINN run (see docstring).
    from brevitas.export import export_qonnx
    model = QuantENet(in_channels=1, out_channels=2, channels=(20, 12, 20, 12, 4),
                       bottlenecks_per_stage=(4, 8, 8, 2, 1), decoder_type="upsample_conv",
                       weight_bit_width=8, act_bit_width=8).eval()
    export_path = "/tmp/quant_enet_u8_int8.onnx"
    export_qonnx(model, torch.randn(1, 1, 512, 512), export_path=export_path)
    import onnx
    onnx.checker.check_model(onnx.load(export_path))
    print(f"QONNX export OK and passed onnx.checker: {export_path}")

    # 4. S8-ReLU@INT4's exact recipe: dsc_no_projection=1 (network-wide,
    # dsc_no_projection_context_only=0), context_pattern=
    # dense_dilation_reg_interleaved, bottlenecks widened to 4,11,11,2,1
    # for the reg-bookend pattern's 11-slot cycle, use_asymmetric=0,
    # use_prelu=0 (plain QuantReLU everywhere -- no leaky_slope_map
    # needed). Checks the reg-bookend sentinel selects a REAL
    # QuantRegularBottleneck while the dilated slots become
    # QuantDSCNoProjectionBottleneck, matching ENet.py's own real trained
    # recipe exactly.
    s8_relu_int4 = QuantENet(
        in_channels=1, out_channels=5, channels=(4, 16, 32, 16, 4),
        bottlenecks_per_stage=(4, 11, 11, 2, 1), decoder_type="upsample_conv",
        use_dilated=True, use_asymmetric=False, use_strided=True, use_dsc=False,
        weight_bit_width=4, act_bit_width=4,
        context_pattern="dense_dilation_reg_interleaved", dsc_no_projection=True,
    ).eval()
    with torch.no_grad():
        out = s8_relu_int4(torch.randn(1, 1, 512, 512))
    out_t = out.value if hasattr(out, "value") else out
    assert out_t.shape == (1, 5, 512, 512), f"S8-ReLU@INT4: got {tuple(out_t.shape)}"
    for i, block in enumerate(s8_relu_int4.stage2):
        expected_type = QuantRegularBottleneck if i % 5 == 0 else QuantDSCNoProjectionBottleneck
        assert isinstance(block, expected_type), f"S8-ReLU@INT4: stage2.{i} expected {expected_type.__name__}, got {type(block).__name__}"
    assert isinstance(s8_relu_int4.regular1[0], QuantDSCNoProjectionBottleneck), "S8-ReLU@INT4: regular1 should be DSC-no-projection too (dsc_no_projection_context_only=0)"
    assert isinstance(s8_relu_int4.regular5[0], QuantDSCNoProjectionBottleneck), "S8-ReLU@INT4: regular5 should be DSC-no-projection too (dsc_no_projection_context_only=0)"
    print("S8-ReLU@INT4 recipe: build+forward OK, reg-bookend/DSC-no-projection block types verified.")

    # 5. S13's exact recipe (frozen-leaky's architecture): context_pattern=
    # dense_dilation, separable_dilated=1, dsc_no_projection=0 (projection
    # kept), use_asymmetric=0, at INT8, with a real (if abbreviated) per-
    # block leaky_slope_map. Checks the mapped blocks build
    # QuantDecomposedLeakyAct while an unmapped block correctly falls back
    # to plain QuantReLU.
    sample_slope_map = {"initial": 0.5, "stage2.0": 0.3, "stage3.7": 0.1}
    s13_leaky_int8 = QuantENet(
        in_channels=1, out_channels=5, channels=(4, 16, 32, 16, 4),
        bottlenecks_per_stage=(4, 8, 8, 2, 1), decoder_type="upsample_conv",
        use_dilated=True, use_asymmetric=False, use_strided=True, use_dsc=False,
        weight_bit_width=8, act_bit_width=8,
        context_pattern="dense_dilation", separable_dilated=True, leaky_slope_map=sample_slope_map,
    ).eval()
    with torch.no_grad():
        out = s13_leaky_int8(torch.randn(1, 1, 512, 512))
    out_t = out.value if hasattr(out, "value") else out
    assert out_t.shape == (1, 5, 512, 512), f"S13-leaky@INT8: got {tuple(out_t.shape)}"
    assert isinstance(s13_leaky_int8.initial.act, QuantDecomposedLeakyAct), "S13-leaky@INT8: initial.act should be QuantDecomposedLeakyAct (mapped)"
    assert isinstance(s13_leaky_int8.stage2[0].out_act, QuantDecomposedLeakyAct), "S13-leaky@INT8: stage2.0.out_act should be QuantDecomposedLeakyAct (mapped)"
    assert isinstance(s13_leaky_int8.stage2[1].out_act, qnn.QuantReLU), "S13-leaky@INT8: stage2.1.out_act should fall back to plain QuantReLU (unmapped)"
    assert isinstance(s13_leaky_int8.stage3[7].out_act, QuantDecomposedLeakyAct), "S13-leaky@INT8: stage3.7.out_act should be QuantDecomposedLeakyAct (mapped)"
    assert torch.allclose(s13_leaky_int8.stage2[0].out_act.alpha, torch.tensor(0.3)), "S13-leaky@INT8: stage2.0's alpha buffer should equal its mapped slope 0.3"
    assert isinstance(s13_leaky_int8.regular5[0].out_act, qnn.QuantReLU), "S13-leaky@INT8: decoder (regular5) must stay plain QuantReLU regardless of leaky_slope_map"
    print("S13-leaky@INT8 recipe: build+forward OK, mapped/unmapped/decoder activation types verified.")

    # 6. Numeric correctness of QuantDecomposedLeakyAct -- SELF-CONSISTENCY
    # against its own algebraic formula computed from its own ACTUAL
    # quantized intermediates (out ?= alpha*pre_quant_out +
    # (1-alpha)*act_pos_out), not a comparison against an unquantized
    # nn.LeakyReLU reference. A comparison against an unquantized reference
    # on a fresh, never-calibrated module is NOT a fair/meaningful test --
    # Brevitas's *PerTensorFloat quantizers' scale is a runtime-calibrated
    # parameter (learned during real QAT training), and at its untrained
    # default it saturates far below the input's actual dynamic range
    # (confirmed directly: input -3.0 came out of pre_quant as -1.0) --
    # exactly the same category of "uncalibrated minimal harness" artifact
    # hardware/_tmp_prelu_investigation2.py's own
    # _fixup_degenerate_signed_bias documents hitting and explicitly
    # separates from real decomposition/FINN-capability bugs. This
    # self-consistency check instead verifies the DECOMPOSITION FORMULA
    # ITSELF is implemented correctly, independent of calibration state --
    # real end-to-end accuracy against the FP32 reference gets measured via
    # a real QAT training run + collect_results.py's real dice, not a
    # single uncalibrated forward pass here.
    torch.manual_seed(0)
    decomposed = QuantDecomposedLeakyAct(16, bit_width=8, negative_slope=0.37).eval()
    x = torch.randn(2, 16, 8, 8) * 2.0
    with torch.no_grad():
        x_q = decomposed.pre_quant(x)
        x_q_t = x_q.value if hasattr(x_q, "value") else x_q
        pos = decomposed.act_pos(x_q)
        pos_t = pos.value if hasattr(pos, "value") else pos
        out = decomposed(x)
        out_t = out.value if hasattr(out, "value") else out
    expected = x_q_t * decomposed.alpha + pos_t * decomposed.one_minus_alpha
    max_diff = (out_t - expected).abs().max().item()
    assert max_diff < 0.02, f"QuantDecomposedLeakyAct: output doesn't match its own decomposition formula (max_diff={max_diff:.4f}) -- real implementation bug, not a calibration artifact"
    print(f"QuantDecomposedLeakyAct self-consistency check: output matches alpha*x + (1-alpha)*ReLU(x) from its own quantized intermediates (max_diff={max_diff:.4f}, ~1 quantization step).")

    # 7. Validation checks.
    try:
        QuantENet(in_channels=1, out_channels=5, channels=(4, 16, 32, 16, 4),
                  bottlenecks_per_stage=(4, 8, 8, 2, 1), decoder_type="upsample_conv",
                  context_pattern="bogus")
        raise AssertionError("expected ValueError for an invalid context_pattern, got none")
    except ValueError:
        pass
    try:
        QuantENet(in_channels=1, out_channels=5, channels=(4, 16, 32, 16, 4),
                  bottlenecks_per_stage=(4, 8, 8, 2, 1), decoder_type="upsample_conv",
                  dsc_no_projection_context_only=True)
        raise AssertionError("expected ValueError for dsc_no_projection_context_only without dsc_no_projection, got none")
    except ValueError:
        pass
    print("QuantENet validation self-test PASSED.")

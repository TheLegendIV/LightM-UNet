"""NOT YET ADOPTED BY ANY TRAINER -- a standalone, parametrized unification
of QuantENetS19Block.py / QuantENet26_5_w24.py / QuantENet5_6Block.py (and,
via uniform-bits broadcasting, QuantENet23_1.py's per-stage granularity
too), prepared for a future refactor once the per-block-QAT-stuck-training
investigation (see compression/configs/ and this session's own notes) has
settled. Created deliberately as a NEW, separate file rather than by
editing any of those four in place: every one of them is currently a
provenance record of exactly what code a specific real checkpoint was
trained with, and touching them now would make it impossible to tell a
real new data point about the stuck-training mystery apart from a
refactor regression. This file is safe to read, extend, and unit-test
against those checkpoints, but nothing currently imports or trains through
it.

WHAT'S ACTUALLY UNIFIED: those four files differ only in (a) which
architecture-shape constants are baked in as module globals (CHANNELS,
BOTTLENECKS_PER_STAGE, CONTEXT_PATTERN, PRELU_VARIANT, SEPARABLE_DILATED,
...) and (b) whether the W/A bit-width interface is per-STAGE (QuantENet23_1,
one shared pair per one of 5 stage groups) or per-BLOCK (the other three,
one pair per individual bottleneck). The actual block-assembly logic
(_make_block_shallow_stage/_make_block_context_stage, near-duplicated
verbatim across all three per-block files bar which dilation PATTERN
constant they select) and the underlying primitives (QuantInitialBlock,
QuantRegularBottleneck, QuantDownsamplingBottleneck, QuantUpsamplingBottleneck,
QuantDecomposedLeakyAct) already live in QuantENet.py and are reused here
unchanged, not reimplemented.

SCOPE (deliberately narrow, same philosophy every file it unifies already
uses): decoder_type is hardcoded to "upsample_conv" (forward() has no
max_unpool/indices path -- none of the four configs unified here use
max_unpool either). context_pattern supports the same subset QuantENet.
_make_context_stage already supports ("default", "dense_dilation",
"dense_dilation_reg_interleaved", "dense_dilation_reg_interleaved_double_mid").
dsc_no_projection's {"reg_bottleneck": True} bookend handling and the
asymmetric-kwargs path are ported faithfully from QuantENet._make_context_stage,
but only the dsc_no_projection=False, use_asymmetric=False path has been
exercised against real configs so far (every one of S19/26_5_w24/5_6 uses
those defaults) -- the other paths are included for completeness/future
use, not independently verified here.

USAGE (not yet exercised in a real trainer):
    from nnunetv2.nets.CombinedQuantENet import CombinedQuantENet, block_names_for

    block_names = block_names_for(bottlenecks_per_stage=(4, 12, 12, 2, 1))
    model = CombinedQuantENet(
        block_weight_bits, block_act_bits,  # one entry per block_names
        out_channels=5, channels=(4, 16, 32, 16, 4), bottlenecks_per_stage=(4, 12, 12, 2, 1),
        context_pattern="dense_dilation_reg_interleaved_double_mid",
        leaky_slope_map=slope_map, trainable_slope=True,
    )
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat

from nnunetv2.nets.ENet import (
    CONTEXT_STAGE_PATTERN,
    DENSE_DILATION_PATTERN,
    DENSE_DILATION_REG_INTERLEAVED_PATTERN,
    DENSE_DILATION_REG_INTERLEAVED_DOUBLE_MID_PATTERN,
)
from nnunetv2.nets.QuantENet import (
    QuantDownsamplingBottleneck,
    QuantDSCNoProjectionBottleneck,
    QuantInitialBlock,
    QuantRegularBottleneck,
    QuantUpsamplingBottleneck,
)

VALID_CONTEXT_PATTERNS = (
    "default", "dense_dilation", "dense_dilation_reg_interleaved",
    "dense_dilation_reg_interleaved_double_mid",
)


def block_names_for(bottlenecks_per_stage: tuple[int, int, int, int, int]) -> tuple[str, ...]:
    """The BLOCK_NAMES tuple every per-block file (QuantENetS19Block.py/
    QuantENet26_5_w24.py/QuantENet5_6Block.py) currently hardcodes by hand,
    derived generically instead -- confirmed to reproduce
    QuantENetS19Block.BLOCK_NAMES exactly for bottlenecks_per_stage=
    (4, 12, 12, 2, 1). proj2_to_3 is never included: every unified config so
    far has stage2_channels == stage3_channels (ENet.py builds it as
    nn.Identity() in that case, nothing to quantize) -- a config that
    genuinely needs it would have to extend this function, not silently
    get a wrong block list."""
    n_stage1, n_stage2, n_stage3, n_regular4, n_regular5 = bottlenecks_per_stage
    return (
        "initial", "down1",
        *[f"regular1.{i}" for i in range(n_stage1)],
        "down2",
        *[f"stage2.{i}" for i in range(n_stage2)],
        *[f"stage3.{i}" for i in range(n_stage3)],
        "up4",
        *[f"regular4.{i}" for i in range(n_regular4)],
        "up5",
        *[f"regular5.{i}" for i in range(n_regular5)],
        "final",
    )


def expand_uniform_bits(weight_bit_width: int, act_bit_width: int, block_names: tuple[str, ...]) -> tuple[dict, dict]:
    """Broadcasts a single (w, a) pair to every block -- lets a caller
    build the QuantENet23_1-style "one shared pair for the whole network"
    case through this same per-block-dict interface, without a separate
    code path. NOT a per-stage broadcast (QuantENet23_1's actual interface,
    one pair per one of 5 STAGE_NAMES groups) -- that needs a stage->block
    membership mapping (STAGE_MODULE_ATTRS in compression/hawq/config_23_1.py)
    this function deliberately doesn't duplicate; add a
    expand_stage_bits(stage_weight_bits, stage_act_bits, stage_module_attrs,
    block_names) here if/when a real caller needs it."""
    return (
        {b: weight_bit_width for b in block_names},
        {b: act_bit_width for b in block_names},
    )


def _make_block_shallow_stage(
    channels: int, n_ops: int, block_weight_bits: dict[str, int], block_act_bits: dict[str, int], dropout_p: float,
    name_prefix: str, slope_map: dict[str, float], use_dsc: bool = False, dsc_no_projection: bool = False,
    dsc_no_projection_context_only: bool = False, trainable_slope: bool = True,
) -> nn.Sequential:
    """Per-block generalization of QuantENet._make_shallow_stage's single
    shared (weight_bit_width, act_bit_width) pair -- looks up its own
    (w, a, negative_slope) triple per loop index via "<name_prefix>.<i>"
    instead. Faithful port of that method's dsc_no_projection branch
    (regular1/regular4/regular5 are never pattern-driven, unlike the
    context stages, so there's no dilation/asymmetric kwargs to thread
    here either -- matches every one of QuantENet.py's own
    _make_shallow_stage/the three per-block files' own _make_block_shallow_
    stage exactly)."""
    slope_map = slope_map or {}
    if dsc_no_projection and not dsc_no_projection_context_only:
        return nn.Sequential(*[
            QuantDSCNoProjectionBottleneck(
                channels, block_weight_bits[f"{name_prefix}.{i}"], block_act_bits[f"{name_prefix}.{i}"],
                dropout_p=dropout_p, negative_slope=slope_map.get(f"{name_prefix}.{i}"),
                trainable_slope=trainable_slope,
            )
            for i in range(n_ops)
        ])
    return nn.Sequential(*[
        QuantRegularBottleneck(
            channels, block_weight_bits[f"{name_prefix}.{i}"], block_act_bits[f"{name_prefix}.{i}"],
            dropout_p=dropout_p, use_dsc=use_dsc, negative_slope=slope_map.get(f"{name_prefix}.{i}"),
            trainable_slope=trainable_slope,
        )
        for i in range(n_ops)
    ])


def _make_block_context_stage(
    channels: int, n_ops: int, block_weight_bits: dict[str, int], block_act_bits: dict[str, int],
    name_prefix: str, slope_map: dict[str, float], context_pattern: str, use_dilated: bool = True,
    use_asymmetric: bool = False, use_dsc: bool = False, dsc_no_projection: bool = False,
    separable_dilated: bool = True, trainable_slope: bool = True,
) -> nn.Sequential:
    """Per-block generalization of QuantENet._make_context_stage's single
    shared pair -- faithful port of that method's full logic (pattern
    selection, dsc_no_projection's {"reg_bottleneck": True} bookend
    handling, asymmetric-kwargs guard, use_dilated gating), just looking up
    a fresh (w, a, negative_slope) triple per loop index via
    "<name_prefix>.<i>" instead of using one pair for the whole stage.

    Only the dsc_no_projection=False, use_asymmetric=False path (what
    S19/26_5_w24/5_6 all actually use) has been exercised against a real
    checkpoint so far -- see module docstring's SCOPE note."""
    if context_pattern not in VALID_CONTEXT_PATTERNS:
        raise ValueError(f"context_pattern must be one of {VALID_CONTEXT_PATTERNS}, got {context_pattern!r}.")
    if context_pattern == "dense_dilation":
        pattern = DENSE_DILATION_PATTERN
    elif context_pattern == "dense_dilation_reg_interleaved":
        pattern = DENSE_DILATION_REG_INTERLEAVED_PATTERN
    elif context_pattern == "dense_dilation_reg_interleaved_double_mid":
        pattern = DENSE_DILATION_REG_INTERLEAVED_DOUBLE_MID_PATTERN
    else:
        pattern = CONTEXT_STAGE_PATTERN

    slope_map = slope_map or {}

    if dsc_no_projection:
        ops = []
        for i in range(n_ops):
            kwargs = dict(pattern[i % len(pattern)])
            block_name = f"{name_prefix}.{i}"
            w, a = block_weight_bits[block_name], block_act_bits[block_name]
            if kwargs.pop("reg_bottleneck", False):
                ops.append(QuantRegularBottleneck(
                    channels, w, a, dropout_p=0.1, use_dsc=False,
                    negative_slope=slope_map.get(block_name), trainable_slope=trainable_slope,
                ))
                continue
            if kwargs.get("asymmetric", False):
                if use_asymmetric:
                    raise ValueError("dsc_no_projection is not defined for asymmetric bottlenecks -- set use_asymmetric=False.")
                kwargs = {}
            dilation = kwargs.get("dilation", 1)
            if dilation != 1 and not use_dilated:
                dilation = 1
            ops.append(QuantDSCNoProjectionBottleneck(
                channels, w, a, kernel_size=3, padding=dilation, dilation=dilation,
                dropout_p=0.1, negative_slope=slope_map.get(block_name), trainable_slope=trainable_slope,
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
        block_name = f"{name_prefix}.{i}"
        ops.append(QuantRegularBottleneck(
            channels, block_weight_bits[block_name], block_act_bits[block_name],
            dropout_p=0.1, use_dsc=use_dsc, separable_dilated=separable_dilated,
            negative_slope=slope_map.get(block_name), trainable_slope=trainable_slope, **kwargs,
        ))
    return nn.Sequential(*ops)


class CombinedQuantENet(nn.Module):
    """Parametrized replacement candidate for QuantENetS19Block/
    QuantENet26_5_w24/QuantENet5_6Block (and, via expand_uniform_bits,
    QuantENet23_1's uniform case) -- see module docstring for scope/status.

    prelu_variant is stored purely as a provenance/documentation field
    (matches every unified file's own usage of it -- it describes what the
    SOURCE FP32 checkpoint's activation actually was, used by the CALLER to
    decide whether leaky_slope_map represents a real trained per-block
    scalar vs. a post-hoc per-channel average; it does not itself gate any
    construction logic here, same as in every file being unified)."""

    def __init__(
        self, block_weight_bits: dict[str, int], block_act_bits: dict[str, int], *,
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
                "CombinedQuantENet's forward() only implements the upsample_conv decoder path "
                "(no pooling-indices plumbing) -- same scope every per-block file it unifies already has."
            )
        if dsc_no_projection_context_only and not dsc_no_projection:
            raise ValueError("dsc_no_projection_context_only narrows dsc_no_projection's scope -- meaningless without dsc_no_projection=True itself.")
        if len(channels) != 5 or len(bottlenecks_per_stage) != 5:
            raise ValueError("channels and bottlenecks_per_stage must each have 5 values (see ENet.py).")

        self.prelu_variant = prelu_variant
        self.block_names = block_names_for(bottlenecks_per_stage)
        missing_w = [b for b in self.block_names if b not in block_weight_bits]
        missing_a = [b for b in self.block_names if b not in block_act_bits]
        if missing_w or missing_a:
            raise ValueError(
                f"block_weight_bits/block_act_bits must have one entry per {len(self.block_names)} blocks -- "
                f"missing weight keys: {missing_w}, missing act keys: {missing_a}."
            )
        self.block_weight_bits = dict(block_weight_bits)
        self.block_act_bits = dict(block_act_bits)
        slope_map = leaky_slope_map or {}

        initial_ch, stage1_ch, stage23_ch, stage4_ch, stage5_ch = channels
        n_stage1, n_stage2, n_stage3, n_regular4, n_regular5 = bottlenecks_per_stage

        w, a = block_weight_bits["initial"], block_act_bits["initial"]
        self.initial = QuantInitialBlock(
            in_channels, initial_ch, w, a, negative_slope=slope_map.get("initial"), trainable_slope=trainable_slope,
        )

        w, a = block_weight_bits["down1"], block_act_bits["down1"]
        self.down1 = QuantDownsamplingBottleneck(
            initial_ch, stage1_ch, w, a, dropout_p=0.01, use_strided=use_strided,
            negative_slope=slope_map.get("down1"), trainable_slope=trainable_slope,
        )
        self.regular1 = _make_block_shallow_stage(
            stage1_ch, n_stage1, block_weight_bits, block_act_bits, 0.01, "regular1", slope_map,
            use_dsc=use_dsc, dsc_no_projection=dsc_no_projection,
            dsc_no_projection_context_only=dsc_no_projection_context_only, trainable_slope=trainable_slope,
        )

        w, a = block_weight_bits["down2"], block_act_bits["down2"]
        self.down2 = QuantDownsamplingBottleneck(
            stage1_ch, stage23_ch, w, a, dropout_p=0.1, use_strided=use_strided,
            negative_slope=slope_map.get("down2"), trainable_slope=trainable_slope,
        )
        self.stage2 = _make_block_context_stage(
            stage23_ch, n_stage2, block_weight_bits, block_act_bits, "stage2", slope_map, context_pattern,
            use_dilated=use_dilated, use_asymmetric=use_asymmetric, use_dsc=use_dsc,
            dsc_no_projection=dsc_no_projection, separable_dilated=separable_dilated, trainable_slope=trainable_slope,
        )
        self.stage3 = _make_block_context_stage(
            stage23_ch, n_stage3, block_weight_bits, block_act_bits, "stage3", slope_map, context_pattern,
            use_dilated=use_dilated, use_asymmetric=use_asymmetric, use_dsc=use_dsc,
            dsc_no_projection=dsc_no_projection, separable_dilated=separable_dilated, trainable_slope=trainable_slope,
        )

        # Decoder (regular4/regular5/up4/up5) is always plain QuantReLU,
        # regardless of leaky_slope_map -- same rule every unified file and
        # ENet.py itself already use (decoder hardcodes relu=True,
        # prelu_variant is an encoder/context-only axis). slope_map={} here
        # makes trainable_slope irrelevant too (no negative_slope means
        # _quant_block_act builds plain QuantReLU, see QuantENet.py).
        w, a = block_weight_bits["up4"], block_act_bits["up4"]
        self.up4 = QuantUpsamplingBottleneck(stage23_ch, stage4_ch, w, a)
        self.regular4 = _make_block_shallow_stage(
            stage4_ch, n_regular4, block_weight_bits, block_act_bits, 0.1, "regular4", {},
            use_dsc=use_dsc, dsc_no_projection=dsc_no_projection,
            dsc_no_projection_context_only=dsc_no_projection_context_only,
        )

        w, a = block_weight_bits["up5"], block_act_bits["up5"]
        self.up5 = QuantUpsamplingBottleneck(stage4_ch, stage5_ch, w, a)
        self.regular5 = _make_block_shallow_stage(
            stage5_ch, n_regular5, block_weight_bits, block_act_bits, 0.1, "regular5", {},
            use_dsc=use_dsc, dsc_no_projection=dsc_no_projection,
            dsc_no_projection_context_only=dsc_no_projection_context_only,
        )

        w = block_weight_bits["final"]
        self.final = qnn.QuantConvTranspose2d(
            stage5_ch, out_channels, kernel_size=2, stride=2, bias=True,
            weight_bit_width=w, weight_quant=Int8WeightPerTensorFloat,
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
        cls, checkpoint_path: str | Path, block_weight_bits: dict[str, int], block_act_bits: dict[str, int], *,
        leaky_slope_map: dict[str, float] | None = None, trainable_slope: bool = True, **kwargs,
    ) -> "CombinedQuantENet":
        """Same strict=False name+shape transfer every unified file's own
        from_pretrained already uses -- Brevitas quantizer-scale params
        aren't in a fresh FP32 checkpoint's state_dict, so a partial
        transfer is expected, not an error. **kwargs forwards every other
        __init__ arg (out_channels, channels, bottlenecks_per_stage,
        context_pattern, ...) -- there are too many architecture-shape
        knobs to repeat individually here without drifting out of sync
        with __init__'s own signature."""
        model = cls(
            block_weight_bits, block_act_bits, leaky_slope_map=leaky_slope_map,
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
            f"CombinedQuantENet.from_pretrained({checkpoint_path}): transferred {len(transferable)}/"
            f"{len(model_state_dict)} model keys ({n_shape_mismatch} shape mismatches, "
            f"{len(missing)} left uninitialized -- expected for Brevitas-only quantizer params)."
        )
        return model

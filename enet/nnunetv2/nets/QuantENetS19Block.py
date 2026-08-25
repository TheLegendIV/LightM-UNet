"""Hardcoded, per-BOTTLENECK-BLOCK-W/A-only Brevitas-quantized ENet for the
nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid
checkpoint ("S19") -- same architecture QuantENet23_1.py already targets
(23_1 is a QAT continuation warm-started FROM this exact checkpoint, see
compression/hawq/config_23_1.py's own docstring: "byte-identical to its
warm-start source, S19"), but at compression/hawq/block_sensitivity.py +
finn_block_costs.py + ilp_search.py's per-block granularity (one entry per
individual bottleneck, e.g. "stage2.7", not one shared value across all 12
stage2 blocks the way QuantENet23_1's per-STAGE search does).

Deliberately narrow, same philosophy as QuantENet23_1.py/QuantENet26_5_w24.py:
every architecture axis is hardcoded to S19's own architecture (see
compression/hawq/config_23_1.py / compression/results.csv's
nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid row).
The only knobs are block_weight_bits/block_act_bits (one entry per
BLOCK_NAMES below) and leaky_slope_map.

Unlike QuantENet26_5_w24.py: S19 IS genuinely prelu_variant="nonneg_block"
-- one real learnable non-negative scalar per BLOCK, extractable losslessly
via compression/post-quantization/extract_leaky_slope_map.py (already run
for this checkpoint: compression/post-quantization/slope_maps/
19_reginterleaved_separable_nonneg_block_double_mid.json). Passing that map
here builds the FINN-verified QuantDecomposedLeakyAct algebraic
decomposition at every mapped block instead of forcing plain QuantReLU (see
QuantENet.py's own QuantDecomposedLeakyAct docstring) -- a lossless
translation of the real trained activation, not an approximation, unlike
26_5_w24's forced-ReLU compromise. Decoder blocks (up4/regular4/up5/
regular5/final) are always plain QuantReLU regardless of the map, same rule
QuantENet23_1.py/QuantENet.py both already use (ENet.py's own decoder
hardcodes relu=True, prelu_variant is an encoder/context-only axis).

Reuses QuantENet.py's block classes (QuantInitialBlock,
QuantDownsamplingBottleneck, QuantRegularBottleneck, QuantUpsamplingBottleneck)
directly. Does NOT reuse QuantENet._make_shallow_stage/_make_context_stage
(both take one (weight_bit_width, act_bit_width) pair for an entire stage);
this file's own _make_block_shallow_stage/_make_block_context_stage look up
a fresh (w, a, negative_slope) triple per loop index instead, scoped to just
this architecture's own pattern (context_pattern=
"dense_dilation_reg_interleaved_double_mid", separable_dilated=True,
dsc_no_projection=False -- so the pattern's {"reg_bottleneck": True} bookend
slots are just a plain (non-dilated, non-factored) RegularBottleneck, same
as ENet.py's own _make_context_stage's "plain" projected loop already
treats them -- see that method's own comment on the sentinel being a
no-op outside dsc_no_projection=True).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat

from nnunetv2.nets.ENet import DENSE_DILATION_REG_INTERLEAVED_DOUBLE_MID_PATTERN, ENet
from nnunetv2.nets.QuantENet import (
    QuantDownsamplingBottleneck,
    QuantInitialBlock,
    QuantRegularBottleneck,
    QuantUpsamplingBottleneck,
)

# S19's exact architecture -- see compression/hawq/config_23_1.py (byte-
# identical) / compression/results.csv's own
# nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid row.
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 16, 32, 16, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 12, 12, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation_reg_interleaved_double_mid"
USE_DILATED = True
USE_ASYMMETRIC = False
USE_STRIDED = True
USE_DSC = False
SEPARABLE_DILATED = True
DSC_NO_PROJECTION = False
PRELU_VARIANT = "nonneg_block"

# One entry per individual bottleneck block -- 37 total, confirmed against
# compression/hawq/finn_block_costs_23_1.json's own 37 keys. proj2_to_3 is
# absent: stage2_channels == stage3_channels (32 == 32) here, so ENet.py
# builds it as nn.Identity() (nothing to quantize).
BLOCK_NAMES = (
    "initial", "down1",
    "regular1.0", "regular1.1", "regular1.2", "regular1.3",
    "down2",
    *[f"stage2.{i}" for i in range(12)],
    *[f"stage3.{i}" for i in range(12)],
    "up4", "regular4.0", "regular4.1", "up5", "regular5.0", "final",
)


def _make_block_shallow_stage(
    channels: int, n_ops: int, block_weight_bits: dict[str, int], block_act_bits: dict[str, int], dropout_p: float,
    name_prefix: str, slope_map: dict[str, float], trainable_slope: bool = False,
) -> nn.Sequential:
    """regular1/regular4/regular5: plain QuantRegularBottleneck repeats,
    each with its OWN (w, a, negative_slope) looked up by
    "<name_prefix>.<i>" -- the per-block generalization of
    QuantENet._make_shallow_stage's single shared pair."""
    return nn.Sequential(*[
        QuantRegularBottleneck(
            channels, block_weight_bits[f"{name_prefix}.{i}"], block_act_bits[f"{name_prefix}.{i}"],
            dropout_p=dropout_p, use_dsc=False, negative_slope=slope_map.get(f"{name_prefix}.{i}"),
            trainable_slope=trainable_slope,
        )
        for i in range(n_ops)
    ])


def _make_block_context_stage(
    channels: int, n_ops: int, block_weight_bits: dict[str, int], block_act_bits: dict[str, int], name_prefix: str,
    slope_map: dict[str, float], trainable_slope: bool = False,
) -> nn.Sequential:
    """stage2/stage3 under context_pattern=
    "dense_dilation_reg_interleaved_double_mid" (reg,2,4,8,16,reg,reg,2,4,8,
    16,reg -- 12 slots, see ENet.py's own pattern constant) -- the per-block
    generalization of QuantENet._make_context_stage's single shared pair.
    {"reg_bottleneck": True} slots are a no-op here (popped, same as
    QuantENet.py's own "plain"/projected loop already does, since
    dsc_no_projection=False for this architecture -- that sentinel only
    selects a different bottleneck CLASS under dsc_no_projection=True) --
    they just become a plain (dilation=1) RegularBottleneck like any other
    empty-kwargs slot."""
    ops = []
    for i in range(n_ops):
        kwargs = dict(DENSE_DILATION_REG_INTERLEAVED_DOUBLE_MID_PATTERN[i % len(DENSE_DILATION_REG_INTERLEAVED_DOUBLE_MID_PATTERN)])
        kwargs.pop("reg_bottleneck", False)
        block_name = f"{name_prefix}.{i}"
        ops.append(QuantRegularBottleneck(
            channels, block_weight_bits[block_name], block_act_bits[block_name],
            dropout_p=0.1, use_dsc=False, separable_dilated=SEPARABLE_DILATED,
            negative_slope=slope_map.get(block_name), trainable_slope=trainable_slope, **kwargs,
        ))
    return nn.Sequential(*ops)


class QuantENetS19Block(nn.Module):
    """Per-BLOCK-W/A quantized mirror of
    nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid.
    block_weight_bits/block_act_bits: dict[str, int], exactly one entry per
    BLOCK_NAMES key, each one of {2, 4, 8}. No defaulting for missing keys
    -- same "an incomplete dict is a bug in the caller" philosophy
    QuantENet23_1/QuantENet26_5_w24 already use, just at finer granularity."""

    def __init__(
        self, block_weight_bits: dict[str, int], block_act_bits: dict[str, int],
        leaky_slope_map: dict[str, float] | None = None,
    ):
        super().__init__()
        missing_w = [b for b in BLOCK_NAMES if b not in block_weight_bits]
        missing_a = [b for b in BLOCK_NAMES if b not in block_act_bits]
        if missing_w or missing_a:
            raise ValueError(
                f"block_weight_bits/block_act_bits must have one entry per {len(BLOCK_NAMES)} BLOCK_NAMES -- "
                f"missing weight keys: {missing_w}, missing act keys: {missing_a}."
            )
        self.block_weight_bits = dict(block_weight_bits)
        self.block_act_bits = dict(block_act_bits)
        slope_map = leaky_slope_map or {}

        initial_ch, stage1_ch, stage23_ch, stage4_ch, stage5_ch = CHANNELS
        n_stage1, n_stage2, n_stage3, n_regular4, n_regular5 = BOTTLENECKS_PER_STAGE

        # trainable_slope=True everywhere a real slope_map value gets used:
        # nonneg_block's own slope is architecturally a trainable-per-block
        # parameter by definition -- the fact that its CURRENT value came
        # from real FP32 training doesn't mean it should become permanently
        # frozen once QAT starts. Freezing it here was this class's
        # original (undefended) design choice; there's no principled reason
        # a QAT retrain shouldn't keep letting it adapt to quantization
        # noise, same as QuantENet5_6Block.py's own (more clearly necessary,
        # since that map is a known-wrong post-hoc average) fix. See
        # QuantDecomposedLeakyAct's own docstring for the mechanism and why
        # this has no FINN-export cost.
        w, a = block_weight_bits["initial"], block_act_bits["initial"]
        self.initial = QuantInitialBlock(
            IN_CHANNELS, initial_ch, w, a, negative_slope=slope_map.get("initial"), trainable_slope=True,
        )

        w, a = block_weight_bits["down1"], block_act_bits["down1"]
        self.down1 = QuantDownsamplingBottleneck(
            initial_ch, stage1_ch, w, a, dropout_p=0.01, use_strided=USE_STRIDED,
            negative_slope=slope_map.get("down1"), trainable_slope=True,
        )
        self.regular1 = _make_block_shallow_stage(
            stage1_ch, n_stage1, block_weight_bits, block_act_bits, 0.01, "regular1", slope_map, trainable_slope=True,
        )

        w, a = block_weight_bits["down2"], block_act_bits["down2"]
        self.down2 = QuantDownsamplingBottleneck(
            stage1_ch, stage23_ch, w, a, dropout_p=0.1, use_strided=USE_STRIDED,
            negative_slope=slope_map.get("down2"), trainable_slope=True,
        )
        self.stage2 = _make_block_context_stage(
            stage23_ch, n_stage2, block_weight_bits, block_act_bits, "stage2", slope_map, trainable_slope=True,
        )
        self.stage3 = _make_block_context_stage(
            stage23_ch, n_stage3, block_weight_bits, block_act_bits, "stage3", slope_map, trainable_slope=True,
        )

        # Decoder (regular4/regular5/up4/up5) is always plain QuantReLU,
        # regardless of leaky_slope_map -- same rule QuantENet23_1.py/
        # ENet.py both already use.
        w, a = block_weight_bits["up4"], block_act_bits["up4"]
        self.up4 = QuantUpsamplingBottleneck(stage23_ch, stage4_ch, w, a)
        self.regular4 = _make_block_shallow_stage(stage4_ch, n_regular4, block_weight_bits, block_act_bits, 0.1, "regular4", {})

        w, a = block_weight_bits["up5"], block_act_bits["up5"]
        self.up5 = QuantUpsamplingBottleneck(stage4_ch, stage5_ch, w, a)
        self.regular5 = _make_block_shallow_stage(stage5_ch, n_regular5, block_weight_bits, block_act_bits, 0.1, "regular5", {})

        w = block_weight_bits["final"]
        self.final = qnn.QuantConvTranspose2d(
            stage5_ch, OUT_CHANNELS, kernel_size=2, stride=2, bias=True,
            weight_bit_width=w, weight_quant=Int8WeightPerTensorFloat,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DECODER_TYPE is fixed to "upsample_conv" -- no pooling indices used.
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
        cls, checkpoint_path: str | Path, block_weight_bits: dict[str, int], block_act_bits: dict[str, int],
        leaky_slope_map: dict[str, float] | None = None,
    ) -> "QuantENetS19Block":
        """Same strict=False name+shape transfer as QuantENet23_1/
        QuantENet26_5_w24's own from_pretrained -- see either's docstring
        for why (Brevitas quantizer-scale params aren't in a fresh FP32
        checkpoint's state_dict)."""
        model = cls(block_weight_bits, block_act_bits, leaky_slope_map)
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
            f"QuantENetS19Block.from_pretrained({checkpoint_path}): transferred {len(transferable)}/"
            f"{len(model_state_dict)} model keys ({n_shape_mismatch} shape mismatches, "
            f"{len(missing)} left uninitialized -- expected for Brevitas-only quantizer params)."
        )
        return model


if __name__ == "__main__":
    torch.manual_seed(0)
    dummy = torch.zeros(1, 1, 512, 512)

    # 1. Build+forward with a genuinely heterogeneous per-block config
    # (homogeneous 8/4 split for a quick construction/shape smoke test --
    # the real ILP output is validated separately once
    # compression/hawq/block_bits_s19.json exists).
    block_weight_bits = {b: (4 if "stage2" in b or "stage3" in b else 8) for b in BLOCK_NAMES}
    block_act_bits = {b: (4 if "stage2" in b or "stage3" in b else 8) for b in BLOCK_NAMES}
    model = QuantENetS19Block(block_weight_bits, block_act_bits).eval()
    with torch.no_grad():
        out = model(dummy)
    out_t = out.value if hasattr(out, "value") else out
    assert out_t.shape == (1, OUT_CHANNELS, 512, 512), f"got {tuple(out_t.shape)}"
    print(f"Heterogeneous build+forward OK, output shape {tuple(out_t.shape)}")

    # 2. Per-block bit-widths landed where expected. stage2[0] is the
    # pattern's own reg-bookend slot ({"reg_bottleneck": True}, a no-op here
    # -- see module docstring) -- a PLAIN (non-separable-factored) conv, not
    # a Sequential, unlike stage2[1] (dilation=2, separable-dilated).
    stage2_0_w = model.stage2[0].conv.weight_quant.bit_width().item()
    assert stage2_0_w == 4, f"stage2.0 weight_bit_width expected 4, got {stage2_0_w}"
    stage2_1_w = model.stage2[1].conv[0].weight_quant.bit_width().item()
    assert stage2_1_w == 4, f"stage2.1 weight_bit_width expected 4, got {stage2_1_w}"
    initial_a = model.initial.act.act_quant.bit_width().item()
    assert initial_a == 8, f"initial act_bit_width expected 8, got {initial_a}"
    print("Per-block weight/act bit-width assignment verified.")

    # 3. Missing-block-key validation.
    try:
        QuantENetS19Block({"initial": 8}, block_act_bits)
        raise AssertionError("expected ValueError for an incomplete block_weight_bits dict, got none")
    except ValueError:
        pass
    print("Missing-block-key validation: OK")

    # 4. leaky_slope_map actually wires a QuantDecomposedLeakyAct (not plain
    # QuantReLU) at a mapped encoder block, and leaves an unmapped decoder
    # block on plain QuantReLU.
    from nnunetv2.nets.QuantENet import QuantDecomposedLeakyAct

    homogeneous_w = {b: 8 for b in BLOCK_NAMES}
    homogeneous_a = {b: 8 for b in BLOCK_NAMES}
    slope_map = {"initial": 0.5, "stage2.0": 0.25}
    slope_model = QuantENetS19Block(homogeneous_w, homogeneous_a, leaky_slope_map=slope_map)
    assert isinstance(slope_model.initial.act, QuantDecomposedLeakyAct), "initial should use the mapped slope"
    assert isinstance(slope_model.stage2[0].out_act, QuantDecomposedLeakyAct), "stage2.0 should use the mapped slope"
    assert not isinstance(slope_model.regular4[0].out_act, QuantDecomposedLeakyAct), "regular4 (decoder) must stay plain QuantReLU regardless of the map"
    print("leaky_slope_map wiring verified (encoder mapped block -> QuantDecomposedLeakyAct, decoder -> plain QuantReLU).")

    # 5. QONNX export smoke test.
    from brevitas.export import export_qonnx
    export_path = "/tmp/quant_enet_s19_block_hetero.onnx"
    export_qonnx(model, torch.randn(1, 1, 512, 512), export_path=export_path)
    import onnx
    onnx.checker.check_model(onnx.load(export_path))
    print(f"QONNX export OK and passed onnx.checker: {export_path}")

    # 6. Topology parity vs the real S19 architecture in ENet.py.
    quant = QuantENetS19Block(homogeneous_w, homogeneous_a)
    fp32 = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
        use_dilated=USE_DILATED, use_asymmetric=USE_ASYMMETRIC, use_strided=USE_STRIDED, use_dsc=USE_DSC,
        context_pattern=CONTEXT_PATTERN, separable_dilated=SEPARABLE_DILATED, dsc_no_projection=DSC_NO_PROJECTION,
        use_prelu=True, prelu_variant=PRELU_VARIANT,
    )
    for attr in ["regular1", "stage2", "stage3", "regular4", "regular5"]:
        fp32_len, quant_len = len(getattr(fp32, attr)), len(getattr(quant, attr))
        assert fp32_len == quant_len, f"topology drift in {attr}: ENet={fp32_len} QuantENetS19Block={quant_len}"
    assert isinstance(fp32.proj2_to_3, nn.Identity), "expected proj2_to_3 to be Identity (stage2/3 channels match) -- BLOCK_NAMES assumes this"
    print("Topology parity vs ENet.py (S19 config): OK")

    print("QuantENetS19Block self-test PASSED.")

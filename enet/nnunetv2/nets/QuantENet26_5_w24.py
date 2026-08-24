"""Hardcoded, per-BOTTLENECK-BLOCK-W/A-only Brevitas-quantized ENet for the
nnUNetTrainerENet_26_5_w24 checkpoint (compression/slurm/
stage_26_s5_6_probe_family_array.job's variant 5: S5-SeparableDense's own
recipe with stage2/3 width bumped 32 -> 24) -- built for compression/hawq/
block_sensitivity.py + finn_block_costs.py + ilp_search.py's per-block HAWQ
search (compression/hawq/block_bits_26_5_w24.json), one level finer than
QuantENet23_1.py's per-STAGE-group search: every one of this architecture's
29 individual bottleneck blocks (see compression/hawq/block_utils.py's
enumerate_blocks) gets its OWN independent weight_bit_width/act_bit_width,
not one shared value across an entire 5-way stage group.

Deliberately narrow, same philosophy as QuantENet23_1.py: every architecture
axis (channels, bottlenecks_per_stage, context_pattern, decoder_type,
use_asymmetric, use_dsc, separable_dilated) is hardcoded to 26_5_w24's own
architecture (see compression/hawq/config_26_5_w24.py / compression/
results.csv's nnUNetTrainerENet_26_5_w24 row), so an automated per-block
bit-width sweep can never accidentally build the wrong network. The only
knobs are block_weight_bits/block_act_bits (one entry per BLOCK_NAMES below).

PReLU: unlike 23_1 (prelu_variant="nonneg_block", one TRAINED learnable
scalar per BLOCK, cleanly extractable losslessly), 26_5_w24 is
prelu_variant="standard" -- REAL per-CHANNEL learnable PReLU (S5.6's own
family was never retrained with nonneg_block the way S13 was; see
compression/slurm/stage_13_separable_dense_nonneg_block_warmstart.job for
that retrain and why it exists). leaky_slope_map here is therefore NOT a
trained per-block scalar the way QuantENet23_1's/QuantENetS19Block's is --
it's produced by compression/post-quantization/extract_leaky_slope_map.py's
--prelu-variant standard path (ENet.py's own collect_prelu_block_means),
which POST-HOC averages each block's already-trained per-channel slopes
down to one mean float, with no retraining at all.

THIS IS A KNOWN, EMPIRICALLY BAD MOVE ON ITS OWN: compression/results.csv's
experiment_leaky_from_prelu_global/_perblock rows document that exact
post-hoc-average-with-no-retrain operation, on this exact S5.6 lineage,
collapsing dice from 0.7985 to ~0.38-0.41 (see compression/slurm/
stage_13_separable_dense_nonneg_block_warmstart.job's own header for the
same finding referenced from S13's side, and why S13 does a real
WARM-STARTED RETRAIN instead of a bare post-hoc average). A from_pretrained
call using this file's own averaged slope map is expected to reproduce
that same collapse, not a mild degradation -- the map is only sound to use
as a WARM-START initialization for a real QAT run (letting every weight
re-adapt to the now-frozen per-block slope over real training, the same
role S13's own ENET_NONNEG_BLOCK_INIT_MAP plays), never as a drop-in
PTQ-style deployment on its own. Passing leaky_slope_map=None (the default)
instead falls back to plain QuantReLU everywhere, same as before this
capability was added -- also a real mismatch, just a different, previously-
documented-survivable one (see compression/slurm/
stage_12_separable_dense_relu.job's own from-scratch ReLU retrain, ~0.026
dice below the PReLU original -- still not what a from_pretrained-only,
no-retrain forced-ReLU number would show either).

Reuses QuantENet.py's block classes (QuantInitialBlock,
QuantDownsamplingBottleneck, QuantRegularBottleneck, QuantUpsamplingBottleneck)
directly -- rather than duplicating their Brevitas wiring -- but does NOT
reuse its _make_shallow_stage/_make_context_stage staticmethods, since both
take one (weight_bit_width, act_bit_width) pair for an entire stage; this
file's own _make_block_shallow_stage/_make_block_context_stage below look up
a fresh pair per loop index instead, scoped to just the context_pattern
("dense_dilation") and flags (separable_dilated=True, everything else off)
this architecture actually uses -- not full parity with QuantENet.py's own
scoped subset (no dsc_no_projection/asymmetric/reg-bookend handling here,
none of it applies to this config).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat

from nnunetv2.nets.ENet import DENSE_DILATION_PATTERN, ENet
from nnunetv2.nets.QuantENet import (
    QuantDownsamplingBottleneck,
    QuantInitialBlock,
    QuantRegularBottleneck,
    QuantUpsamplingBottleneck,
)

# 26_5_w24's exact architecture -- see compression/hawq/config_26_5_w24.py /
# compression/results.csv's nnUNetTrainerENet_26_5_w24 row.
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 8, 24, 8, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation"
USE_DILATED = True
USE_ASYMMETRIC = False
USE_STRIDED = True
USE_DSC = False
SEPARABLE_DILATED = True
PRELU_VARIANT = "standard"  # see module docstring -- forced to plain QuantReLU here, not deployable losslessly

# One entry per individual bottleneck block (block_utils.enumerate_blocks's
# own naming convention: leaf modules get their own attr name, multi-block
# containers get "<attr>.<index>") -- 29 total, confirmed against
# compression/hawq/finn_block_costs_26_5_w24.json's own 29 keys. proj2_to_3
# is absent: stage2_channels == stage3_channels (24 == 24) here, so ENet.py
# builds it as nn.Identity() (nothing to quantize).
BLOCK_NAMES = (
    "initial", "down1",
    "regular1.0", "regular1.1", "regular1.2", "regular1.3",
    "down2",
    "stage2.0", "stage2.1", "stage2.2", "stage2.3", "stage2.4", "stage2.5", "stage2.6", "stage2.7",
    "stage3.0", "stage3.1", "stage3.2", "stage3.3", "stage3.4", "stage3.5", "stage3.6", "stage3.7",
    "up4", "regular4.0", "regular4.1", "up5", "regular5.0", "final",
)


def _make_block_shallow_stage(
    channels: int, n_ops: int, block_weight_bits: dict[str, int], block_act_bits: dict[str, int], dropout_p: float,
    name_prefix: str, slope_map: dict[str, float],
) -> nn.Sequential:
    """regular1/regular4/regular5: plain QuantRegularBottleneck repeats,
    each with its OWN (w, a, negative_slope) looked up by
    "<name_prefix>.<i>" -- the per-block generalization of
    QuantENet._make_shallow_stage's single shared pair. Scoped to just this
    config's own flags (use_dsc=False, dsc_no_projection=False -- neither
    applies to 26_5_w24)."""
    return nn.Sequential(*[
        QuantRegularBottleneck(
            channels, block_weight_bits[f"{name_prefix}.{i}"], block_act_bits[f"{name_prefix}.{i}"],
            dropout_p=dropout_p, use_dsc=False, negative_slope=slope_map.get(f"{name_prefix}.{i}"),
        )
        for i in range(n_ops)
    ])


def _make_block_context_stage(
    channels: int, n_ops: int, block_weight_bits: dict[str, int], block_act_bits: dict[str, int], name_prefix: str,
    slope_map: dict[str, float],
) -> nn.Sequential:
    """stage2/stage3 under context_pattern="dense_dilation" (every slot
    dilated, 2/4/8/16 repeated twice over 8 slots, separable_dilated=True --
    see ENet.py's DENSE_DILATION_PATTERN) -- the per-block generalization of
    QuantENet._make_context_stage's single shared pair, scoped to just this
    one pattern (26_5_w24 never uses reg-bookend/dsc_no_projection/
    asymmetric slots, unlike QuantENet23_1's dense_dilation_reg_interleaved_
    double_mid)."""
    ops = []
    for i in range(n_ops):
        kwargs = dict(DENSE_DILATION_PATTERN[i % len(DENSE_DILATION_PATTERN)])
        block_name = f"{name_prefix}.{i}"
        ops.append(QuantRegularBottleneck(
            channels, block_weight_bits[block_name], block_act_bits[block_name],
            dropout_p=0.1, use_dsc=False, separable_dilated=SEPARABLE_DILATED,
            negative_slope=slope_map.get(block_name), **kwargs,
        ))
    return nn.Sequential(*ops)


class QuantENet26_5_w24(nn.Module):
    """Per-BLOCK-W/A quantized mirror of nnUNetTrainerENet_26_5_w24.
    block_weight_bits/block_act_bits: dict[str, int], exactly one entry per
    BLOCK_NAMES key, each one of {2, 4, 8} (the candidate set compression/
    hawq/ilp_search.py chooses from). No defaulting for missing keys -- same
    "an incomplete dict is a bug in the caller" philosophy QuantENet23_1
    already uses, just at finer granularity."""

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

        w, a = block_weight_bits["initial"], block_act_bits["initial"]
        self.initial = QuantInitialBlock(IN_CHANNELS, initial_ch, w, a, negative_slope=slope_map.get("initial"))

        w, a = block_weight_bits["down1"], block_act_bits["down1"]
        self.down1 = QuantDownsamplingBottleneck(
            initial_ch, stage1_ch, w, a, dropout_p=0.01, use_strided=USE_STRIDED,
            negative_slope=slope_map.get("down1"),
        )
        self.regular1 = _make_block_shallow_stage(stage1_ch, n_stage1, block_weight_bits, block_act_bits, 0.01, "regular1", slope_map)

        w, a = block_weight_bits["down2"], block_act_bits["down2"]
        self.down2 = QuantDownsamplingBottleneck(
            stage1_ch, stage23_ch, w, a, dropout_p=0.1, use_strided=USE_STRIDED,
            negative_slope=slope_map.get("down2"),
        )
        self.stage2 = _make_block_context_stage(stage23_ch, n_stage2, block_weight_bits, block_act_bits, "stage2", slope_map)
        self.stage3 = _make_block_context_stage(stage23_ch, n_stage3, block_weight_bits, block_act_bits, "stage3", slope_map)

        # Decoder (regular4/regular5/up4/up5) is always plain QuantReLU,
        # regardless of leaky_slope_map -- same rule every other file in
        # this repo's QuantENet family uses (ENet.py's own decoder hardcodes
        # relu=True, prelu_variant is an encoder/context-only axis).
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
        # DECODER_TYPE is fixed to "upsample_conv" -- no pooling indices used
        # (mirrors QuantENet23_1.forward's own upsample_conv-only branch).
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
    ) -> "QuantENet26_5_w24":
        """Builds the quantized model, then transfers FP32 conv/BN weights
        from an nnU-Net ENet checkpoint by direct name+shape match
        (strict=False) -- same pattern as QuantENet23_1.from_pretrained (see
        its own docstring for why strict=False, not nnU-Net's generic
        load_pretrained_weights, is needed here). See module docstring's
        big warning: passing leaky_slope_map here (the post-hoc
        per-channel-average, NOT a trained scalar) does NOT re-adapt the
        transferred weights to it -- this repo's own prior experiment on
        this exact operation collapsed dice to ~0.38-0.41. Only sound as a
        QAT warm-start init, never as a standalone evaluation."""
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
            f"QuantENet26_5_w24.from_pretrained({checkpoint_path}, "
            f"leaky_slope_map={'set' if leaky_slope_map else 'None'}): transferred {len(transferable)}/"
            f"{len(model_state_dict)} model keys ({n_shape_mismatch} shape mismatches, "
            f"{len(missing)} left uninitialized -- expected for Brevitas-only quantizer params, plus every "
            f"real per-channel PReLU parameter regardless -- this checkpoint's own PReLU shape never matches "
            f"a QuantReLU/QuantDecomposedLeakyAct site's own params either way, see module docstring)."
        )
        return model


if __name__ == "__main__":
    torch.manual_seed(0)
    dummy = torch.zeros(1, 1, 512, 512)

    # 1. Build+forward with the REAL block_bits_26_5_w24.json assignment
    # (compression/hawq/ilp_search.py's actual per-block output).
    block_weight_bits = {
        "initial": 4, "down1": 4, "regular1.0": 4, "regular1.1": 4, "regular1.2": 4, "regular1.3": 4, "down2": 4,
        "stage2.0": 4, "stage2.1": 4, "stage2.2": 2, "stage2.3": 2, "stage2.4": 4, "stage2.5": 2, "stage2.6": 2,
        "stage2.7": 4, "stage3.0": 4, "stage3.1": 2, "stage3.2": 2, "stage3.3": 2, "stage3.4": 2, "stage3.5": 4,
        "stage3.6": 2, "stage3.7": 4, "up4": 4, "regular4.0": 4, "regular4.1": 4, "up5": 4, "regular5.0": 4, "final": 4,
    }
    block_act_bits = {
        "initial": 8, "down1": 8, "regular1.0": 4, "regular1.1": 8, "regular1.2": 8, "regular1.3": 8, "down2": 4,
        "stage2.0": 4, "stage2.1": 4, "stage2.2": 4, "stage2.3": 4, "stage2.4": 4, "stage2.5": 4, "stage2.6": 4,
        "stage2.7": 4, "stage3.0": 4, "stage3.1": 4, "stage3.2": 4, "stage3.3": 4, "stage3.4": 4, "stage3.5": 4,
        "stage3.6": 4, "stage3.7": 4, "up4": 2, "regular4.0": 4, "regular4.1": 4, "up5": 8, "regular5.0": 8, "final": 8,
    }
    model = QuantENet26_5_w24(block_weight_bits, block_act_bits).eval()
    with torch.no_grad():
        out = model(dummy)
    out_t = out.value if hasattr(out, "value") else out
    assert out_t.shape == (1, OUT_CHANNELS, 512, 512), f"got {tuple(out_t.shape)}"
    print(f"Real block_bits_26_5_w24 build+forward OK, output shape {tuple(out_t.shape)}")

    # 2. Per-block bit-widths actually landed where expected: stage2.2's
    # conv weight quantizer is 2-bit while stage2.0's is 4-bit (both inside
    # the SAME stage2 container -- exactly the per-block divergence a
    # per-stage-only search could never express).
    s202_w = model.stage2[2].conv[0].weight_quant.bit_width().item()
    assert s202_w == 2, f"stage2.2 weight_bit_width expected 2, got {s202_w}"
    s200_w = model.stage2[0].conv[0].weight_quant.bit_width().item()
    assert s200_w == 4, f"stage2.0 weight_bit_width expected 4, got {s200_w}"
    up4_a = model.up4.out_act.act_quant.bit_width().item()
    assert up4_a == 2, f"up4 act_bit_width expected 2, got {up4_a}"
    up5_a = model.up5.out_act.act_quant.bit_width().item()
    assert up5_a == 8, f"up5 act_bit_width expected 8, got {up5_a}"
    print("Per-block weight/act bit-width assignment verified (stage2.0 W=4 vs stage2.2 W=2 within the SAME stage; up4 A=2 vs up5 A=8).")

    # 3. Missing-block-key validation.
    try:
        QuantENet26_5_w24({"initial": 8}, block_act_bits)
        raise AssertionError("expected ValueError for an incomplete block_weight_bits dict, got none")
    except ValueError:
        pass
    print("Missing-block-key validation: OK")

    # 4. leaky_slope_map wiring: a mapped encoder block gets a
    # QuantDecomposedLeakyAct (not plain QuantReLU), an unmapped decoder
    # block stays plain QuantReLU regardless -- same check
    # QuantENetS19Block's own self-test makes, just against the post-hoc
    # per-channel-averaged map this file's own module docstring warns is
    # NOT safe to evaluate standalone (see extract_leaky_slope_map.py
    # --prelu-variant standard's real output, compression/post-quantization/
    # slope_maps/26_5_w24.json, for the actual averaged values).
    from nnunetv2.nets.QuantENet import QuantDecomposedLeakyAct

    slope_map = {"initial": 0.5, "stage2.0": 0.25}
    slope_model = QuantENet26_5_w24(homogeneous_w := {b: 8 for b in BLOCK_NAMES},
                                     homogeneous_a := {b: 8 for b in BLOCK_NAMES}, leaky_slope_map=slope_map)
    assert isinstance(slope_model.initial.act, QuantDecomposedLeakyAct), "initial should use the mapped slope"
    assert isinstance(slope_model.stage2[0].out_act, QuantDecomposedLeakyAct), "stage2.0 should use the mapped slope"
    assert not isinstance(slope_model.regular4[0].out_act, QuantDecomposedLeakyAct), "regular4 (decoder) must stay plain QuantReLU regardless of the map"
    assert not isinstance(slope_model.stage2[1].out_act, QuantDecomposedLeakyAct), "stage2.1 (unmapped) should stay plain QuantReLU"
    print("leaky_slope_map wiring verified (mapped block -> QuantDecomposedLeakyAct, unmapped/decoder -> plain QuantReLU).")

    # 5. QONNX export smoke test on the SAME heterogeneous config.
    from brevitas.export import export_qonnx
    export_path = "/tmp/quant_enet26_5_w24_hetero.onnx"
    export_qonnx(model, torch.randn(1, 1, 512, 512), export_path=export_path)
    import onnx
    onnx.checker.check_model(onnx.load(export_path))
    print(f"QONNX export OK and passed onnx.checker: {export_path}")

    # 6. Topology parity vs the real 26_5_w24 architecture in ENet.py, at a
    # homogeneous 8-bit config (topology must not depend on bit-width).
    homogeneous_w = {b: 8 for b in BLOCK_NAMES}
    homogeneous_a = {b: 8 for b in BLOCK_NAMES}
    quant = QuantENet26_5_w24(homogeneous_w, homogeneous_a)
    fp32 = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
        use_dilated=USE_DILATED, use_asymmetric=USE_ASYMMETRIC, use_strided=USE_STRIDED, use_dsc=USE_DSC,
        context_pattern=CONTEXT_PATTERN, separable_dilated=SEPARABLE_DILATED,
        use_prelu=True, prelu_variant=PRELU_VARIANT,
    )
    for attr in ["regular1", "stage2", "stage3", "regular4", "regular5"]:
        fp32_len, quant_len = len(getattr(fp32, attr)), len(getattr(quant, attr))
        assert fp32_len == quant_len, f"topology drift in {attr}: ENet={fp32_len} QuantENet26_5_w24={quant_len}"
    assert isinstance(fp32.proj2_to_3, nn.Identity), "expected proj2_to_3 to be Identity (stage2/3 channels match) -- BLOCK_NAMES assumes this"
    print("Topology parity vs ENet.py (26_5_w24 config): OK")

    print("QuantENet26_5_w24 self-test PASSED.")

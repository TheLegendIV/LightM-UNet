"""Hardcoded, per-BOTTLENECK-BLOCK-W/A-only Brevitas-quantized ENet for the
classic, ALREADY-TRAINED nnUNetTrainerENet_5_6_separable_dense_dilation
checkpoint ("S5.6" -- compression/results.csv's own row, stage=
5_arch_probe_pairs, dice=0.7985) -- built for compression/hawq/
block_sensitivity.py + finn_block_costs.py + ilp_search.py's per-block HAWQ
search (compression/hawq/block_bits_5_6.json), one level finer than a
per-STAGE-group search: every one of this architecture's 23 individual
bottleneck blocks (see compression/hawq/block_utils.py's enumerate_blocks)
gets its OWN independent weight_bit_width/act_bit_width.

Structurally near-identical to QuantENet26_5_w24.py (SAME bottlenecks_per_
stage=(4,8,8,2,1), SAME context_pattern="dense_dilation", SAME
separable_dilated=True, SAME decoder_type="upsample_conv" -- 26_5_w24 is
literally S5.6's own width-bumped/narrowed descendant, see compression/
hawq/config_26_5_w24.py's own docstring) -- differs ONLY in CHANNELS
(S5.6's own (4,16,32,16,4), not 26_5_w24's narrowed (4,8,24,8,4)). Every
architecture axis is hardcoded to S5.6's own architecture so an automated
per-block bit-width sweep can never accidentally build the wrong network.
The only knobs are block_weight_bits/block_act_bits (one entry per
BLOCK_NAMES below).

PReLU: S5.6 is prelu_variant="standard" -- REAL per-CHANNEL learnable
PReLU, confirmed directly against the checkpoint's own state_dict
(`initial.act.weight` shape (4,), `down1.out_act.weight` shape (16,), i.e.
one scalar per channel, not per block). leaky_slope_map here is therefore
NOT a trained per-block scalar (unlike QuantENetS19Block's real
nonneg_block one) -- it's produced by compression/post-quantization/
extract_leaky_slope_map.py's --prelu-variant standard path (ENet.py's own
collect_prelu_block_means), which POST-HOC averages each block's already-
trained per-channel slopes down to one mean float per block, with no
retraining at all (compression/post-quantization/slope_maps/
5_6_separable_dense_dilation.json -- confirmed identical to compression/
results.csv's own pre-existing experiment_leaky_from_prelu_perblock row's
recorded leaky_slope_map for this exact checkpoint).

THIS IS A KNOWN, EMPIRICALLY BAD MOVE AS A DROP-IN, NO-RETRAIN SUBSTITUTE
(same warning QuantENet26_5_w24.py's own docstring gives, this IS that
same S5.6 lineage): compression/results.csv's experiment_leaky_from_prelu_
global/_perblock rows document that exact post-hoc-average-with-no-retrain
operation collapsing dice from 0.7985 to ~0.38-0.41. The averaged map is
only sound to use as a WARM-START INITIALIZATION for a real QAT run.

UPDATE (2026-08-25): a first real QAT attempt using this averaged map as a
FROZEN slope (QuantDecomposedLeakyAct's original register_buffer-only
behavior) showed exactly the failure mode that history predicts, WORSE
than expected -- 21-27 real epochs on real HPC hardware with train_loss
falling steadily but Pseudo dice never moving off ~0.0 (compare the
already-successful QuantENetS19Block per-block QAT run, which broke 0.5+
per-class dice by epoch 15-20 under the same training setup). Every call
site below now passes trainable_slope=True (see QuantDecomposedLeakyAct's
own docstring) -- the averaged value is still the WARM-START
initialization (same map, same file), but is no longer frozen: gradients
can correct a block's slope away from the post-hoc average if the real
per-channel distribution it was averaged from was too wide for one shared
scalar to represent well. Whether this alone fixes the stuck-at-zero-dice
issue, or whether the per-block bit assignment itself is part of the
problem, is not yet established -- this is a genuine change to test, not
a confirmed fix.

Reuses QuantENet.py's block classes (QuantInitialBlock,
QuantDownsamplingBottleneck, QuantRegularBottleneck, QuantUpsamplingBottleneck)
directly -- rather than duplicating their Brevitas wiring -- but does NOT
reuse its _make_shallow_stage/_make_context_stage staticmethods (both take
one (weight_bit_width, act_bit_width) pair for an entire stage); this
file's own _make_block_shallow_stage/_make_block_context_stage below look
up a fresh pair per loop index instead, scoped to just this config's own
pattern/flags.
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

# S5.6's exact architecture -- see compression/hawq/config_5_6.py /
# compression/results.csv's nnUNetTrainerENet_5_6_separable_dense_dilation row.
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 16, 32, 16, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation"
USE_DILATED = True
USE_ASYMMETRIC = False
USE_STRIDED = True
USE_DSC = False
SEPARABLE_DILATED = True
PRELU_VARIANT = "standard"  # see module docstring -- forced to plain QuantReLU unless a slope map is passed

# One entry per individual bottleneck block -- 23 total, confirmed against
# compression/hawq/finn_block_costs_5_6.json's own 23 keys. proj2_to_3 is
# absent: stage2_channels == stage3_channels (32 == 32) here, so ENet.py
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
    name_prefix: str, slope_map: dict[str, float], trainable_slope: bool = False,
) -> nn.Sequential:
    """regular1/regular4/regular5: plain QuantRegularBottleneck repeats,
    each with its OWN (w, a, negative_slope) looked up by
    "<name_prefix>.<i>" -- the per-block generalization of
    QuantENet._make_shallow_stage's single shared pair. trainable_slope is
    a no-op wherever slope_map has no entry for a given block (decoder
    callers pass slope_map={} and never set trainable_slope=True, since
    negative_slope=None there makes it irrelevant either way)."""
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
    """stage2/stage3 under context_pattern="dense_dilation" (every slot
    dilated, 2/4/8/16 repeated over BOTTLENECKS_PER_STAGE's own 8-slot
    depth, separable_dilated=True -- see ENet.py's DENSE_DILATION_PATTERN)
    -- the per-block generalization of QuantENet._make_context_stage's
    single shared pair."""
    ops = []
    for i in range(n_ops):
        kwargs = dict(DENSE_DILATION_PATTERN[i % len(DENSE_DILATION_PATTERN)])
        block_name = f"{name_prefix}.{i}"
        ops.append(QuantRegularBottleneck(
            channels, block_weight_bits[block_name], block_act_bits[block_name],
            dropout_p=0.1, use_dsc=False, separable_dilated=SEPARABLE_DILATED,
            negative_slope=slope_map.get(block_name), trainable_slope=trainable_slope, **kwargs,
        ))
    return nn.Sequential(*ops)


class QuantENet5_6Block(nn.Module):
    """Per-BLOCK-W/A quantized mirror of
    nnUNetTrainerENet_5_6_separable_dense_dilation. block_weight_bits/
    block_act_bits: dict[str, int], exactly one entry per BLOCK_NAMES key,
    each one of {2, 4, 8}. No defaulting for missing keys -- same
    "an incomplete dict is a bug in the caller" philosophy every other
    QuantENet*Block.py in this repo uses."""

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
        # this class's own leaky_slope_map is ALWAYS the post-hoc per-channel
        # average (S5.6 is prelu_variant="standard", never nonneg_block --
        # see module docstring), a known-wrong-but-reasonable starting point,
        # never a genuinely-trained scalar the way QuantENetS19Block's own
        # map is -- so unlike that class, there is no reason to freeze it
        # here. See QuantDecomposedLeakyAct's own docstring for the full
        # rationale and why this has no FINN-export cost.
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
        # regardless of leaky_slope_map -- same rule every other QuantENet
        # file in this repo uses.
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
    ) -> "QuantENet5_6Block":
        """Builds the quantized model, then transfers FP32 conv/BN weights
        from an nnU-Net ENet checkpoint by direct name+shape match
        (strict=False) -- same pattern as QuantENet26_5_w24.from_pretrained.
        Passing leaky_slope_map here (the post-hoc per-channel-average, NOT
        a trained scalar) does NOT re-adapt the transferred weights to it --
        see module docstring's warning. Only sound as a QAT warm-start
        init, never as a standalone evaluation."""
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
            f"QuantENet5_6Block.from_pretrained({checkpoint_path}, "
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

    # 1. Build+forward with a homogeneous 8-bit config (real block_bits_5_6.json
    # is validated separately once it exists).
    homogeneous_w = {b: 8 for b in BLOCK_NAMES}
    homogeneous_a = {b: 8 for b in BLOCK_NAMES}
    model = QuantENet5_6Block(homogeneous_w, homogeneous_a).eval()
    with torch.no_grad():
        out = model(dummy)
    out_t = out.value if hasattr(out, "value") else out
    assert out_t.shape == (1, OUT_CHANNELS, 512, 512), f"got {tuple(out_t.shape)}"
    print(f"Homogeneous build+forward OK, output shape {tuple(out_t.shape)}")

    # 2. Missing-block-key validation.
    try:
        QuantENet5_6Block({"initial": 8}, homogeneous_a)
        raise AssertionError("expected ValueError for an incomplete block_weight_bits dict, got none")
    except ValueError:
        pass
    print("Missing-block-key validation: OK")

    # 3. leaky_slope_map wiring: a mapped encoder block gets a
    # QuantDecomposedLeakyAct (not plain QuantReLU), an unmapped decoder
    # block stays plain QuantReLU regardless.
    from nnunetv2.nets.QuantENet import QuantDecomposedLeakyAct

    slope_map = {"initial": 0.5, "stage2.0": 0.25}
    slope_model = QuantENet5_6Block(homogeneous_w, homogeneous_a, leaky_slope_map=slope_map)
    assert isinstance(slope_model.initial.act, QuantDecomposedLeakyAct), "initial should use the mapped slope"
    assert isinstance(slope_model.stage2[0].out_act, QuantDecomposedLeakyAct), "stage2.0 should use the mapped slope"
    assert not isinstance(slope_model.regular4[0].out_act, QuantDecomposedLeakyAct), "regular4 (decoder) must stay plain QuantReLU regardless of the map"
    assert not isinstance(slope_model.stage2[1].out_act, QuantDecomposedLeakyAct), "stage2.1 (unmapped) should stay plain QuantReLU"
    print("leaky_slope_map wiring verified (mapped block -> QuantDecomposedLeakyAct, unmapped/decoder -> plain QuantReLU).")

    # 4. QONNX export smoke test.
    from brevitas.export import export_qonnx
    export_path = "/tmp/quant_enet5_6_block_hetero.onnx"
    export_qonnx(model, torch.randn(1, 1, 512, 512), export_path=export_path)
    import onnx
    onnx.checker.check_model(onnx.load(export_path))
    print(f"QONNX export OK and passed onnx.checker: {export_path}")

    # 5. Topology parity vs the real S5.6 architecture in ENet.py.
    fp32 = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
        use_dilated=USE_DILATED, use_asymmetric=USE_ASYMMETRIC, use_strided=USE_STRIDED, use_dsc=USE_DSC,
        context_pattern=CONTEXT_PATTERN, separable_dilated=SEPARABLE_DILATED,
        use_prelu=True, prelu_variant=PRELU_VARIANT,
    )
    for attr in ["regular1", "stage2", "stage3", "regular4", "regular5"]:
        fp32_len, quant_len = len(getattr(fp32, attr)), len(getattr(model, attr))
        assert fp32_len == quant_len, f"topology drift in {attr}: ENet={fp32_len} QuantENet5_6Block={quant_len}"
    assert isinstance(fp32.proj2_to_3, nn.Identity), "expected proj2_to_3 to be Identity (stage2/3 channels match) -- BLOCK_NAMES assumes this"
    print("Topology parity vs ENet.py (S5.6 config): OK")

    print("QuantENet5_6Block self-test PASSED.")

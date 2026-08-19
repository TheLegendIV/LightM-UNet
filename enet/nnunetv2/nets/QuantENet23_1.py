"""Hardcoded, per-stage-W/A-only Brevitas-quantized ENet for the
nnUNetTrainerENet_23_1_s19_warmstart_4c checkpoint -- built for the HAWQ-
style per-stage weight/activation bit-width search (compression/hawq/), not
a general-purpose config surface like QuantENet.py.

Deliberately narrow: every architecture axis (channels, bottlenecks_per_
stage, context_pattern, decoder_type, use_asymmetric, use_dsc,
separable_dilated) is hardcoded to 23_1's own architecture (byte-identical
to its warm-start source, S19 -- see
compression/slurm/stage_23_1_s19_warmstart_4c.job and
compression/results.csv), so an automated bit-width sweep can never
accidentally build the wrong network. The only knobs are per-stage
weight_bit_width/act_bit_width (see STAGE_NAMES below) and leaky_slope_map.

leaky_slope_map: the 23_1 checkpoint trains prelu_variant="nonneg_block"
(one shared learnable non-negative PReLU scalar per block -- see ENet.py's
NonNegativePReLU/PReluVariant). Brevitas has no quantized PReLU op at all
(see QuantENet.py's module docstring), so the correct translation is the
same one QuantENet.py already builds for a frozen-LeakyReLU deployment
target: extract each block's own trained scalar and feed it as a FIXED
per-block slope into QuantDecomposedLeakyAct. Use
compression/post-quantization/extract_leaky_slope_map.py against the 23_1
checkpoint to produce this dict (its own docstring explains why this step
is not optional for a nonneg_block-trained model).

Reuses QuantENet.py's block classes (QuantInitialBlock,
QuantDownsamplingBottleneck, QuantUpsamplingBottleneck) and its
_make_shallow_stage/_make_context_stage staticmethods directly -- both
already take separate weight_bit_width/act_bit_width (see QuantENet.py's
module docstring) -- rather than duplicating ~800 lines of Brevitas wiring
into a second copy that would drift out of sync by hand.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat

from nnunetv2.nets.ENet import ENet
from nnunetv2.nets.QuantENet import (
    QuantDownsamplingBottleneck,
    QuantENet,
    QuantInitialBlock,
    QuantUpsamplingBottleneck,
)

# 23_1's exact architecture (== its warm-start source, S19) -- see
# compression/slurm/stage_23_1_s19_warmstart_4c.job /
# compression/results.csv's nnUNetTrainerENet_23_1_s19_warmstart_4c row.
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
DSC_NO_PROJECTION_CONTEXT_ONLY = False
PRELU_VARIANT = "nonneg_block"

# The 5-stage grouping the HAWQ per-stage search operates over -- matches
# ENET_CHANNELS/ENET_BOTTLENECKS convention and QuantENet.py's own
# bit-width call sites (see the HAWQ plan's stage table):
#   initial -> self.initial
#   stage1  -> self.down1, self.regular1
#   context -> self.down2, self.stage2, self.stage3
#   stage4  -> self.up4, self.regular4
#   stage5  -> self.up5, self.regular5, self.final
STAGE_NAMES = ("initial", "stage1", "context", "stage4", "stage5")


class QuantENet23_1(nn.Module):
    """Per-stage-W/A quantized mirror of
    nnUNetTrainerENet_23_1_s19_warmstart_4c. stage_weight_bits/
    stage_act_bits: dict[str, int], exactly one entry per STAGE_NAMES key,
    each one of {2, 4, 8} (the candidate set compression/hawq/ilp_search.py
    chooses from). No defaulting for missing keys -- this class only exists
    to be driven by a search result, so an incomplete dict is a bug in the
    caller, not something to silently paper over."""

    def __init__(
        self,
        stage_weight_bits: dict[str, int],
        stage_act_bits: dict[str, int],
        leaky_slope_map: dict[str, float] | None = None,
    ):
        super().__init__()
        missing_w = [s for s in STAGE_NAMES if s not in stage_weight_bits]
        missing_a = [s for s in STAGE_NAMES if s not in stage_act_bits]
        if missing_w or missing_a:
            raise ValueError(
                f"stage_weight_bits/stage_act_bits must have one entry per {STAGE_NAMES} -- "
                f"missing weight keys: {missing_w}, missing act keys: {missing_a}."
            )
        self.stage_weight_bits = dict(stage_weight_bits)
        self.stage_act_bits = dict(stage_act_bits)
        slope_map = leaky_slope_map or {}

        initial_ch, stage1_ch, stage23_ch, stage4_ch, stage5_ch = CHANNELS
        n_stage1, n_stage2, n_stage3, n_regular4, n_regular5 = BOTTLENECKS_PER_STAGE

        w, a = stage_weight_bits["initial"], stage_act_bits["initial"]
        self.initial = QuantInitialBlock(IN_CHANNELS, initial_ch, w, a, negative_slope=slope_map.get("initial"))

        w, a = stage_weight_bits["stage1"], stage_act_bits["stage1"]
        self.down1 = QuantDownsamplingBottleneck(
            initial_ch, stage1_ch, w, a, dropout_p=0.01, use_strided=USE_STRIDED,
            negative_slope=slope_map.get("down1"),
        )
        self.regular1 = QuantENet._make_shallow_stage(
            stage1_ch, n_stage1, w, a, dropout_p=0.01, use_dsc=USE_DSC,
            dsc_no_projection=DSC_NO_PROJECTION, dsc_no_projection_context_only=DSC_NO_PROJECTION_CONTEXT_ONLY,
            name_prefix="regular1", leaky_slope_map=slope_map,
        )

        w, a = stage_weight_bits["context"], stage_act_bits["context"]
        self.down2 = QuantDownsamplingBottleneck(
            stage1_ch, stage23_ch, w, a, dropout_p=0.1, use_strided=USE_STRIDED,
            negative_slope=slope_map.get("down2"),
        )
        self.stage2 = QuantENet._make_context_stage(
            stage23_ch, n_stage2, w, a, USE_DILATED, USE_ASYMMETRIC, USE_DSC,
            CONTEXT_PATTERN, DSC_NO_PROJECTION, SEPARABLE_DILATED, "stage2", slope_map,
        )
        self.stage3 = QuantENet._make_context_stage(
            stage23_ch, n_stage3, w, a, USE_DILATED, USE_ASYMMETRIC, USE_DSC,
            CONTEXT_PATTERN, DSC_NO_PROJECTION, SEPARABLE_DILATED, "stage3", slope_map,
        )

        # Decoder (regular4/regular5/up4/up5) is always plain QuantReLU,
        # regardless of leaky_slope_map -- same rule as QuantENet.py/ENet.py
        # (decoder hardcodes relu=True everywhere, prelu_variant unused there).
        w, a = stage_weight_bits["stage4"], stage_act_bits["stage4"]
        self.up4 = QuantUpsamplingBottleneck(stage23_ch, stage4_ch, w, a)
        self.regular4 = QuantENet._make_shallow_stage(
            stage4_ch, n_regular4, w, a, dropout_p=0.1, use_dsc=USE_DSC,
            dsc_no_projection=DSC_NO_PROJECTION, dsc_no_projection_context_only=DSC_NO_PROJECTION_CONTEXT_ONLY,
            name_prefix="regular4", leaky_slope_map=None,
        )

        w, a = stage_weight_bits["stage5"], stage_act_bits["stage5"]
        self.up5 = QuantUpsamplingBottleneck(stage4_ch, stage5_ch, w, a)
        self.regular5 = QuantENet._make_shallow_stage(
            stage5_ch, n_regular5, w, a, dropout_p=0.1, use_dsc=USE_DSC,
            dsc_no_projection=DSC_NO_PROJECTION, dsc_no_projection_context_only=DSC_NO_PROJECTION_CONTEXT_ONLY,
            name_prefix="regular5", leaky_slope_map=None,
        )
        self.final = qnn.QuantConvTranspose2d(
            stage5_ch, OUT_CHANNELS, kernel_size=2, stride=2, bias=True,
            weight_bit_width=w, weight_quant=Int8WeightPerTensorFloat,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DECODER_TYPE is fixed to "upsample_conv" -- no pooling indices used
        # (mirrors QuantENet.forward's upsample_conv branch, decoder_type
        # is not a per-instance axis here since it's a module-level constant).
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
        cls,
        checkpoint_path: str | Path,
        stage_weight_bits: dict[str, int],
        stage_act_bits: dict[str, int],
        leaky_slope_map: dict[str, float] | None = None,
    ) -> "QuantENet23_1":
        """Builds the quantized model, then transfers FP32 conv/BN weights
        from an nnU-Net ENet checkpoint by direct name+shape match
        (strict=False) -- same pattern as
        compression/post-quantization/ptq.py's transfer_fp32_weights (see
        its own docstring for why this needs strict=False rather than
        nnU-Net's generic load_pretrained_weights: Brevitas's
        `*.scaling_impl.value` quantizer-scale parameters are real
        nn.Parameters demanded by strict=True but never written by a fresh,
        uncalibrated model's own state_dict())."""
        model = cls(stage_weight_bits, stage_act_bits, leaky_slope_map)
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
            f"QuantENet23_1.from_pretrained({checkpoint_path}): transferred {len(transferable)}/"
            f"{len(model_state_dict)} model keys ({n_shape_mismatch} shape mismatches, "
            f"{len(missing)} left uninitialized -- expected for Brevitas-only quantizer params)."
        )
        return model


if __name__ == "__main__":
    torch.manual_seed(0)
    dummy = torch.zeros(1, 1, 512, 512)

    # 1. Build+forward with a genuinely heterogeneous per-stage config,
    # different mixes for weights vs activations.
    stage_weight_bits = {"initial": 8, "stage1": 8, "context": 4, "stage4": 4, "stage5": 8}
    stage_act_bits = {"initial": 8, "stage1": 4, "context": 4, "stage4": 8, "stage5": 8}
    model = QuantENet23_1(stage_weight_bits, stage_act_bits).eval()
    with torch.no_grad():
        out = model(dummy)
    out_t = out.value if hasattr(out, "value") else out
    assert out_t.shape == (1, OUT_CHANNELS, 512, 512), f"got {tuple(out_t.shape)}"
    print(f"Heterogeneous build+forward OK, output shape {tuple(out_t.shape)}")

    # 2. Per-stage bit-widths actually landed where expected: context's
    # conv weight quantizer is 4-bit while stage1's is 8-bit; stage1's
    # activation quantizer is 4-bit while stage4's is 8-bit.
    context_conv = model.stage2[0].conv
    context_w_bits = context_conv.weight_quant.bit_width().item()
    assert context_w_bits == 4, f"context weight_bit_width expected 4, got {context_w_bits}"
    stage1_conv = model.regular1[0].reduce[0]
    stage1_w_bits = stage1_conv.weight_quant.bit_width().item()
    assert stage1_w_bits == 8, f"stage1 weight_bit_width expected 8, got {stage1_w_bits}"
    stage1_act = model.regular1[0].out_act
    stage1_a_bits = stage1_act.act_quant.bit_width().item()
    assert stage1_a_bits == 4, f"stage1 act_bit_width expected 4, got {stage1_a_bits}"
    stage4_act = model.regular4[0].out_act
    stage4_a_bits = stage4_act.act_quant.bit_width().item()
    assert stage4_a_bits == 8, f"stage4 act_bit_width expected 8, got {stage4_a_bits}"
    print("Per-stage weight/act bit-width assignment verified (context W=4, stage1 W=8, stage1 A=4, stage4 A=8).")

    # 3. Missing-stage-key validation.
    try:
        QuantENet23_1({"initial": 8}, stage_act_bits)
        raise AssertionError("expected ValueError for an incomplete stage_weight_bits dict, got none")
    except ValueError:
        pass
    print("Missing-stage-key validation: OK")

    # 4. QONNX export smoke test on the SAME heterogeneous config -- confirms
    # Brevitas/onnx.checker accept a genuinely mixed per-stage bit-width
    # model, not just a homogeneous one (QuantENet.py's own self-test #3
    # only ever exports a homogeneous config).
    from brevitas.export import export_qonnx
    export_path = "/tmp/quant_enet23_1_hetero.onnx"
    export_qonnx(model, torch.randn(1, 1, 512, 512), export_path=export_path)
    import onnx
    onnx.checker.check_model(onnx.load(export_path))
    print(f"QONNX export OK and passed onnx.checker: {export_path}")

    # 5. Topology parity vs the real 23_1 architecture in ENet.py, at a
    # homogeneous 8-bit config (topology must not depend on bit-width).
    homogeneous_w = {s: 8 for s in STAGE_NAMES}
    homogeneous_a = {s: 8 for s in STAGE_NAMES}
    quant = QuantENet23_1(homogeneous_w, homogeneous_a)
    fp32 = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
        use_dilated=USE_DILATED, use_asymmetric=USE_ASYMMETRIC, use_strided=USE_STRIDED, use_dsc=USE_DSC,
        context_pattern=CONTEXT_PATTERN, separable_dilated=SEPARABLE_DILATED,
        use_prelu=True, prelu_variant=PRELU_VARIANT,
    )
    for attr in ["regular1", "stage2", "stage3", "regular4", "regular5"]:
        fp32_len, quant_len = len(getattr(fp32, attr)), len(getattr(quant, attr))
        assert fp32_len == quant_len, f"topology drift in {attr}: ENet={fp32_len} QuantENet23_1={quant_len}"
    print("Topology parity vs ENet.py (23_1 config): OK")

    print("QuantENet23_1 self-test PASSED.")

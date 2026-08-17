"""Export S19 (19_reginterleaved_separable_nonneg_block_double_mid) PTQ int8
checkpoint through the FINN-safe topology -- WITH REAL TRAINED WEIGHTS
transferred wherever the FINN-safe topology is structurally identical to
QuantENet.py's real architecture.

This is DIFFERENT from every prior FINN export in this repo
(finn_enet_prod_export.py, finn_export_s13_leaky_frozen.py,
finn_export_s5_dscnoproj_dense.py), all of which use fresh
torch.manual_seed(0) weights throughout -- fine for their purpose (FINN's
LUT/BRAM/DSP resource estimate and OOC synthesis result depend only on
architecture + bit-width + which nodes exist, NOT on weight values, an
already-established finding in this repo). Here we actually want the real
S19 PTQ checkpoint's learned weights carried through, since the goal this
time is closer to a real deployable/accuracy-relevant artifact, not just
another resource estimate.

THE COMPLICATION (why this can't be a simple checkpoint-load): the FINN-
safe topology genuinely differs from QuantENet.py's real architecture in
several places (same 4 deviations documented in every FINN*Block class
above and in finn_enet_prod_export.py's own module docstring):
  1. Initial block: FINN's conv directly produces the full initial_channels
     count; the real trained conv only produces initial_channels-in_channels
     (the rest comes from a concatenated, parameter-free MaxPool branch).
     Different conv weight SHAPE, and even where BN happens to match shape
     (out_channels), its statistics are tied to a completely different
     channel layout -- not reusable at all.
  2. Downsampling shortcut: FINN adds a NEW learned shortcut_proj (1x1 conv
     + BN); the real trained shortcut is parameter-free (MaxPool + zero-pad
     channels). No pretrained equivalent exists.
  3. Upsampling main path: FINN replaces the real trained path (1x1
     main_proj conv + parameter-free bilinear interpolate) with a single
     NEW learned conv-transpose (main_up + main_bn). No pretrained
     equivalent exists (different op entirely, main_proj's weights are not
     reusable for a conv-transpose).
  4. Final layer: real checkpoint has a trained bias; FINN's final layer
     requires bias=False (simplifies threshold streamlining) -- bias
     dropped, not transferred. The conv-transpose WEIGHT itself IS
     transferred (same shape).
  5. Every activation site: the real model uses QuantReLU (plain) or
     QuantDecomposedLeakyAct (fork + 2 muls + add) from QuantENet.py; the
     FINN-safe topology always uses THIS file's own DecomposedLeakyAct
     (pre_quant -> F.leaky_relu -> out_quant, no fork/add-join -- the
     "proven v3 shape", see finn_export_s13_leaky_frozen.py's own
     docstring for why the fork/add-join alternative fails FINN's
     step_create_dataflow_partition). Different classes, different
     internal quantizer parameter sets -- not transferable, even though
     they represent "the same" activation semantically and use the same
     real per-block negative_slope VALUE (which int32/float slope value to
     use IS transferred, just baked in as a constructor arg from the real
     slope map, not as a tensor parameter).

Everything else -- every conv/BN pair inside reduce/conv(_bn_act)/expand
across every regular/downsampling/upsampling bottleneck -- IS structurally
identical between the real model and this FINN-safe mirror (same kernel
sizes, channels, dilation, groups), so those ARE transferred directly via
transfer_weights() below. See that function's per-block helpers for the
exact index-level mapping (activation sub-modules are always skipped,
conv+BN sub-modules are always copied).

Net effect: the majority of the network's learned capacity (every
conv/BN weight) comes from the real S19 PTQ checkpoint; a well-defined,
explicitly-reported minority (initial block, 2x shortcut_proj, 2x main_up/
main_bn, every activation quantizer's scale, final bias) is fresh-
initialized. This means the resulting network's accuracy will NOT match
S19's real dice=0.6925 measurement out of the box -- if real accuracy
matters (not just resource/timing estimates, which don't care about
weight values at all), a short fine-tuning pass with this exact topology
would be needed afterward (see the "how to train a net that mimics this
fix" conversation this session -- same principle: build the constraint/
topology in from the start and train through it, don't bolt it on after
the fact and expect full fidelity).

Usage (run inside the pytorch training container):
    python hardware/finn_export_s19_double_mid.py

Output: hardware/outputs/finn_exports/quantEnet_s19_double_mid_int8.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_s19_double_mid_int8.onnx \\
        <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/
    docker exec <finn_container_id> python /home/thelegendiv/finn/notebooks/enet/finn_enet_build_decomposed_prelu.py \\
        quantEnet_s19_double_mid_int8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnunetv2.nets.QuantENet import QuantENet  # noqa: E402
from nnunetv2.nets.ENet import DENSE_DILATION_REG_INTERLEAVED_DOUBLE_MID_PATTERN  # noqa: E402

# Reuse the proven, generic FINN-safe building blocks from the S13 script
# rather than re-copying ~150 lines of identical class definitions --
# these classes are generic (not S13-specific), only S13's own
# assembly/context-stage-construction code (further down in that file)
# is specific to S13's config. See that file's own docstrings for why each
# class is shaped the way it is (FINN-compatibility rationale, proven
# end-to-end against the real streamlining/HW-conversion pipeline).
from finn_export_s13_leaky_frozen import (  # noqa: E402
    DecomposedLeakyAct, _EPSILON_SLOPE, _plain_relu_factory, _make_act_factory, _val,
    _ConstScaleInt8Act, _requant_factory, FINNInitialBlock, FINNDownsamplingBottleneck,
    FINNUpsamplingBottleneck, FINNRegularBottleneck, FINNRegularBottleneckSepDilated,
    _fast_cleanup, export_model,
)

DEFAULT_CHANNELS = (4, 16, 32, 16, 4)   # initial, s1, s23 (shared), s4, s5 -- same as S13, coincidentally
DEFAULT_BNECKS = (4, 12, 12, 2, 1)      # S19's own depth (deeper stage2/3 than S13's 4,8,8,2,1)
BIT_WIDTH = 8
DEFAULT_SLOPE_MAP_FILE = (
    REPO_ROOT / "compression" / "post-quantization" / "slope_maps"
    / "19_reginterleaved_separable_nonneg_block_double_mid.json"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data" / "nnUNet_results" / "Dataset509_ARCADE_1x1_4c"
    / "nnUNetTrainerENetQuant_19_double_mid_ptq_int8__nnUNetPlans__2d"
    / "fold_0" / "checkpoint_best.pth"
)


# ---------------------------------------------------------------------------
# FINN-safe model assembly (S19's own context_pattern, unlike S13's uniform
# dense_dilation)
# ---------------------------------------------------------------------------

def _make_context_stage(channels: int, n: int, bit_width: int, residual: bool,
                         slope_map: dict, name_prefix: str) -> nn.Sequential:
    """Builds one context stage (stage2/stage3) following
    DENSE_DILATION_REG_INTERLEAVED_DOUBLE_MID_PATTERN's exact 12-slot
    schedule (4 reg_bottleneck bookends incl. the doubled mid pair, 8
    dilated slots at 2/4/8/16 order twice) -- mirrors QuantENet.py's own
    _make_context_stage (dilation==1 slots -> FINNRegularBottleneck,
    dilation!=1 slots -> FINNRegularBottleneckSepDilated). Unlike
    finn_export_s13_leaky_frozen.py's own _make_dense_context_stage (S13's
    context_pattern=dense_dilation has EVERY slot dilated, so that version
    hardcodes FINNRegularBottleneckSepDilated unconditionally) -- not valid
    here, S19 has non-dilated reg_bottleneck slots mixed in, each of which
    must build as a plain FINNRegularBottleneck instead."""
    pattern = DENSE_DILATION_REG_INTERLEAVED_DOUBLE_MID_PATTERN
    blocks = []
    for i in range(n):
        slot = dict(pattern[i % len(pattern)])
        slot.pop("reg_bottleneck", False)
        dilation = slot.get("dilation", 1)
        block_name = f"{name_prefix}.{i}"
        act_factory = _make_act_factory(slope_map.get(block_name))
        if dilation != 1:
            blocks.append(FINNRegularBottleneckSepDilated(
                channels, bit_width, act_factory, dilation=dilation,
                dropout_p=0.1, residual=residual,
            ))
        else:
            blocks.append(FINNRegularBottleneck(
                channels, bit_width, act_factory, dropout_p=0.1, residual=residual,
            ))
    return nn.Sequential(*blocks)


class FINNQuantENetS19DoubleMid(nn.Module):
    """FINN-compatible mirror of QuantENet.py's real S19 recipe
    (19_reginterleaved_separable_nonneg_block_double_mid): channels=
    (4,16,32,16,4), bottlenecks=(4,12,12,2,1), context_pattern=
    dense_dilation_reg_interleaved_double_mid, decoder_type=upsample_conv,
    separable_dilated=1, asymmetric=0, strided=1, dsc=0, real per-block
    frozen leaky slopes (nonneg_block PReLU variant). See module docstring
    for the 4 FINN-topology deviations vs the real architecture."""

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
            raise ValueError(f"channels must be a 5-tuple, got {channels!r}.")
        if len(bottlenecks_per_stage) != 5:
            raise ValueError("bottlenecks_per_stage must have 5 values.")
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
        self.stage2 = _make_context_stage(c23, n2, bit_width, residual, slope_map, "stage2")
        self.stage3 = _make_context_stage(c23, n3, bit_width, residual, slope_map, "stage3")

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

        self.final = qnn.QuantConvTranspose2d(
            c5, out_channels, kernel_size=2, stride=2, bias=False,
            weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat,
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


# ---------------------------------------------------------------------------
# Real-weight transfer (the part every prior FINN export in this repo skips)
# ---------------------------------------------------------------------------

def _copy_indices(dst_seq, src_seq, indices: list[int], note: str, report: list[str]) -> None:
    """Copies dst_seq[i].load_state_dict(src_seq[i].state_dict()) for each i
    in indices -- used to transfer just the conv/BN sub-layers of a block
    while skipping activation-quantizer positions that differ in type
    between the real QuantENet.py model and this FINN-safe mirror."""
    n = 0
    for i in indices:
        dst_seq[i].load_state_dict(src_seq[i].state_dict())
        n += sum(p.numel() for p in src_seq[i].parameters())
    report.append(f"  [OK]    {note}: {n} params transferred")


def _fresh(note: str, report: list[str]) -> None:
    report.append(f"  [FRESH] {note}")


def _transfer_regular(dst, src, name: str, report: list[str]) -> None:
    """dst: FINNRegularBottleneck. src: QuantRegularBottleneck, dilation==1
    branch (src.conv is a single raw conv, not a Sequential; src.conv_bn_act
    wraps it: Sequential(raw_conv, BN, act) -- matches dst.conv's own
    Sequential(conv, BN, act) 1:1 at indices [0]=conv,[1]=BN)."""
    _copy_indices(dst.reduce, src.reduce, [0, 1], f"{name}.reduce (conv+BN)", report)
    _copy_indices(dst.conv, src.conv_bn_act, [0, 1], f"{name}.conv<-conv_bn_act (conv+BN)", report)
    _copy_indices(dst.expand, src.expand, [0, 1], f"{name}.expand (conv+BN)", report)
    _fresh(f"{name}: reduce/conv/expand trailing activations + .act/.requant "
           f"(QuantReLU/QuantDecomposedLeakyAct -> DecomposedLeakyAct, different classes)", report)


def _transfer_sep_dilated(dst, src, name: str, report: list[str]) -> None:
    """dst: FINNRegularBottleneckSepDilated. src: QuantRegularBottleneck,
    separable_dilated branch (src.conv is itself a 4-item Sequential:
    (k,1)conv, BN, act, (1,k)conv; src.conv_bn_act wraps it: Sequential
    (src.conv, BN, act) -- IDENTICAL structure to dst.conv_bn_act, since
    FINNRegularBottleneckSepDilated was written to mirror this branch
    block-for-block)."""
    _copy_indices(dst.reduce, src.reduce, [0, 1], f"{name}.reduce (conv+BN)", report)
    _copy_indices(dst.conv_bn_act[0], src.conv_bn_act[0], [0, 1, 3],
                  f"{name}.conv_bn_act[0] ((k,1)conv + BN + (1,k)conv)", report)
    dst.conv_bn_act[1].load_state_dict(src.conv_bn_act[1].state_dict())
    n_bn = sum(p.numel() for p in src.conv_bn_act[1].parameters())
    report.append(f"  [OK]    {name}.conv_bn_act[1] outer BN: {n_bn} params transferred")
    _copy_indices(dst.expand, src.expand, [0, 1], f"{name}.expand (conv+BN)", report)
    _fresh(f"{name}: inner+outer trailing activations + .act/.requant", report)


def _transfer_downsampling(dst, src, name: str, report: list[str]) -> None:
    """dst: FINNDownsamplingBottleneck. src: QuantDownsamplingBottleneck.
    S19 uses use_strided=1, so src.reduce is the 3-item strided-conv branch
    -- matches dst.reduce's own structure exactly."""
    _copy_indices(dst.reduce, src.reduce, [0, 1], f"{name}.reduce (conv+BN)", report)
    _copy_indices(dst.conv, src.conv, [0, 1], f"{name}.conv (conv+BN)", report)
    _copy_indices(dst.expand, src.expand, [0, 1], f"{name}.expand (conv+BN)", report)
    _fresh(f"{name}.shortcut_proj (NEW learned 1x1 projection -- real model's downsampling "
           f"shortcut is parameter-free MaxPool+zero-pad, no projection conv at all)", report)
    _fresh(f"{name}: reduce/conv/expand trailing activations + .act/.requant", report)


def _transfer_upsampling(dst, src, name: str, report: list[str]) -> None:
    """dst: FINNUpsamplingBottleneck. src: QuantUpsamplingBottleneck."""
    _copy_indices(dst.reduce, src.reduce, [0, 1], f"{name}.reduce (conv+BN)", report)
    _copy_indices(dst.up, src.up, [0, 1], f"{name}.up (conv_transpose+BN)", report)
    _copy_indices(dst.expand, src.expand, [0, 1], f"{name}.expand (conv+BN)", report)
    _fresh(f"{name}.main_up/main_bn (NEW learned conv-transpose replacing the real model's "
           f"main_proj 1x1-conv + parameter-free bilinear interpolate -- different op entirely, "
           f"main_proj's weights are not reusable for a conv-transpose)", report)
    _fresh(f"{name}: reduce/up/expand trailing activations + .main_act/.act/.requant", report)


def transfer_weights(dst: FINNQuantENetS19DoubleMid, src: QuantENet) -> list[str]:
    """Transfers every structurally-identical conv/BN pair from the real,
    checkpoint-loaded `src` (QuantENet) into the FINN-safe `dst`
    (FINNQuantENetS19DoubleMid), leaving the topology-mismatched components
    (see module docstring) at their fresh-init values. Returns a report
    (list of lines) describing exactly what was transferred vs left fresh --
    print/inspect this before trusting the result."""
    report: list[str] = []

    _fresh("initial.conv+bn (real conv produces initial_channels-in_channels then concats a "
           "parameter-free MaxPool branch to reach initial_channels; FINN-safe conv directly "
           "produces the full initial_channels -- different shape AND different per-channel "
           "semantics even where BN shape happens to coincide, not reusable)", report)

    _transfer_downsampling(dst.down1, src.down1, "down1", report)
    for i, (d, s) in enumerate(zip(dst.regular1, src.regular1)):
        _transfer_regular(d, s, f"regular1.{i}", report)

    _transfer_downsampling(dst.down2, src.down2, "down2", report)

    pattern = DENSE_DILATION_REG_INTERLEAVED_DOUBLE_MID_PATTERN
    for stage_name, dst_stage, src_stage in (
        ("stage2", dst.stage2, src.stage2), ("stage3", dst.stage3, src.stage3),
    ):
        for i, (d, s) in enumerate(zip(dst_stage, src_stage)):
            dilation = pattern[i % len(pattern)].get("dilation", 1)
            name = f"{stage_name}.{i}"
            if dilation != 1:
                _transfer_sep_dilated(d, s, name, report)
            else:
                _transfer_regular(d, s, name, report)

    _transfer_upsampling(dst.up4, src.up4, "up4", report)
    for i, (d, s) in enumerate(zip(dst.regular4, src.regular4)):
        _transfer_regular(d, s, f"regular4.{i}", report)

    _transfer_upsampling(dst.up5, src.up5, "up5", report)
    for i, (d, s) in enumerate(zip(dst.regular5, src.regular5)):
        _transfer_regular(d, s, f"regular5.{i}", report)

    dst.final.weight.data.copy_(src.final.weight.data)
    report.append(f"  [OK]    final.weight (conv_transpose kernel): {src.final.weight.numel()} params transferred")
    _fresh("final.bias (real model has a trained bias; FINN-safe final layer requires "
           "bias=False for threshold streamlining -- dropped, not transferable)", report)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--slope-map-file", default=str(DEFAULT_SLOPE_MAP_FILE))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--input-hw", type=int, nargs=2, default=(64, 64), metavar=("H", "W"))
    parser.add_argument("--bit-width", type=int, default=BIT_WIDTH)
    parser.add_argument("--no-residuals", action="store_true", help="Remove all residual shortcuts.")
    args = parser.parse_args()

    h, w = args.input_hw
    if h % 8 != 0 or w % 8 != 0:
        parser.error(f"--input-hw {h}x{w}: both dims must be divisible by 8.")

    with open(args.slope_map_file) as f:
        leaky_slope_map = json.load(f)

    print("\n=== 1. Building + loading the REAL trained QuantENet (S19 config) ===")
    real_model = QuantENet(
        in_channels=args.in_channels, out_channels=args.out_channels,
        channels=DEFAULT_CHANNELS, bottlenecks_per_stage=DEFAULT_BNECKS,
        decoder_type="upsample_conv", use_dilated=True, use_asymmetric=False,
        use_strided=True, use_dsc=False,
        weight_bit_width=args.bit_width, act_bit_width=args.bit_width,
        context_pattern="dense_dilation_reg_interleaved_double_mid",
        separable_dilated=True, leaky_slope_map=leaky_slope_map,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["network_weights"]
    new_state_dict = {
        (key[7:] if key.startswith("module.") and key not in real_model.state_dict() else key): value
        for key, value in state_dict.items()
    }
    missing, unexpected = real_model.load_state_dict(new_state_dict, strict=True)
    real_model.eval()
    print(f"  Loaded {args.checkpoint} (epoch {checkpoint.get('current_epoch')}) -- missing={missing} unexpected={unexpected}")

    print("\n=== 2. Building the FINN-safe mirror + transferring real weights ===")
    residual = not args.no_residuals
    finn_model = FINNQuantENetS19DoubleMid(
        in_channels=args.in_channels, out_channels=args.out_channels,
        channels=DEFAULT_CHANNELS, bottlenecks_per_stage=DEFAULT_BNECKS,
        bit_width=args.bit_width, residual=residual, leaky_slope_map=leaky_slope_map,
    ).eval()

    report = transfer_weights(finn_model, real_model)
    print("\n".join(report))
    n_ok = sum(1 for line in report if "[OK]" in line)
    n_fresh = sum(1 for line in report if "[FRESH]" in line)
    print(f"\n  {n_ok} components transferred from the real checkpoint, {n_fresh} components fresh-initialized.")
    print("  (Real accuracy will NOT match the checkpoint's own dice until/unless fine-tuned "
          "with this exact FINN-safe topology -- see module docstring.)")

    print("\n=== 3. Forward-pass sanity check + QONNX export ===")
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1
    with torch.no_grad():
        out = finn_model(dummy)
    assert out.shape[2:] == (h, w), f"output HxW {tuple(out.shape[2:])} != input ({h},{w})"
    assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
    print(f"  forward OK: output shape {tuple(out.shape)}")

    suffix = "_no_res" if not residual else ""
    name = f"quantEnet_s19_double_mid_int{args.bit_width}{suffix}"
    export_model(finn_model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")
    print(f"  docker exec <finn_container_id> python /home/thelegendiv/finn/notebooks/enet/finn_enet_build_decomposed_prelu.py {name}")


if __name__ == "__main__":
    main()

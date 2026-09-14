"""Export a FINN-compatible, PER-LAYER HAWQ mirror of
nnUNetTrainerENet_12_dense_relu_warmstart150ep (dense KxK dilated context,
SEPARABLE_DILATED=False, plain ReLU, alpha=0.25 per-layer HAWQ bit-width
assignment -- see compression/hawq/config_12_dense_relu_warmstart150ep.py
and compression/hawq/artifacts/12_dense_relu_warmstart150ep_ILP_outputs_
perlayer_forcedsp_lut70/layer_bits_SITES_12_dense_relu_warmstart150ep_joint_
alpha0.25_candidatebits468_forcedsp_lut70.json -- NOT the sibling
layer_bits_folding_*.json, whose layer_weight_bits/layer_act_bits use a
DIFFERENT, coarser conv-input-activation schema; see
memories/repo/finn_12_dense_relu_alpha025_perlayer.md for the full
distinction).

DUMMY variant: fresh (torch.manual_seed(0)) conv/BN weights throughout --
validates the graph/export pipeline structurally, and validates that the
real per-layer bits JSON's key set lines up exactly with this exact
architecture's real site names, before loading the real ft15ep QAT
checkpoint (see finn_export_12_dense_relu_warmstart150ep_alpha025_trained.py,
which reuses this file's FINNInitialBlockConcat/FINNDownsamplingBottleneck/
FINNUpsamplingBottleneck/LayerQuantEnetFINN/load_layer_bits verbatim).

Since USE_PRELU=False for this config, the REAL LayerQuantENet's own
activation modules devolve to plain qnn.QuantReLU (`_quant_block_act` with
negative_slope=None -- see QuantENet.py) and its residual_add is a real,
already-FINN-safe qnn.QuantEltwiseAdd (single shared input_quant across
both operands, per LayerQuantRegularBottleneck's own docstring). This means
LayerQuantRegularBottleneck (imported directly from
enet/nnunetv2/nets/LayerQuantENet.py, via the same
_make_layer_shallow_stage/_make_layer_context_stage assembly helpers that
file itself uses) is reused UNMODIFIED here -- no FINN-specific
reimplementation needed. LayerQuantInitialBlock needs one (see point 0
below). LayerQuantDSCNoProjectionBottleneck is never instantiated
(USE_DSC=False globally for this config).

4 substitutions are needed vs. the real LayerQuantENet -- see memories/repo/
finn_12_dense_relu_alpha025_perlayer.md for the full derivation/verification
of each:

0. FINNInitialBlockConcat: real LayerQuantInitialBlock does
   `torch.cat([self.conv(x), self.pool(x)], dim=1)` with NO shared quantizer
   forcing the conv branch and the pool branch onto the same (scale,
   zero_point) pair first. Brevitas's QuantTensor-aware `cat` requires
   coherent quant params across all its inputs to stay in the QuantTensor
   representation, and silently falls back to a plain float tensor when
   they differ -- confirmed against the real S12 preamble build's
   step_enet_convert_to_hw.onnx: Concat_0's inputs are annotated FLOAT32,
   not integer, which is why Concat/the BN-affine Mul+Add/MaxPool never
   lower to FINN HW nodes (every Infer*Layer HW-conversion pass hard-
   requires an integer input DataType). Fix, and ONLY departure from the
   real block: one new shared `branch_quant` (qnn.QuantIdentity instance,
   no HAWQ site -> bit-width matched to the chain, see
   FINNInitialBlockConcat) applied to BOTH self.conv(x) and
   self.pool(x) right before the cat, forcing them onto one common scale so
   the cat stays integer-annotated all the way through. This is the ONLY
   new rounding point added; self.bn is NOT hand-folded into self.conv --
   once Concat sees coherent integer inputs it converts to StreamingConcat,
   and the standard BatchNormToAffine + AbsorbMulIntoMultiThreshold /
   AbsorbAddIntoMultiThreshold streamlining (already wired into
   step_enet_streamline, and already relied on for every OTHER conv+BN+act
   instance in this network) absorbs self.bn into `act`'s thresholds for
   free, exactly like everywhere else. conv/pool/bn/act are otherwise
   identical (same modules, same real per-site weights, same site names) to
   LayerQuantInitialBlock.
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
   literal 0.0/1.0 (confirmed empirically -- MatMul_0/MatMul_1 stayed
   FLOAT32-annotated and never lowered to MVAU), so an explicit Quant
   wrapper is required -- same fix already applied to `main_up` below.
2. FINNUpsamplingBottleneck: real up4/up5 main path is main_proj (real,
   trained, transferred as-is) -> F.interpolate(bilinear,
   align_corners=False) (LayerQuantENet always passes indices=None in
   decoder_type="upsample_conv" mode -- confirmed NEVER MaxUnpool2d).
   Resize/interpolate isn't FINN-synthesizable, so this substitutes a fixed
   (frozen) depthwise ConvTranspose2d(kernel=4, stride=2, padding=1,
   groups=channels) with the classic bilinear kernel
   outer([1,3,3,1]/4, [1,3,3,1]/4) -- verified in
   hardware/verify_bilinear_kernel.py to match F.interpolate EXACTLY on
   every interior pixel; only the outermost 1px border ring differs
   (zero-pad vs. the real op's edge-replicate/clamp assumption at the
   image border) -- decided against the edge-replicate-Pad node needed for
   full bit-exactness (unverified FINN support, real risk of failing like
   Resize does).
3. `final` bias: real LayerQuantENet's `final` is bias=True with a plain
   (unquantized) float bias Add -- fine on GPU/CPU but not FINN-HW-
   convertible (MVAU has no bias input at all, and InferChannelwiseLinearLayer
   requires the constant to be exactly integer-valued, confirmed from FINN's
   own convert_to_hw_layers.py source). Fix: bias_quant=Int32Bias, which
   quantizes the bias into the MatMul accumulator's own implicit scale
   (input_scale * weight_scale), making it exact-integer and letting the
   resulting Add lower to a real ChannelwiseOp HW node instead of a CPU-
   side post-processing step.

Usage (run inside the pytorch training container, e.g. `lightm_pytorch`):
    docker exec lightm_pytorch python /workspace/LightM-UNet/hardware/finn_export_12_dense_relu_warmstart150ep_alpha025_dummy.py

Output: hardware/outputs/finn_exports/quantEnet_12_dense_relu_warmstart150ep_alpha025_dummy_int8.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_12_dense_relu_warmstart150ep_alpha025_dummy_int8.onnx \\
        <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int32Bias

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.QuantENet import _quant_conv2d, _quant_act, _quant_block_act  # noqa: E402
from nnunetv2.nets.LayerQuantENet import (  # noqa: E402
    _make_layer_shallow_stage, _make_layer_context_stage,
    _local_single, layer_names_for,
)
from finn_export_s13_leaky_frozen import export_model  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"
CHANNELS = (4, 16, 32, 16, 4)          # initial, s1, s23 (shared), s4, s5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation"
DEFAULT_BITS_FILE = (
    REPO_ROOT / "compression" / "hawq" / "artifacts"
    / "12_dense_relu_warmstart150ep_ILP_outputs_perlayer_forcedsp_lut70"
    / "layer_bits_SITES_12_dense_relu_warmstart150ep_joint_alpha0.25_candidatebits468_forcedsp_lut70.json"
)
FALLBACK_BITS = 6  # user decision -- any site missing from the real JSON falls back to 6, not 8


def load_layer_bits(
    bits_file: Path, weight_names: tuple[str, ...], act_names: tuple[str, ...], fallback: int = FALLBACK_BITS,
) -> tuple[dict[str, int], dict[str, int]]:
    """Loads a layer_bits_SITES_*.json (fine-grained per-activation-SITE
    schema -- NOT the sibling layer_bits_folding_*.json, see module
    docstring) and expands it to a complete dict covering every real site
    this exact architecture needs, falling back to `fallback` for any site
    missing from the file (never crashes on a missing key, unlike
    LayerQuantENet's own strict validation)."""
    with open(bits_file) as f:
        raw = json.load(f)
    raw_w, raw_a = raw["layer_weight_bits"], raw["layer_act_bits"]

    missing_w = [n for n in weight_names if n not in raw_w]
    missing_a = [n for n in act_names if n not in raw_a]
    if missing_w or missing_a:
        print(f"  WARNING: {len(missing_w)} weight site(s) / {len(missing_a)} act site(s) missing from "
              f"{bits_file.name}, using fallback={fallback}: {missing_w + missing_a}")

    layer_weight_bits = {n: raw_w.get(n, fallback) for n in weight_names}
    layer_act_bits = {n: raw_a.get(n, fallback) for n in act_names}
    return layer_weight_bits, layer_act_bits


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
    FINN's tidy-up does NOT infer an integer DataType for it (confirmed
    empirically against the real S12 preamble build's
    step_enet_convert_to_hw.onnx -- MatMul_0/MatMul_1's weight tensors stay
    annotated FLOAT32), so InferQuantizedMatrixVectorActivation's
    `wdt.is_integer()` gate fails and the conv never lowers to an MVAU --
    same failure mode already found and fixed for `main_up`'s bilinear
    kernel (see _bilinear_kernel_conv_transpose below). Fix: wrap in
    qnn.QuantConv2d (INT8 weight quant) so an explicit Quant node forces a
    real integer weight DataType, then overwrite+freeze the weight exactly
    as before."""
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


def _bilinear_kernel_conv_transpose(channels: int) -> qnn.QuantConvTranspose2d:
    """Fixed (frozen, non-learned) ConvTranspose2d reproducing 2x bilinear
    upsampling (align_corners=False) exactly on interior pixels -- see
    module docstring point 2 and hardware/verify_bilinear_kernel.py.

    A plain (non-quantized) nn.ConvTranspose2d here would export as a bare
    float ONNX ConvTranspose node: FINN's step_enet_convert_to_hw only
    lowers Quant-wrapped ops into HW dataflow nodes, so a plain float op
    stays a non-fpgadataflow node and gets silently EXCLUDED from every
    StreamingDataflowPartition -- confirmed empirically (2 bare
    "ConvTranspose" nodes found outside all partitions when this used
    nn.ConvTranspose2d), meaning this upsample would never actually be
    synthesized into the accelerator. Fix: wrap in qnn.QuantConvTranspose2d
    (INT8 weight quant, same class already used for the real `up.0` deconv)
    so it lowers to a real MVAU node like everything else, then overwrite
    its weight with the exact fixed kernel and freeze it (requires_grad=
    False) -- same "quantized but not learned" pattern as _padded_identity_
    conv, just needing an explicit Quant wrapper here because the bilinear
    kernel's values (1,3,9)/16 aren't literal integers the way the
    padded-identity conv's {0,1} values are (QONNX's tidy-up can infer an
    INT datatype for exact-integer float weights even without a Quant
    wrapper, which is how the identity conv "gets away with" staying plain
    -- fractional weights have no such fallback).

    Built DENSE (groups=1, weight (channels,channels,4,4), zero off the
    per-channel diagonal) rather than depthwise (groups=channels): this
    architecture has 0 VVAU/depthwise nodes everywhere else (SEPARABLE_
    DILATED=False, USE_DSC=False), so staying dense reuses the same
    already-proven MVAU/ConvTranspose HW lowering path as `up.0` instead of
    introducing an untested depthwise-ConvTranspose-as-VVAU code path."""
    up = qnn.QuantConvTranspose2d(
        channels, channels, kernel_size=4, stride=2, padding=1, groups=1, bias=False,
        weight_bit_width=8, weight_quant=Int8WeightPerTensorFloat,
    )
    k1d = torch.tensor([1.0, 3.0, 3.0, 1.0]) / 4.0
    k2d = torch.outer(k1d, k1d)
    weight = torch.zeros(channels, channels, 4, 4)
    for c in range(channels):
        weight[c, c] = k2d
    with torch.no_grad():
        up.weight.copy_(weight)
    up.weight.requires_grad_(False)
    return up


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
    after it is substituted (fixed bilinear kernel, see module docstring
    point 2). reduce/up/expand/residual_add/out_act are the real per-site
    ops (same site names as the real block)."""

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
        # conv, and FINN's Conv->MVAU fusion pass requires a properly-quantized
        # (integer-datatype) INPUT to fuse, else it leaves a dangling non-HW
        # Im2Col+MatMul+MultiThreshold triple that breaks partitioning. Add an
        # explicit signed passthrough quantizer (FINN-export-only, no HAWQ site)
        # to bridge this gap, bit-width matched to residual_add -- the one real
        # "post" consumer this branch's output feeds into after main_up.
        self.main_act = qnn.QuantIdentity(bit_width=act_bits["residual_add"], act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.main_up = _bilinear_kernel_conv_transpose(out_channels)

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
    """FINN-compatible mirror of LayerQuantENet for
    nnUNetTrainerENet_12_dense_relu_warmstart150ep (per-layer HAWQ
    alpha=0.25). regular1/regular4/regular5/stage2/stage3 are the REAL,
    unmodified LayerQuantRegularBottleneck; initial/down1/down2/up4/up5 are
    the FINN-safe (but numerically exact except for `initial`'s new
    branch_quant, see module docstring) substitutes above. `final`'s bias
    (bias=True, bias_quant=Int32Bias) quantizes the bias into the MatMul
    accumulator's own scale (input_scale * weight_scale), which makes it an
    exact-integer constant -- letting InferChannelwiseLinearLayer lower the
    resulting Add straight to a real ChannelwiseOp HW node instead of a
    CPU-side post-processing step (confirmed against the real S12 preamble
    build's step_enet_convert_to_hw.onnx: plain float biases fail
    InferChannelwiseLinearLayer's `ll_cinit.astype(np.int32) == ll_cinit`
    check)."""

    def __init__(
        self, layer_weight_bits: dict[str, int], layer_act_bits: dict[str, int], *,
        in_channels: int = 1, out_channels: int = 5,
        channels: tuple[int, ...] = CHANNELS, bottlenecks_per_stage: tuple[int, ...] = BOTTLENECKS_PER_STAGE,
        context_pattern: str = CONTEXT_PATTERN, final_bias: bool = True,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits-file", default=str(DEFAULT_BITS_FILE))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--input-hw", type=int, nargs=2, default=(64, 64), metavar=("H", "W"))
    args = parser.parse_args()

    h, w = args.input_hw
    if h % 8 != 0 or w % 8 != 0:
        parser.error(f"--input-hw {h}x{w}: both dims must be divisible by 8.")

    shape_kwargs = dict(
        out_channels=args.out_channels, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(Path(args.bits_file), weight_names, act_names)

    print(f"\n=== Building fresh-weight, per-layer-bit-width FINN-safe 12_dense_relu_warmstart150ep "
          f"(alpha=0.25) -- {len(weight_names)} weight sites, {len(act_names)} act sites ===")
    torch.manual_seed(0)
    model = LayerQuantEnetFINN(
        layer_weight_bits, layer_act_bits, in_channels=args.in_channels, out_channels=args.out_channels,
    ).eval()

    print("\n=== Forward-pass sanity check + QONNX export ===")
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1
    with torch.no_grad():
        out = model(dummy)
    assert out.shape[2:] == (h, w), f"output HxW {tuple(out.shape[2:])} != input ({h},{w})"
    assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
    print(f"  forward OK: output shape {tuple(out.shape)}")

    name = "quantEnet_12_dense_relu_warmstart150ep_alpha025_dummy_int8"
    export_model(model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()

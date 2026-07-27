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
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Uint8ActPerTensorFloat

from nnunetv2.nets.ENet import CONTEXT_STAGE_PATTERN

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


class QuantInitialBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bit_width: int):
        super().__init__()
        if out_channels <= in_channels:
            raise ValueError("QuantInitialBlock out_channels must exceed in_channels.")
        self.input_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.conv = _quant_conv2d(in_channels, out_channels - in_channels, bit_width, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = _quant_act(bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        return self.act(self.bn(torch.cat([self.conv(x), self.pool(x)], dim=1)))


class QuantRegularBottleneck(nn.Module):
    def __init__(
        self, channels: int, bit_width: int, internal_ratio: int = 4,
        kernel_size: int = 3, padding: int = 1, dilation: int = 1,
        asymmetric: bool = False, dropout_p: float = 0.1, use_dsc: bool = False,
    ):
        super().__init__()
        internal_channels = max(1, channels // internal_ratio)

        self.reduce = nn.Sequential(
            _quant_conv2d(channels, internal_channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(internal_channels),
            _quant_act(bit_width),
        )
        if asymmetric:
            if use_dsc:
                raise ValueError("use_dsc is not defined for asymmetric bottlenecks -- see ENet.py's RegularBottleneck.")
            self.conv = nn.Sequential(
                _quant_conv2d(internal_channels, internal_channels, bit_width, kernel_size=(kernel_size, 1), padding=(padding, 0)),
                nn.BatchNorm2d(internal_channels),
                _quant_act(bit_width),
                _quant_conv2d(internal_channels, internal_channels, bit_width, kernel_size=(1, kernel_size), padding=(0, padding)),
            )
        elif use_dsc:
            self.conv = nn.Sequential(
                _quant_conv2d(internal_channels, internal_channels, bit_width, kernel_size=kernel_size,
                               padding=padding, dilation=dilation, groups=internal_channels),
                _quant_conv2d(internal_channels, internal_channels, bit_width, kernel_size=1),
            )
        else:
            self.conv = _quant_conv2d(internal_channels, internal_channels, bit_width, kernel_size=kernel_size, padding=padding, dilation=dilation)

        self.conv_bn_act = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(internal_channels),
            _quant_act(bit_width),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_channels, channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.residual_add = qnn.QuantEltwiseAdd(bit_width=bit_width, input_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.out_act = _quant_act(bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reduce(x)
        out = self.conv_bn_act(out)
        out = self.dropout(self.expand(out))
        return self.out_act(self.residual_add(x, out))


class QuantDownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bit_width: int, internal_ratio: int = 4, dropout_p: float = 0.01, use_strided: bool = True):
        super().__init__()
        internal_channels = max(1, out_channels // internal_ratio)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        if use_strided:
            self.reduce = nn.Sequential(
                _quant_conv2d(in_channels, internal_channels, bit_width, kernel_size=2, stride=2),
                nn.BatchNorm2d(internal_channels),
                _quant_act(bit_width),
            )
        else:
            self.reduce = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                _quant_conv2d(in_channels, internal_channels, bit_width, kernel_size=1),
                nn.BatchNorm2d(internal_channels),
                _quant_act(bit_width),
            )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_channels, internal_channels, bit_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(internal_channels),
            _quant_act(bit_width),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_channels, out_channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.residual_add = qnn.QuantEltwiseAdd(bit_width=bit_width, input_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.out_act = _quant_act(bit_width)
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
    ):
        super().__init__()
        if len(channels) != 5 or len(bottlenecks_per_stage) != 5:
            raise ValueError("channels and bottlenecks_per_stage must each have 5 values (see ENet.py).")
        initial_channels, stage1_channels, stage23_channels, stage4_channels, stage5_channels = channels
        if decoder_type not in ("max_unpool", "upsample_conv"):
            raise ValueError(f"decoder_type must be 'max_unpool' or 'upsample_conv', got {decoder_type!r}.")

        self.decoder_type: DecoderType = decoder_type
        bw = weight_bit_width  # conv weight bit-width; act_bit_width used for _quant_act/input/residual quantizers below
        n_stage1, n_stage2, n_stage3, n_regular4, n_regular5 = bottlenecks_per_stage

        self.initial = QuantInitialBlock(in_channels, initial_channels, act_bit_width)
        self.down1 = QuantDownsamplingBottleneck(initial_channels, stage1_channels, bw, dropout_p=0.01, use_strided=use_strided)
        self.regular1 = nn.Sequential(*[QuantRegularBottleneck(stage1_channels, bw, dropout_p=0.01, use_dsc=use_dsc) for _ in range(n_stage1)])

        self.down2 = QuantDownsamplingBottleneck(stage1_channels, stage23_channels, bw, dropout_p=0.1, use_strided=use_strided)
        self.stage2 = self._make_context_stage(stage23_channels, n_stage2, bw, use_dilated, use_asymmetric, use_dsc)
        self.stage3 = self._make_context_stage(stage23_channels, n_stage3, bw, use_dilated, use_asymmetric, use_dsc)

        self.up4 = QuantUpsamplingBottleneck(stage23_channels, stage4_channels, bw)
        self.regular4 = nn.Sequential(*[QuantRegularBottleneck(stage4_channels, bw, dropout_p=0.1, use_dsc=use_dsc) for _ in range(n_regular4)])
        self.up5 = QuantUpsamplingBottleneck(stage4_channels, stage5_channels, bw)
        self.regular5 = nn.Sequential(*[QuantRegularBottleneck(stage5_channels, bw, dropout_p=0.1, use_dsc=use_dsc) for _ in range(n_regular5)])
        self.final = qnn.QuantConvTranspose2d(stage5_channels, out_channels, kernel_size=2, stride=2, bias=True,
                                               weight_bit_width=bw, weight_quant=Int8WeightPerTensorFloat)

    @staticmethod
    def _make_context_stage(channels: int, n_ops: int, bit_width: int, use_dilated: bool, use_asymmetric: bool, use_dsc: bool = False) -> nn.Sequential:
        ops = []
        for i in range(n_ops):
            kwargs = dict(CONTEXT_STAGE_PATTERN[i % len(CONTEXT_STAGE_PATTERN)])
            if kwargs.get("dilation", 1) != 1 and not use_dilated:
                kwargs = {}
            if kwargs.get("asymmetric", False) and not use_asymmetric:
                kwargs = {}
            ops.append(QuantRegularBottleneck(channels, bit_width, dropout_p=0.1, use_dsc=use_dsc, **kwargs))
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

"""Export a MINIMAL FINN-safe model containing exactly ONE S19-style
regular bottleneck block: 1x1 reduce -> separable-dilated 3x3 (factored as
(3,1)+(1,3), dilation=2) -> 1x1 expand, int8 throughout. Plain residual add
+ ReLU (FINN-safe convention, matches every other FINN*Bottleneck in
hardware/finn_export_s13_leaky_frozen.py).

This is a standalone resource-probe artifact -- NOT a slice of the real S19
network graph (no real trained weights, fresh torch.manual_seed(0) init,
same established convention as every other FINN*-prefixed export in this
repo: FINN's resource/timing estimate depends only on architecture +
bit-width + which nodes exist, not on weight values).

Reuses FINNRegularBottleneckSepDilated (the exact class S19's own stage2/
stage3 context blocks use, see finn_export_s19_double_mid.py) and the
proven export_model()/_fast_cleanup() pipeline from
finn_export_s13_leaky_frozen.py, rather than redefining either.

Usage (run inside a container with torch+brevitas+qonnx, e.g. `lightm_pytorch`):
    docker exec lightm_pytorch python /workspace/hardware/finn_export_s19_single_block.py

Output: hardware/outputs/finn_exports/quantEnet_s19_single_block_int8.onnx
Then, inside the FINN container (`brave_lewin`):
    docker cp hardware/outputs/finn_exports/quantEnet_s19_single_block_int8.onnx \\
        brave_lewin:/home/thelegendiv/finn/notebooks/enet/
    docker exec -e HOME=/tmp/home_dir brave_lewin python3 \\
        /home/thelegendiv/finn/notebooks/enet/finn_enet_build_decomposed_prelu.py \\
        quantEnet_s19_single_block_int8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finn_export_s13_leaky_frozen import (  # noqa: E402
    FINNRegularBottleneckSepDilated, _plain_relu_factory, export_model,
)

DEFAULT_CHANNELS = 32   # S19's own stage2/stage3 channel count (c23), see DEFAULT_CHANNELS in finn_export_s19_double_mid.py
DEFAULT_DILATION = 2
BIT_WIDTH = 8


class FINNS19SingleBlock(nn.Module):
    """input_quant (int8) -> ONE FINNRegularBottleneckSepDilated -> output.
    No initial/downsampling/upsampling stages -- just the bottleneck itself,
    at a fixed spatial resolution."""

    def __init__(self, channels: int = DEFAULT_CHANNELS, bit_width: int = BIT_WIDTH,
                 dilation: int = DEFAULT_DILATION, residual: bool = True):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.block = FINNRegularBottleneckSepDilated(
            channels, bit_width, _plain_relu_factory, dilation=dilation,
            dropout_p=0.0, residual=residual,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(self.input_quant(x))
        return out.value if hasattr(out, "value") else out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--dilation", type=int, default=DEFAULT_DILATION)
    parser.add_argument("--bit-width", type=int, default=BIT_WIDTH)
    parser.add_argument("--input-hw", type=int, nargs=2, default=(32, 32), metavar=("H", "W"))
    parser.add_argument("--no-residual", action="store_true")
    args = parser.parse_args()

    h, w = args.input_hw
    torch.manual_seed(0)
    dummy = torch.rand(1, args.channels, h, w) * 2 - 1

    print("\nFINNS19SingleBlock export")
    print(f"  channels={args.channels}  dilation={args.dilation}  bit_width={args.bit_width}")
    print(f"  input=({args.channels},{h},{w})  residual={not args.no_residual}")

    model = FINNS19SingleBlock(
        channels=args.channels, bit_width=args.bit_width,
        dilation=args.dilation, residual=not args.no_residual,
    ).eval()

    with torch.no_grad():
        out = model(dummy)
    assert out.shape == dummy.shape, f"output shape {tuple(out.shape)} != input shape {tuple(dummy.shape)}"
    print(f"  forward OK: output shape {tuple(out.shape)}")

    name = f"quantEnet_s19_single_block_int{args.bit_width}"
    export_model(model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx brave_lewin:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()

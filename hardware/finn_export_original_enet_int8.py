"""Export the 'original' (full-scale, unpruned) ENet architecture -- 8-bit
quantized, no PReLU (QuantReLU throughout), no MaxUnpool (ConvTranspose2d-based
upsampling, no F.interpolate) -- to QONNX, for a fully-unfolded FINN resource
estimate (side job, run 2026-08-17; independent of the running S19 partitioned
IP-build job).

Reuses FINNQuantENet from finn_enet_prod_export.py verbatim (that class is
already exactly this: default channels=(20,72,144,72,20) [[the ENet-paper-
scale "original" widths]], bottlenecks_per_stage=(4,8,8,2,1), QuantReLU-only
activations, ConvTranspose2d-based upsampling bottlenecks -- see that file's
module docstring for the 6 FINN-compatibility fixes vs QuantENet.py).

Usage (inside the FINN container):
    python3 finn_export_original_enet_int8.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_deps"))
from finn_enet_prod_export import FINNQuantENet, export_model  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# "Original" ENet scale (ENet-paper channel widths / bottleneck counts -- the
# repo's own DEFAULT_CHANNELS/DEFAULT_BNECKS, i.e. NOT one of the compressed
# sweep configs like S8/S13/S19).
CHANNELS = (20, 72, 144, 72, 20)
BNECKS = (4, 8, 8, 2, 1)
BIT_WIDTH = 8
IN_CHANNELS = 1
OUT_CHANNELS = 2
INPUT_HW = (512, 512)  # matches compression/utils.py's count_flops/count_buffer_elements convention


def main() -> None:
    torch.manual_seed(0)
    dummy = torch.rand(1, IN_CHANNELS, *INPUT_HW) * 2 - 1

    print("FINNQuantENet (original scale) export")
    print(f"  channels={CHANNELS}, bnecks={BNECKS}, bit_width={BIT_WIDTH}")
    print(f"  input=({IN_CHANNELS},{INPUT_HW[0]},{INPUT_HW[1]})")

    model = FINNQuantENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,
        channels=CHANNELS, bottlenecks_per_stage=BNECKS,
        use_dilated=True, bit_width=BIT_WIDTH, residual=True,
    ).eval()

    with torch.no_grad():
        out = model(dummy)
    out_t = out.value if hasattr(out, "value") else out
    assert out_t.shape[2:] == INPUT_HW, f"output HxW {tuple(out_t.shape[2:])} != input {INPUT_HW}"
    print(f"  forward OK: output shape {tuple(out_t.shape)}")

    global OUT_DIR
    import finn_enet_prod_export as prod_export
    prod_export.OUT_DIR = OUT_DIR
    path = export_model(model, "quantEnet_original_int8", dummy)
    print(f"\nSaved QONNX model to: {path}")


if __name__ == "__main__":
    main()

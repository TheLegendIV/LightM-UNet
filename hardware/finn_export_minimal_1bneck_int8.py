"""Export a MINIMAL derisking variant of FINNQuantENet: same topology/channel
widths as the real ENet baseline (initial block, down1, down2, up4, up5,
final ConvTranspose all present and unchanged), but only 1 total extra
"regular/context" bottleneck (bottlenecks_per_stage=(0,1,0,0,0) -- a single,
non-dilated RegularBottleneck placed in stage2) instead of the full (4,8,8,2,1).
Dummy (randomly self-initialized, untrained) weights, INT8 throughout, plain
QuantReLU activations (FINNQuantENet's default -- no PReLU/LeakyReLU decomposition).

Goal: the smallest possible model that still exercises the SAME full
encoder/decoder data path (residuals, both downsampling and both upsampling
bottlenecks, final ConvTranspose) as production ENet, to derisk the full
Vivado ZynqBuild (single partition, real bitstream + DMA + PYNQ driver)
end-to-end before spending time on the much larger real models.

Small 64x64 input (vs production's 512x512) to keep HLS/Vivado synthesis time
low for this derisking build -- divisible by 8, well within both downsampling
bottlenecks' stride-2 halving.

Usage (inside the FINN container):
    python3 finn_export_minimal_1bneck_int8.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_deps"))
from finn_enet_prod_export import FINNQuantENet, export_model  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# Same baseline widths as finn_export_original_enet_int8.py's true ENet-paper
# CHANNELS -- "everything else the same as ENet".
CHANNELS = (16, 64, 128, 64, 16)
# Only 1 extra bottleneck total (stage2 slot 0 -> CONTEXT_DILATIONS[0]==1,
# i.e. a plain non-dilated RegularBottleneck). down1/down2/up4/up5 are
# separate, always-present structural bottlenecks, unaffected by this tuple.
BNECKS = (0, 1, 0, 0, 0)
BIT_WIDTH = 8
IN_CHANNELS = 1
OUT_CHANNELS = 2
INPUT_HW = (64, 64)
MODEL_NAME = "quantEnet_minimal_1bneck_int8"


def main() -> None:
    torch.manual_seed(0)
    dummy = torch.rand(1, IN_CHANNELS, *INPUT_HW) * 2 - 1

    print("FINNQuantENet (minimal 1-bottleneck derisking build) export")
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
    assert torch.isfinite(out_t).all(), "output contains NaN/Inf -- weights not properly initialized"
    assert out_t.abs().sum() > 0, "output is all-zero -- weights not properly initialized"
    print(f"  forward OK: output shape {tuple(out_t.shape)}, finite, non-zero (dummy weights confirmed live)")

    global OUT_DIR
    import finn_enet_prod_export as prod_export
    prod_export.OUT_DIR = OUT_DIR
    path = export_model(model, MODEL_NAME, dummy)
    print(f"\nSaved QONNX model to: {path}")


if __name__ == "__main__":
    main()

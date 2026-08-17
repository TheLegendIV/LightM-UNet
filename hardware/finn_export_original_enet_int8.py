"""Export the 'original' (full-scale, unpruned) ENet architecture -- 8-bit
quantized, no PReLU (QuantReLU throughout), no MaxUnpool (ConvTranspose2d-based
upsampling, no F.interpolate) -- to QONNX, for a fully-unfolded FINN resource
estimate (side job, run 2026-08-17; independent of the running S19 partitioned
IP-build job).

Reuses FINNQuantENet from finn_enet_prod_export.py (that class defaults to
the E1 sweep-grid widths, NOT the true baseline -- see CHANNELS override
below), bottlenecks_per_stage=(4,8,8,2,1), QuantReLU-only activations,
ConvTranspose2d-based upsampling bottlenecks -- see that file's module
docstring for the 6 FINN-compatibility fixes vs QuantENet.py).

CORRECTED 2026-08-17: this script previously used CHANNELS=(20,72,144,72,20)
(finn_enet_prod_export.py's DEFAULT_CHANNELS), which is actually the `E1`
sweep-grid cell from agent_instructions_1.yaml, mislabeled here as the
"original" ENet-paper widths. The real baseline -- confirmed against
compression/results.csv's f_i/f1/f2/f3/f4/f5 columns for
nnUNetTrainerENet_1_naive_baseline_Baseline / _3_transfer_original /
_25_s19_baseline_width (16,64,128,128,64,16) -- is CHANNELS=(16,64,128,64,16)
below (f2==f3==128 collapses to one shared c23 value in this 5-tuple
convention). Every downstream artifact exported/estimated with the old
tuple (quantEnet_original_int8.onnx and the
quantEnet_original_int8_unfolded_report/ outputs) is stale and needs
regenerating -- see regenerate_original_enet_int8.py.

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

# True ENet-paper baseline widths (16,64,128,128,64,16 -- f2==f3==128 shares
# one c23 value here), NOT the E1 sweep-grid cell. See CORRECTED note above.
CHANNELS = (16, 64, 128, 64, 16)
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
    # nn.Module/Brevitas layers self-init (reset_parameters()) on construction,
    # so weights are already non-empty random dummy values -- this just proves
    # a real forward pass actually ran through them (not NaN/all-zero, which
    # would indicate a broken/uninitialized layer rather than legitimate init).
    assert torch.isfinite(out_t).all(), "output contains NaN/Inf -- weights not properly initialized"
    assert out_t.abs().sum() > 0, "output is all-zero -- weights not properly initialized"
    print(f"  forward OK: output shape {tuple(out_t.shape)}, finite, non-zero (dummy weights confirmed live)")

    global OUT_DIR
    import finn_enet_prod_export as prod_export
    prod_export.OUT_DIR = OUT_DIR
    path = export_model(model, "quantEnet_original_int8", dummy)
    print(f"\nSaved QONNX model to: {path}")


if __name__ == "__main__":
    main()

"""One-off export sweep: O4_native, O2_native, ENet_native (published widths) x INT2/4/8/16.
Run inside the pytorch container (brevitas/qonnx/onnx only, no FINN needed).
"""
import sys
sys.path.insert(0, '/workspace/LightM-UNet/enet')
sys.path.insert(0, '/workspace/LightM-UNet/hardware')

import torch
from pathlib import Path
import finn_enet_prod_export as mod
from finn_enet_prod_export import FINNQuantENet, export_model

mod.OUT_DIR = Path('/tmp/enet_export_out')

CONFIGS = {
    # name_prefix: (channels 5-tuple, bnecks 5-tuple)
    "O2_native":  ((16, 32, 64, 32, 8),   (4, 8, 8, 2, 1)),
    "O4_native":  ((16, 16, 32, 16, 4),   (4, 8, 8, 2, 1)),
    "ENet_native": ((16, 64, 128, 64, 16), (4, 8, 8, 2, 1)),  # published ENet paper widths
}
BIT_WIDTHS = [2, 4, 8, 16]

H, W = 64, 64

for cfg_name, (channels, bnecks) in CONFIGS.items():
    for bw in BIT_WIDTHS:
        name = f"quantEnet_{cfg_name}_int{bw}"
        torch.manual_seed(0)
        dummy = torch.rand(1, 1, H, W) * 2 - 1
        print(f"\n=== {name} ===")
        print(f"channels={channels} bnecks={bnecks} bit_width={bw}")
        model = FINNQuantENet(
            in_channels=1, out_channels=2,
            channels=channels, bottlenecks_per_stage=bnecks,
            use_dilated=True, bit_width=bw, residual=True,
        ).eval()
        with torch.no_grad():
            out = model(dummy)
        out_t = out.value if hasattr(out, 'value') else out
        assert out_t.shape == (1, 2, H, W), f"shape mismatch: {tuple(out_t.shape)}"
        print(f"Forward OK: {tuple(out_t.shape)}")
        export_model(model, name, dummy)

print("\nAll exports done.")

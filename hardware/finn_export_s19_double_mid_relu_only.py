"""ReLU-only variant of finn_export_s19_double_mid.py's export: the REAL
QuantENet (source of transferred conv/BN weights) is built with the REAL
leaky_slope_map (required to match the trained checkpoint's state_dict
keys/shapes), but the FINN-safe mirror (finn_model, the one actually
exported to ONNX) is built with an EMPTY slope map, so every activation
site falls back to `_make_act_factory(None)` -> `_plain_relu_factory`
(the epsilon-slope ReLU-equivalent DecomposedLeakyAct(negative_slope=1e-6)
already used unconditionally for decoder blocks in the original script) --
i.e. a plain-ReLU topology with the same conv/BN weights, used purely to
measure the hardware-resource (BRAM/LUT/DSP) impact of PReLU/LeakyReLU vs.
plain ReLU. Not an accuracy-preserving export.

Usage (run inside the pytorch training container):
    python hardware/finn_export_s19_double_mid_relu_only.py
Output: hardware/outputs/finn_exports/quantEnet_s19_double_mid_relu_int8.onnx
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnunetv2.nets.QuantENet import QuantENet  # noqa: E402
from finn_export_s13_leaky_frozen import export_model  # noqa: E402
from finn_export_s19_double_mid import (  # noqa: E402
    FINNQuantENetS19DoubleMid, transfer_weights,
    DEFAULT_CHANNELS, DEFAULT_BNECKS, BIT_WIDTH,
    DEFAULT_SLOPE_MAP_FILE, DEFAULT_CHECKPOINT,
)

IN_CHANNELS = 1
OUT_CHANNELS = 5
H = W = 64

print("\n=== 1. Building + loading the REAL trained QuantENet (S19 config, REAL slope map) ===")
with open(DEFAULT_SLOPE_MAP_FILE) as f:
    real_slope_map = json.load(f)

real_model = QuantENet(
    in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,
    channels=DEFAULT_CHANNELS, bottlenecks_per_stage=DEFAULT_BNECKS,
    decoder_type="upsample_conv", use_dilated=True, use_asymmetric=False,
    use_strided=True, use_dsc=False,
    weight_bit_width=BIT_WIDTH, act_bit_width=BIT_WIDTH,
    context_pattern="dense_dilation_reg_interleaved_double_mid",
    separable_dilated=True, leaky_slope_map=real_slope_map,
)
checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location="cpu")
state_dict = checkpoint["network_weights"]
new_state_dict = {
    (key[7:] if key.startswith("module.") and key not in real_model.state_dict() else key): value
    for key, value in state_dict.items()
}
missing, unexpected = real_model.load_state_dict(new_state_dict, strict=True)
real_model.eval()
print(f"  Loaded {DEFAULT_CHECKPOINT} (epoch {checkpoint.get('current_epoch')}) -- missing={missing} unexpected={unexpected}")

print("\n=== 2. Building the FINN-safe mirror WITH EMPTY slope map (plain ReLU everywhere) + transferring real weights ===")
finn_model = FINNQuantENetS19DoubleMid(
    in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,
    channels=DEFAULT_CHANNELS, bottlenecks_per_stage=DEFAULT_BNECKS,
    bit_width=BIT_WIDTH, residual=True, leaky_slope_map={},
).eval()

report = transfer_weights(finn_model, real_model)
n_ok = sum(1 for line in report if "[OK]" in line)
n_fresh = sum(1 for line in report if "[FRESH]" in line)
print(f"  {n_ok} components transferred from the real checkpoint, {n_fresh} components fresh-initialized.")

print("\n=== 3. Forward-pass sanity check + QONNX export ===")
dummy = torch.rand(1, IN_CHANNELS, H, W) * 2 - 1
with torch.no_grad():
    out = finn_model(dummy)
assert out.shape[2:] == (H, W), f"output HxW {tuple(out.shape[2:])} != input ({H},{W})"
assert out.shape[1] == OUT_CHANNELS, f"output channels {out.shape[1]} != {OUT_CHANNELS}"
print(f"  forward OK: output shape {tuple(out.shape)}")

name = f"quantEnet_s19_double_mid_relu_int{BIT_WIDTH}"
export_model(finn_model, name, dummy)
print(f"\nDone. hardware/outputs/finn_exports/{name}.onnx")

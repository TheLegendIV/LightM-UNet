"""Run the FINN-compatible ENet production export inside the container.

Generalized to take channel/bottleneck shapes + an output name on the CLI,
so different sweep-grid architectures (e.g. 2a_O8_native, 2a_O4_native from
compression/results.csv) can be exported without editing this file each time.
Writes directly into the FINN container's notebooks/enet/ folder (no
separate docker cp needed for the .onnx itself).

Usage (run inside the FINN container, after docker cp-ing this file and
finn_enet_prod_export.py to /tmp/):
    python /tmp/run_export_in_container.py \
        --channels 16 8 16 8 4 --bnecks 4 8 8 2 1 --name quantEnet_O8_native

    # defaults to the E1 config used in the first estimate-only build:
    python /tmp/run_export_in_container.py
"""
import argparse
import sys

# FINN + deps path
sys.path.insert(0, '/home/thelegendiv/finn/src')
sys.path.insert(0, '/home/thelegendiv/finn/deps/qonnx/src')
sys.path.insert(0, '/home/thelegendiv/finn/deps/brevitas/src')
sys.path.insert(0, '/home/thelegendiv/finn/deps/pyverilator')
sys.path.insert(0, '/home/thelegendiv/finn/deps/finn-experimental')

# ENet source path
sys.path.insert(0, '/tmp/enet_src')

# Patch OUT_DIR to write to the enet notebooks folder
import finn_enet_prod_export as mod
from pathlib import Path
mod.OUT_DIR = Path('/home/thelegendiv/finn/notebooks/enet')

# Import needed items
from finn_enet_prod_export import FINNQuantENet, export_model
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--channels", type=int, nargs=5, default=[20, 72, 144, 72, 20],
                    metavar=("C0", "C1", "C23", "C4", "C5"))
parser.add_argument("--bnecks", type=int, nargs=5, default=[4, 8, 8, 2, 1],
                    metavar=("N1", "N2", "N3", "N4", "N5"))
parser.add_argument("--name", default="quantEnet_finn_v1")
parser.add_argument("--bit-width", type=int, default=8)
parser.add_argument("--no-dilated", action="store_true")
parser.add_argument("--no-residuals", action="store_true")
parser.add_argument("--input-hw", type=int, nargs=2, default=(64, 64), metavar=("H", "W"))
args = parser.parse_args()

torch.manual_seed(0)
channels = tuple(args.channels)
bnecks   = tuple(args.bnecks)
H, W     = args.input_hw
residual = not args.no_residuals
use_dilated = not args.no_dilated

print(f"\n=== Export FINNQuantENet ({args.name}) ===")
print(f"channels={channels}, bnecks={bnecks}, bit_width={args.bit_width}, "
      f"residual={residual}, use_dilated={use_dilated}, input=(1,1,{H},{W})")

dummy = torch.rand(1, 1, H, W) * 2 - 1
model = FINNQuantENet(
    in_channels=1, out_channels=2,
    channels=channels, bottlenecks_per_stage=bnecks,
    use_dilated=use_dilated, bit_width=args.bit_width, residual=residual,
).eval()

with torch.no_grad():
    out = model(dummy)
out_t = out.value if hasattr(out, 'value') else out
assert out_t.shape == (1, 2, H, W), f"shape mismatch: {tuple(out_t.shape)}"
print(f"Forward OK: {tuple(out_t.shape)}")

export_model(model, args.name, dummy)
print("\nExport done.")


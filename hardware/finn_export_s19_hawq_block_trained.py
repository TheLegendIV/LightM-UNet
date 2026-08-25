"""Export the REAL TRAINED per-block HAWQ S19 checkpoint
(nnUNetTrainerENetQuantS19Block_s19_qat_block_bits, checkpoint_best.pth,
epoch 147 -- see compression/configs/s19_per_block_qat_results.md) through
the FINN-safe topology, WITH REAL TRAINED WEIGHTS transferred wherever the
FINN-safe topology is structurally identical to QuantENetS19Block's real
architecture. Same convention/complication as finn_export_s19_double_mid.py
(see that file's own docstring for the 4 FINN-topology deviations that stay
fresh-initialized: initial block conv, downsampling shortcut_proj, upsampling
main_up/main_bn, final bias) -- and the SAME per-block bit-widths
(compression/hawq/block_bits_s19.json) as finn_export_s19_hawq_block.py, but
here every conv/BN pair that IS structurally transferable carries the real
trained value instead of a fresh torch.manual_seed(0) init, and the model
keeps the REAL trained per-block nonneg_block leaky slopes (QuantDecomposedLeakyAct)
-- NOT plain ReLU -- since the config's whole point (this is the real
deployable per-block QAT result, dice=0.7458) depends on that activation.

Reuses FINNQuantENetS19BlockHAWQ (finn_export_s19_hawq_block.py) as the FINN-
safe destination topology, and transfer_weights() + its private per-block
helpers (finn_export_s19_double_mid.py) as the transfer logic -- both are
already fully generic over "any dst/src pair with QuantENet.py's block-level
Sequential structure", which QuantENetS19Block also uses directly (imports
QuantDownsamplingBottleneck/QuantRegularBottleneck/QuantUpsamplingBottleneck
from QuantENet.py rather than redefining them), so no new transfer code is
needed here -- only the src model class + checkpoint + per-block bit-width
dicts differ from finn_export_s19_double_mid.py's own uniform-bit-width case.

Usage (run inside the pytorch training container):
    python hardware/finn_export_s19_hawq_block_trained.py

Output: hardware/outputs/finn_exports/quantEnet_s19_hawq_block_trained.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_s19_hawq_block_trained.onnx \\
        <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnunetv2.nets.QuantENetS19Block import QuantENetS19Block  # noqa: E402
from finn_export_s13_leaky_frozen import export_model  # noqa: E402
from finn_export_s19_double_mid import transfer_weights  # noqa: E402
from finn_export_s19_hawq_block import FINNQuantENetS19BlockHAWQ, DEFAULT_SLOPE_MAP_FILE, DEFAULT_BLOCK_BITS_FILE  # noqa: E402

DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data" / "nnUNet_results" / "Dataset509_ARCADE_1x1_4c"
    / "nnUNetTrainerENetQuantS19Block_s19_qat_block_bits__nnUNetPlans__2d"
    / "fold_0" / "checkpoint_best.pth"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--block-bits-file", default=str(DEFAULT_BLOCK_BITS_FILE))
    parser.add_argument("--slope-map-file", default=str(DEFAULT_SLOPE_MAP_FILE))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--input-hw", type=int, nargs=2, default=(64, 64), metavar=("H", "W"))
    parser.add_argument("--no-residuals", action="store_true")
    args = parser.parse_args()

    h, w = args.input_hw
    if h % 8 != 0 or w % 8 != 0:
        parser.error(f"--input-hw {h}x{w}: both dims must be divisible by 8.")

    with open(args.block_bits_file) as f:
        block_bits = json.load(f)
    block_weight_bits = block_bits["stage_weight_bits"]
    block_act_bits = block_bits["stage_act_bits"]

    with open(args.slope_map_file) as f:
        leaky_slope_map = json.load(f)

    print("\n=== 1. Building + loading the REAL trained QuantENetS19Block (per-block HAWQ bit-width, REAL slope map) ===")
    real_model = QuantENetS19Block(block_weight_bits, block_act_bits, leaky_slope_map=leaky_slope_map)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["network_weights"]
    new_state_dict = {
        (key[7:] if key.startswith("module.") and key not in real_model.state_dict() else key): value
        for key, value in state_dict.items()
    }
    missing, unexpected = real_model.load_state_dict(new_state_dict, strict=True)
    real_model.eval()
    print(f"  Loaded {args.checkpoint} (epoch {checkpoint.get('current_epoch')}) -- missing={missing} unexpected={unexpected}")

    print("\n=== 2. Building the FINN-safe HAWQ mirror + transferring real weights ===")
    residual = not args.no_residuals
    finn_model = FINNQuantENetS19BlockHAWQ(
        block_weight_bits, block_act_bits,
        in_channels=args.in_channels, out_channels=args.out_channels,
        residual=residual, leaky_slope_map=leaky_slope_map,
    ).eval()

    report = transfer_weights(finn_model, real_model)
    print("\n".join(report))
    n_ok = sum(1 for line in report if "[OK]" in line)
    n_fresh = sum(1 for line in report if "[FRESH]" in line)
    print(f"\n  {n_ok} components transferred from the real checkpoint, {n_fresh} components fresh-initialized.")
    print("  (Real accuracy will NOT exactly match the checkpoint's own dice=0.7458 until/unless "
          "fine-tuned with this exact FINN-safe topology -- see finn_export_s19_double_mid.py's docstring.)")

    print("\n=== 3. Forward-pass sanity check + QONNX export ===")
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1
    with torch.no_grad():
        out = finn_model(dummy)
    assert out.shape[2:] == (h, w), f"output HxW {tuple(out.shape[2:])} != input ({h},{w})"
    assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
    print(f"  forward OK: output shape {tuple(out.shape)}")

    suffix = "_no_res" if not residual else ""
    name = f"quantEnet_s19_hawq_block_trained{suffix}"
    export_model(finn_model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()

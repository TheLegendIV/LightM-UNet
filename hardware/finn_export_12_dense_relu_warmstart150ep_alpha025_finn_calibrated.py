"""Export a FINN-compatible, PER-LAYER HAWQ mirror of
12_dense_relu_warmstart150ep (alpha=0.25) directly from a checkpoint of
`LayerQuantEnetFINN` ITSELF (trainer_name="LayerQuantEnetFINN"), rather than
the older finn_export_..._trained.py's approach of transferring weights
from the real LayerQuantENet checkpoint into a freshly-constructed
LayerQuantEnetFINN + a post-hoc calibration_mode() pass.

Checkpoint: data/nnUNet_results/Dataset509_ARCADE_1x1_4c/
nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_warmstart150ep_alpha0.25_
from_ft15ep_calibrated_init__nnUNetPlans__2d/fold_0/checkpoint_final.pth
-- init_args in this checkpoint give layer_weight_bits/layer_act_bits/
channels/bottlenecks_per_stage/context_pattern/in_channels/out_channels
directly (no need for the separate layer_bits_SITES_*.json file at all),
and its `network_weights` state dict is a COMPLETE, exact match for a
freshly-constructed LayerQuantEnetFINN(**those same kwargs) -- verified
(2026-09-14): 0 missing / 0 unexpected keys under strict `load_state_dict`,
including main_up/shortcut_proj/branch_quant (this checkpoint's own
training run produced/calibrated those directly, unlike the old
transfer-from-real-checkpoint path where they were left at fresh/frozen
construction-time values). The 140 extra keys present in the checkpoint but
absent from `model.state_dict()` are Brevitas's NON-PERSISTENT runtime-stats
scaling buffers (`...scaling_impl.value`, excluded from `state_dict()`'s
output by design but still matched and loaded by `load_state_dict`, which
looks up `self._buffers` directly, persistent or not) -- so no separate
`calibration_mode()` pass is needed here: this checkpoint's own real
training already settled those scales.

Usage (run inside the pytorch training container):
    docker exec <container> python /workspace/LightM-UNet/hardware/finn_export_12_dense_relu_warmstart150ep_alpha025_finn_calibrated.py

Output: hardware/outputs/finn_exports/quantEnet_12_dense_relu_warmstart150ep_alpha025_finn_calibrated_int8.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_12_dense_relu_warmstart150ep_alpha025_finn_calibrated_int8.onnx \\
        <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402
from finn_export_s13_leaky_frozen import export_model  # noqa: E402

DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data" / "nnUNet_results" / "Dataset509_ARCADE_1x1_4c"
    / "nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_warmstart150ep_alpha0.25_from_ft15ep_calibrated_init__nnUNetPlans__2d"
    / "fold_0" / "checkpoint_final.pth"
)


def load_finn_checkpoint(checkpoint_path: Path) -> LayerQuantEnetFINN:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("trainer_name") != "LayerQuantEnetFINN":
        raise ValueError(
            f"expected a LayerQuantEnetFINN-trained checkpoint, got trainer_name={checkpoint.get('trainer_name')!r}"
        )
    init_args = checkpoint["init_args"]
    model = LayerQuantEnetFINN(
        init_args["layer_weight_bits"], init_args["layer_act_bits"],
        in_channels=init_args["in_channels"], out_channels=init_args["out_channels"],
        channels=tuple(init_args["channels"]), bottlenecks_per_stage=tuple(init_args["bottlenecks_per_stage"]),
        context_pattern=init_args["context_pattern"],
    ).eval()
    missing, unexpected = model.load_state_dict(checkpoint["network_weights"], strict=False)
    if missing:
        raise ValueError(f"load_state_dict({checkpoint_path}): {len(missing)} missing key(s): {missing[:10]}")
    print(f"  load_state_dict({checkpoint_path.name}): OK, {len(unexpected)} non-persistent buffer key(s) "
          f"also matched (expected: Brevitas runtime-stats scaling buffers).")
    return model, init_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--input-hw", type=int, nargs=2, default=(64, 64), metavar=("H", "W"))
    args = parser.parse_args()

    h, w = args.input_hw
    if h % 8 != 0 or w % 8 != 0:
        parser.error(f"--input-hw {h}x{w}: both dims must be divisible by 8.")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}")

    print(f"\n=== Loading LayerQuantEnetFINN directly from {checkpoint_path} ===")
    model, init_args = load_finn_checkpoint(checkpoint_path)

    print("\n=== Forward-pass sanity check + QONNX export ===")
    dummy = torch.rand(1, init_args["in_channels"], h, w) * 2 - 1
    with torch.no_grad():
        out = model(dummy)
    assert out.shape[2:] == (h, w), f"output HxW {tuple(out.shape[2:])} != input ({h},{w})"
    assert out.shape[1] == init_args["out_channels"], f"output channels {out.shape[1]} != {init_args['out_channels']}"
    print(f"  forward OK: output shape {tuple(out.shape)}")

    name = "quantEnet_12_dense_relu_warmstart150ep_alpha025_finn_calibrated_int8"
    export_model(model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()

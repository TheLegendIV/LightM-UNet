"""Conv-order dump for the existing v3 export. Reuses finn_export_..._v3_trained.py's exact
checkpoint/bits/architecture -- does NOT re-run calibration or re-export the onnx (conv_order is
bit-independent, so this is safe to produce standalone against the onnx already on disk).

Usage (pytorch container):
    docker exec <container> python3 hardware/builds/12_dense_relu_nearest_conv_upsample_256_v3/dump_conv_order_v3.py

Output: hardware/builds/12_dense_relu_nearest_conv_upsample_256_v3/outputs/quantEnet_12_dense_relu_nearest_conv_upsample_256_v3_trained_conv_order.json
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "hardware"))
sys.path.insert(0, str(REPO_ROOT / "hardware" / "builds" / "12_dense_relu_nearest_conv_upsample_256"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402
from s12_export import load_layer_bits, dump_conv_order  # noqa: E402
import finn_export_12_dense_relu_nearest_conv_upsample_256_v3_trained as v3  # noqa: E402

OUT_JSON = (Path(__file__).resolve().parent / "outputs"
            / "quantEnet_12_dense_relu_nearest_conv_upsample_256_v3_trained_conv_order.json")


def main() -> None:
    layer_names_kwargs = dict(
        out_channels=5, channels=v3.CHANNELS, bottlenecks_per_stage=v3.BOTTLENECKS_PER_STAGE,
        context_pattern=v3.CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
        decoder_type=v3.DECODER_TYPE,
    )
    weight_names, act_names = layer_names_for(**layer_names_kwargs)
    w_bits, a_bits = load_layer_bits(v3.DEFAULT_BITS_FILE, weight_names, act_names)
    print(f"=== Rebuilding v3 model for conv_order dump: {v3.DEFAULT_CHECKPOINT.name}, "
          f"{len(weight_names)} weight / {len(act_names)} act sites ===")
    model = LayerQuantEnetFINN.from_pretrained(
        v3.DEFAULT_CHECKPOINT, w_bits, a_bits, in_channels=1, out_channels=5,
        channels=v3.CHANNELS, bottlenecks_per_stage=v3.BOTTLENECKS_PER_STAGE,
        context_pattern=v3.CONTEXT_PATTERN, decoder_type=v3.DECODER_TYPE,
    ).eval()
    dump_conv_order(model, in_channels=1, out_path=OUT_JSON)


if __name__ == "__main__":
    main()

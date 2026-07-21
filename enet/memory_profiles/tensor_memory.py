from __future__ import annotations

import argparse
import sys
from pathlib import Path

from memory_utils import (
    ParameterRecord,
    TensorRecord,
    dtype_bytes,
    downsampled_hw,
    encoder_shapes,
    parse_channels,
    print_parameter_summary,
    print_parameter_table,
    print_path_summary,
    print_tensor_table,
)


def build_records(
    batch: int,
    height: int,
    width: int,
    in_channels: int,
    out_channels: int,
    channels: tuple[int, ...],
    edge_channels: int,
) -> list[TensorRecord]:
    enc = encoder_shapes(batch, height, width, channels)
    records: list[TensorRecord] = []

    records.append(TensorRecord("main", "input", (batch, in_channels, height, width)))

    for index, shape in enumerate(enc, start=1):
        records.append(
            TensorRecord(
                "main",
                f"encoder_f{index}",
                shape,
                "encoder stage output",
            )
        )

    decoder_specs = [
        ("decoder_to_f5", channels[4], enc[4][2], enc[4][3]),
        ("decoder_to_f4", channels[3], enc[3][2], enc[3][3]),
        ("decoder_to_f3", channels[2], enc[2][2], enc[2][3]),
        ("decoder_to_f2", channels[1], enc[1][2], enc[1][3]),
        ("decoder_to_f1", channels[0], enc[0][2], enc[0][3]),
    ]
    for name, channel, h, w in decoder_specs:
        records.append(
            TensorRecord("main", name, (batch, channel, h, w), "decoder block output")
        )

    records.append(
        TensorRecord(
            "main",
            "final_logits",
            (batch, out_channels, height, width),
            "segmentation logits before argmax/sigmoid",
        )
    )

    for index, shape in enumerate(enc[:-1], start=1):
        records.append(
            TensorRecord(
                "skip",
                f"raw_skip_f{index}",
                shape,
                "encoder feature retained for decoder",
            )
        )

    for index, shape in enumerate(enc, start=1):
        records.append(
            TensorRecord(
                "skip",
                f"mmsc_refined_f{index}",
                shape,
                "MMSC output used by decoder",
            )
        )

    low_h, low_w = enc[1][2], enc[1][3]
    high_h, high_w = enc[5][2], enc[5][3]

    edge_records = [
        TensorRecord("edge", "efe_low_pv", enc[1], "processed f2 detail feature"),
        TensorRecord("edge", "efe_high_pv", enc[5], "processed f6 semantic feature"),
        TensorRecord(
            "edge",
            "efe_high_projected",
            (batch, channels[1], high_h, high_w),
            "high feature projected to f2 channels",
        ),
        TensorRecord(
            "edge",
            "efe_high_upsampled",
            (batch, channels[1], low_h, low_w),
            "projected high feature resized to f2",
        ),
        TensorRecord(
            "edge",
            "efe_concat_low_high",
            (batch, channels[1] * 2, low_h, low_w),
            "low/high concat before edge fuse",
        ),
        TensorRecord(
            "edge",
            "efe_fuse_hidden",
            (batch, edge_channels, low_h, low_w),
            "edge fuse hidden feature",
        ),
        TensorRecord("edge", "edge_map", (batch, 1, low_h, low_w), "reused by EFF modules"),
    ]
    records.extend(edge_records)

    for index, shape in enumerate(enc, start=1):
        records.append(
            TensorRecord(
                "edge",
                f"edge_resized_for_f{index}",
                (batch, 1, shape[2], shape[3]),
                "edge map resized to feature scale",
            )
        )

    return records


def parameter_path(name: str) -> str:
    if name.startswith("efe."):
        return "edge"
    if name.startswith("mmsc."):
        return "skip"
    return "main"


def build_parameter_records(
    in_channels: int,
    out_channels: int,
    channels: tuple[int, ...],
    edge_channels: int,
) -> list[ParameterRecord]:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    try:
        from nnunetv2.nets.LMUNet import LMUNet
    except Exception as exc:
        raise RuntimeError(
            "Could not import nnunetv2.nets.LMUNet. Run this script from the LightM-UNet "
            "environment where mamba_ssm and nnunetv2 are installed."
        ) from exc

    model = LMUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        edge_channels=edge_channels,
    )

    records = []
    for name, param in model.named_parameters():
        records.append(
            ParameterRecord(
                path=parameter_path(name),
                name=name,
                shape=tuple(param.shape),
                elements=param.numel(),
                note="trainable parameter",
            )
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic LM-UNet tensor memory breakdown.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=4)
    parser.add_argument("--channels", type=parse_channels, default=parse_channels("12,20,32,44,64,72"))
    parser.add_argument("--edge-channels", type=int, default=20)
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument(
        "--param-dtype",
        default="fp32",
        help="Element type used for parameter memory. Current unoptimized model uses fp32.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bytes_per_element = dtype_bytes(args.dtype)
    param_bytes_per_element = dtype_bytes(args.param_dtype)
    # Validate divisibility up front so stage shapes are unambiguous.
    downsampled_hw(args.height, args.width, 5)

    records = build_records(
        batch=args.batch,
        height=args.height,
        width=args.width,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        channels=args.channels,
        edge_channels=args.edge_channels,
    )
    parameter_records = build_parameter_records(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        channels=args.channels,
        edge_channels=args.edge_channels,
    )

    print("LM-UNet basic tensor memory")
    print(f"Input: [{args.batch}, {args.in_channels}, {args.height}, {args.width}]")
    print(f"Output channels: {args.out_channels}")
    print(f"Channels: {args.channels}")
    print(f"Edge channels: {args.edge_channels}")
    print(f"Feature dtype: {args.dtype} ({bytes_per_element} bytes/element)")
    print(f"Parameter dtype: {args.param_dtype} ({param_bytes_per_element} bytes/element)")
    print_tensor_table("Feature Map Tensor Footprint By Path", records, bytes_per_element)
    print_path_summary(records, bytes_per_element)
    print_parameter_table("Weight/Bias Parameter Memory By Path", parameter_records, param_bytes_per_element)
    print_parameter_summary(parameter_records, param_bytes_per_element)

    print("\nNote:")
    print("This is a catalog of feature-map tensors and trainable weight/bias parameter tensors.")
    print("It is the static architecture view: stage inputs/outputs, skip tensors, edge tensors, and parameters.")
    print("It does not include extra temporary/live buffers needed while computing those tensors.")
    print("Use working_memory.py for the working/dynamic memory view.")


if __name__ == "__main__":
    main()

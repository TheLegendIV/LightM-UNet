from __future__ import annotations

import argparse

from memory_utils import (
    TensorRecord,
    WorkingRecord,
    dtype_bytes,
    downsampled_hw,
    encoder_shapes,
    mib,
    parse_channels,
    print_path_summary,
    print_tensor_table,
    print_working_summary,
    print_working_table,
)


def tensor(path: str, name: str, shape: tuple[int, ...], note: str = "") -> TensorRecord:
    return TensorRecord(path, name, shape, note)


def stage_group(module_name: str) -> str:
    if module_name.startswith("encoder_stage_"):
        return module_name.split(".", 1)[0]
    if module_name.startswith("encoder_pv_"):
        pv_index = int(module_name.split("_")[2].split(".")[0])
        return f"encoder_stage_{pv_index + 3}"
    if module_name.startswith("efe."):
        return "edge_feature_extraction"
    if module_name.startswith("efe_pv_"):
        return "edge_feature_extraction"
    if module_name.startswith("mmsc.eff_"):
        return module_name.split(".", 2)[1]
    if module_name.startswith("mmsc.mmsa_"):
        return module_name.split(".", 2)[1]
    if module_name.startswith("eff_pv_"):
        eff_index = module_name.split("_")[2].split(".")[0]
        return f"eff_{eff_index}"
    if module_name.startswith("decoder_"):
        return module_name.split(".", 1)[0]
    return module_name.split(".", 1)[0]


def print_stage_peak_summary(records: list[WorkingRecord], bytes_per_element: int) -> None:
    peaks: dict[str, WorkingRecord] = {}
    for record in records:
        group = stage_group(record.module)
        if group not in peaks or record.bytes(bytes_per_element) > peaks[group].bytes(bytes_per_element):
            peaks[group] = record

    print("\n=== Peak Working Memory By Stage/Module ===")
    print(f"{'Stage/Module':<28} {'Path':<10} {'Peak MiB':>10}  Peak Source")
    print("-" * 88)
    for group in sorted(peaks):
        record = peaks[group]
        print(
            f"{group:<28} "
            f"{record.path:<10} "
            f"{mib(record.bytes(bytes_per_element)):>10.3f}  "
            f"{record.module} ({record.reason})"
        )
    largest = max(peaks.values(), key=lambda record: record.bytes(bytes_per_element))
    print("-" * 88)
    print(
        "Largest stage/module peak: "
        f"{stage_group(largest.module)} = {mib(largest.bytes(bytes_per_element)):.3f} MiB "
        f"from {largest.module}"
    )


def build_local_working_records(
    batch: int,
    height: int,
    width: int,
    channels: tuple[int, ...],
    edge_channels: int,
    include_mamba_estimate: bool,
) -> list[WorkingRecord]:
    enc = encoder_shapes(batch, height, width, channels)
    records: list[WorkingRecord] = []

    for index, shape in enumerate(enc, start=1):
        x = tensor("main", f"f{index}", shape)
        records.append(
            WorkingRecord(
                "main",
                f"encoder_stage_{index}.local_residual",
                "x + local(x)",
                (x, x),
                "same-shape residual requires original and transformed feature",
            )
        )
        records.append(
            WorkingRecord(
                "main",
                f"encoder_stage_{index}.global_residual",
                "x + global(x)",
                (x, x),
                "same-shape residual requires original and transformed feature",
            )
        )
        records.append(
            WorkingRecord(
                "main",
                f"encoder_stage_{index}.ema_residual",
                "x + ema(x)",
                (x, x),
                "EMA output is added back to original feature",
            )
        )

    low = tensor("edge", "efe_low", enc[1])
    high_up = tensor("edge", "efe_high_upsampled", (batch, channels[1], enc[1][2], enc[1][3]))
    concat = tensor(
        "edge",
        "efe_concat",
        (batch, channels[1] * 2, enc[1][2], enc[1][3]),
    )
    records.append(
        WorkingRecord(
            "edge",
            "efe.concat_low_high",
            "concat",
            (low, high_up, concat),
            "low detail feature and upsampled high semantic feature meet here",
        )
    )
    hidden = tensor("edge", "efe_hidden", (batch, edge_channels, enc[1][2], enc[1][3]))
    records.append(
        WorkingRecord(
            "edge",
            "efe.fuse_residual_like_peak",
            "fuse hidden",
            (hidden, hidden),
            "approximate hidden feature plus transformed version inside fuse PVMamba",
        )
    )

    for index, shape in enumerate(enc, start=1):
        feature = tensor("skip", f"feature_f{index}", shape)
        edge = tensor("edge", f"edge_for_f{index}", (batch, 1, shape[2], shape[3]))
        records.append(
            WorkingRecord(
                "skip",
                f"mmsc.eff_{index}.edge_gate",
                "x + x*edge",
                (feature, feature, edge),
                "edge-guided residual gate",
            )
        )
        att = tensor("skip", f"channel_att_f{index}", (batch, shape[1], 1, 1))
        records.append(
            WorkingRecord(
                "skip",
                f"mmsc.eff_{index}.channel_gate",
                "feature * att",
                (feature, att),
                "per-channel attention vector is small compared with feature",
            )
        )
        spatial_att = tensor("skip", f"spatial_att_f{index}", (batch, 1, shape[2], shape[3]))
        records.append(
            WorkingRecord(
                "skip",
                f"mmsc.mmsa_{index}.spatial_gate",
                "x + x*att",
                (feature, feature, spatial_att),
                "spatial attention residual boost",
            )
        )

    decoder_pairs = [
        (enc[5], enc[4], channels[4], "decoder_1_to_f5"),
        ((batch, channels[4], enc[4][2], enc[4][3]), enc[3], channels[3], "decoder_2_to_f4"),
        ((batch, channels[3], enc[3][2], enc[3][3]), enc[2], channels[2], "decoder_3_to_f3"),
        ((batch, channels[2], enc[2][2], enc[2][3]), enc[1], channels[1], "decoder_4_to_f2"),
        ((batch, channels[1], enc[1][2], enc[1][3]), enc[0], channels[0], "decoder_5_to_f1"),
    ]
    for x_shape, skip_shape, out_channel, name in decoder_pairs:
        up = tensor("main", f"{name}_upsampled", (batch, x_shape[1], skip_shape[2], skip_shape[3]))
        skip = tensor("skip", f"{name}_skip", skip_shape)
        cat = tensor(
            "main",
            f"{name}_concat",
            (batch, x_shape[1] + skip_shape[1], skip_shape[2], skip_shape[3]),
        )
        records.append(
            WorkingRecord(
                "skip",
                f"{name}.concat",
                "upsample + skip concat",
                (up, skip, cat),
                "decoder must combine current stream with retained skip",
            )
        )
        out = tensor("main", f"{name}_out", (batch, out_channel, skip_shape[2], skip_shape[3]))
        records.append(
            WorkingRecord(
                "main",
                f"{name}.stage_residual_peak",
                "decoder LMStage residual",
                (out, out),
                "same-shape local/global residual peak after reduction",
            )
        )

    if include_mamba_estimate:
        for path, prefix, shapes in (
            ("main", "encoder_pv", enc[3:]),
            ("edge", "efe_pv", (enc[1], enc[5], (batch, edge_channels, enc[1][2], enc[1][3]))),
            ("skip", "eff_pv", enc),
        ):
            for index, shape in enumerate(shapes, start=1):
                b, c, h, w = shape
                tokens = h * w
                # PVMamba splits C into four chunks. Each chunk uses Mamba with
                # d_inner = expand * (C / 4). With expand=2, the combined
                # in-projection temporary across four chunks is 4C, and the
                # combined SSM-like inner feature is 2C.
                flat = tensor(path, f"{prefix}_{index}_flat", (b, tokens, c))
                in_proj = tensor(path, f"{prefix}_{index}_in_proj", (b, tokens, 4 * c))
                ssm_like = tensor(path, f"{prefix}_{index}_ssm_like", (b, tokens, 2 * c))
                records.append(
                    WorkingRecord(
                        path,
                        f"{prefix}_{index}.pvmamba_internal",
                        "approx Mamba temp",
                        (flat, in_proj, ssm_like),
                        "analytical full-sequence upper-bound; tiled FPGA design may reduce this",
                    )
                )

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming FPGA-style LM-UNet memory estimate.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--channels", type=parse_channels, default=parse_channels("12,20,32,44,64,72"))
    parser.add_argument("--edge-channels", type=int, default=20)
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument(
        "--no-mamba-estimate",
        action="store_true",
        help="Skip analytical full-sequence Mamba working-buffer estimates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bytes_per_element = dtype_bytes(args.dtype)
    downsampled_hw(args.height, args.width, 5)

    working = build_local_working_records(
        batch=args.batch,
        height=args.height,
        width=args.width,
        channels=args.channels,
        edge_channels=args.edge_channels,
        include_mamba_estimate=not args.no_mamba_estimate,
    )

    print("LM-UNet working/dynamic memory estimate")
    print(f"Input size: [{args.batch}, {args.in_channels}, {args.height}, {args.width}]")
    print(f"Channels: {args.channels}")
    print(f"Edge channels: {args.edge_channels}")
    print(f"Dtype: {args.dtype} ({bytes_per_element} bytes/element)")
    print("Assumption: static feature-map tensors are cataloged in tensor_memory.py.")
    print("This report only estimates extra live/working tensors needed during computation.")

    print_working_table("Local Working Peaks", working, bytes_per_element)
    print_stage_peak_summary(working, bytes_per_element)
    print_working_summary(working, bytes_per_element)

    largest_working = max(working, key=lambda record: record.bytes(bytes_per_element))
    print("\n=== Deployment-Oriented Summary ===")
    print(
        "Largest single local working set: "
        f"{mib(largest_working.bytes(bytes_per_element)):.3f} MiB "
        f"({largest_working.module})"
    )
    print("\nInterpretation:")
    print("Static tensors such as raw skips, edge maps, decoder outputs, and parameters are in tensor_memory.py.")
    print("This file estimates extra working memory while producing those tensors.")
    print("Local working peaks are candidates for tiling, BRAM/URAM buffering, in-place scheduling, or recomputation.")
    print("Mamba internal estimates are full-sequence upper bounds, not measured GPU workspace.")


if __name__ == "__main__":
    main()

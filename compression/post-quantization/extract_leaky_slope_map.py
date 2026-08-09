"""Extracts a per-block leaky-slope map from an already-trained FP32
prelu_variant="nonneg_block" ENet checkpoint, for feeding into ptq.py's
--leaky-slope-map-file (INT8/INT4 PTQ, builds QuantDecomposedLeakyAct at
each mapped block instead of defaulting to plain QuantReLU) or
collect_results.py's --leaky-slope-map (evaluating the resulting quantized
checkpoint).

This step is NOT optional for any nonneg_block-trained model: skipping it
(i.e. running ptq.py/collect_results.py with no slope map at all) silently
builds plain QuantReLU everywhere, discarding every block's real learned
negative-slope activation -- a genuine architecture mismatch on top of
quantization, not honest quantization noise. Confirmed empirically on
S19: the buggy no-slope-map PTQ run measured dice=0.4383; the corrected
run with this script's own slope map measured meaningfully higher (see
compression/results.csv's experiment_s19_ptq_int8 row for the current
number). ENet.py's own collect_prelu_block_means already handles
NonNegativePReLU (S19's activation class) the same way it handles a
real per-channel PReLU checkpoint -- this script is a thin CLI wrapper
around it, nothing new algorithmically.

Usage (S19):
    python compression/post-quantization/extract_leaky_slope_map.py \
        --net-name nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid \
        --channels 4,16,32,16,4 --bottlenecks 4,12,12,2,1 --decoder-type upsample_conv \
        --use-asymmetric 0 --context-pattern dense_dilation_reg_interleaved_double_mid \
        --separable-dilated 1 --dsc-no-projection 0 \
        --out-file compression/post-quantization/slope_maps/19_reginterleaved_separable_nonneg_block_double_mid.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet, collect_prelu_block_means  # noqa: E402

NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"


def parse_tuple5(value: str, name: str) -> tuple[int, ...]:
    parts = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if len(parts) != 5:
        raise ValueError(f"--{name} must have exactly 5 comma-separated integers, got {value!r}.")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--net-name", required=True, help="config_name of the trained prelu_variant=nonneg_block FP32 checkpoint.")
    parser.add_argument("--dataset-name", default="Dataset509_ARCADE_1x1_4c")
    parser.add_argument("--channels", required=True, type=lambda v: parse_tuple5(v, "channels"))
    parser.add_argument("--bottlenecks", default="4,8,8,2,1", type=lambda v: parse_tuple5(v, "bottlenecks"))
    parser.add_argument("--decoder-type", default="upsample_conv", choices=["max_unpool", "upsample_conv"])
    parser.add_argument("--use-dilated", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-asymmetric", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-strided", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-dsc", type=int, default=0, choices=[0, 1])
    parser.add_argument("--context-pattern", default="default")
    parser.add_argument("--dsc-no-projection", type=int, default=0, choices=[0, 1])
    parser.add_argument("--separable-dilated", type=int, default=0, choices=[0, 1])
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--plans-name", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--out-file", required=True, type=Path)
    args = parser.parse_args()

    model = ENet(
        in_channels=args.in_channels, out_channels=args.out_channels, channels=args.channels,
        bottlenecks_per_stage=args.bottlenecks, decoder_type=args.decoder_type,
        use_dilated=bool(args.use_dilated), use_asymmetric=bool(args.use_asymmetric),
        use_strided=bool(args.use_strided), use_dsc=bool(args.use_dsc),
        context_pattern=args.context_pattern, dsc_no_projection=bool(args.dsc_no_projection),
        separable_dilated=bool(args.separable_dilated),
        use_prelu=True, prelu_variant="nonneg_block",
    )
    ckpt_path = (NNUNET_RESULTS / args.dataset_name
                 / f"{args.net_name}__{args.plans_name}__{args.configuration}"
                 / f"fold_{args.fold}" / args.checkpoint_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["network_weights"], strict=True)
    print(f"Loaded {ckpt_path}: strict load OK")

    slope_map = collect_prelu_block_means(model)
    print(f"Extracted {len(slope_map)} block slopes:")
    for k, v in slope_map.items():
        print(f"  {k}: {v:.4f}")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(slope_map, f, indent=2)
    print(f"Wrote {args.out_file}")


if __name__ == "__main__":
    main()

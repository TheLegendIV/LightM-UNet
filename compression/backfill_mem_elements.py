"""One-off (but rerunnable) backfill: computes mem_elements for every row
already in results.csv, for configs collected before that column existed.

Purely architecture-derived (channels, bottlenecks_per_stage, decoder_type,
ops_flags) -- no checkpoint or prediction dir needed, unlike collect_results.py's
own inference-driven flow, so this is much cheaper than re-running every stage's
job just to add one new column.

Usage:
    python compression/backfill_mem_elements.py [--results-csv PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from utils import count_buffer_elements  # noqa: E402

# Constant across every row in this sweep (Dataset509_ARCADE_1x1_4c: 1-channel
# grayscale input, 5 segmentation classes) -- not itself a results.csv column.
IN_CHANNELS = 1
OUT_CHANNELS = 5
INPUT_HW = (512, 512)

_BOOL_FIELDS = (
    "dilated", "asymmetric", "strided", "dsc", "shallow_dilation", "separable_dilated",
    "merge_dilated_pairs", "dsc_dilated_only", "double_projections", "two_block_skip",
    "dsc_no_projection", "shallow_dilation_wide", "shallow_dilation_dense",
    "dsc_no_projection_context_only", "reg_bookend_dsc",
)


def parse_ops_flags(ops_flags: str) -> dict:
    """ops_flags is a comma-joined key=value string that grew new fields over
    the sweep's lifetime (older rows lack shallow_dilation_wide/dense,
    dsc_no_projection_context_only, reg_bookend_dsc entirely) -- parse
    whatever's present and let missing keys fall back to ENet's own
    constructor defaults (all False) rather than assuming every row has
    every key."""
    parsed: dict[str, str] = {}
    for part in ops_flags.split(","):
        key, _, value = part.partition("=")
        parsed[key] = value
    return parsed


def build_model(row: pd.Series) -> ENet:
    flags = parse_ops_flags(row["ops_flags"])
    bottlenecks = tuple(int(n) for n in str(row["bottlenecks_per_stage"]).split(","))
    channels = (
        int(row["f_i"]), int(row["f1"]), int(row["f2"]), int(row["f3"]), int(row["f4"]), int(row["f5"]),
    )
    kwargs = {}
    for field in _BOOL_FIELDS:
        raw = flags.get(field)
        kwargs[field] = raw == "1" if raw is not None else False
    # prelu is stored as "n/a(quant-forces-relu)" for quantized rows -- not a
    # real per-run flag in that case (see collect_results.py's own comment on
    # why); every row in this sweep is quant_bits=32 so far, but fall back to
    # ENet's own default (True) defensively rather than crash on int("n/a").
    prelu_raw = flags.get("prelu", "1")
    use_prelu = prelu_raw == "1" if prelu_raw in ("0", "1") else True

    return ENet(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        channels=channels,
        bottlenecks_per_stage=bottlenecks,
        decoder_type=row["decoder_type"],
        use_dilated=kwargs["dilated"],
        use_asymmetric=kwargs["asymmetric"],
        use_strided=kwargs["strided"],
        use_dsc=kwargs["dsc"],
        context_pattern=flags.get("context_pattern", "default"),
        use_prelu=use_prelu,
        shallow_dilation=kwargs["shallow_dilation"],
        separable_dilated=kwargs["separable_dilated"],
        merge_dilated_pairs=kwargs["merge_dilated_pairs"],
        dsc_dilated_only=kwargs["dsc_dilated_only"],
        double_projections=kwargs["double_projections"],
        two_block_skip=kwargs["two_block_skip"],
        dsc_no_projection=kwargs["dsc_no_projection"],
        shallow_dilation_wide=kwargs["shallow_dilation_wide"],
        shallow_dilation_dense=kwargs["shallow_dilation_dense"],
        dsc_no_projection_context_only=kwargs["dsc_no_projection_context_only"],
        reg_bookend_dsc=kwargs["reg_bookend_dsc"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-csv", type=Path, default=REPO_ROOT / "compression" / "results.csv")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without writing.")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    if "mem_elements" not in df.columns:
        df["mem_elements"] = pd.NA

    n_filled, n_skipped = 0, 0
    for idx, row in df.iterrows():
        if pd.notna(row.get("mem_elements")):
            n_skipped += 1
            continue
        model = build_model(row)
        result = count_buffer_elements(model, IN_CHANNELS, INPUT_HW)
        df.at[idx, "mem_elements"] = result["total"]
        print(f"{row['config_name']:55s} mem_elements={result['total']:>10,}")
        n_filled += 1

    print(f"\nFilled {n_filled} rows, skipped {n_skipped} already-populated rows.")
    if args.dry_run:
        print("--dry-run: not writing.")
        return 0
    df.to_csv(args.results_csv, index=False)
    print(f"Wrote {args.results_csv}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Summarize graduated upscale/ configs' real test-set Dice, alongside the
Original/E1 baselines for reference, into their own CSV -- separate from
compression/results.csv (which has every stage mixed together) and from
upscale/results/pareto_e15/summary.csv (which is the 15-epoch proxy, not
real test-set performance).

Usage:
    python upscale/summarize_graduated.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from archive.upscale.pareto_common import repo_root

BASELINE_CONFIGS = ["nnUNetTrainerENet_Original", "nnUNetTrainerENet_E1"]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Summarize upscale/ graduated configs' real Dice.")
    parser.add_argument("--results-csv", type=Path, default=root / "compression" / "results.csv")
    parser.add_argument("--out-csv", type=Path, default=root / "upscale" / "results" / "graduated_summary.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.results_csv.exists():
        print(f"{args.results_csv} not found.")
        return 1

    df = pd.read_csv(args.results_csv)
    graduated = df[df["stage"] == "upscale_graduate"].copy()
    if graduated.empty:
        print("No stage=upscale_graduate rows found -- run compression/collect_results.py for the graduated checkpoints first.")
        return 1

    baselines = df[df["config_name"].isin(BASELINE_CONFIGS)].copy()
    original_dice = baselines.loc[baselines["config_name"] == "nnUNetTrainerENet_Original", "dice"]
    original_dice = float(original_dice.iloc[0]) if not original_dice.empty else None

    combined = pd.concat([baselines, graduated], ignore_index=True)
    combined["dice_vs_original"] = combined["dice"] - original_dice if original_dice is not None else None
    combined["beats_original"] = combined["dice"] > original_dice if original_dice is not None else None
    combined = combined.sort_values("dice", ascending=False)

    cols = ["config_name", "stage", "params", "dice", "cldice", "n_components",
            "dice_vs_original", "beats_original", "epochs", "converged_flag"]
    combined[cols].to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(combined)} rows)")

    print("\n=== Graduated configs vs. Original/E1 baselines (real test-set Dice) ===")
    header = f"{'config_name':<32}{'params':>10}{'dice':>10}{'cldice':>10}{'vs_orig':>10}{'beats_orig':>12}"
    print(header)
    print("-" * len(header))
    for _, row in combined.iterrows():
        delta = f"{row['dice_vs_original']:+.4f}" if pd.notna(row["dice_vs_original"]) else "n/a"
        beats = str(bool(row["beats_original"])) if pd.notna(row["beats_original"]) else "n/a"
        print(f"{row['config_name']:<32}{int(row['params']):>10}{row['dice']:>10.4f}{row['cldice']:>10.4f}{delta:>10}{beats:>12}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

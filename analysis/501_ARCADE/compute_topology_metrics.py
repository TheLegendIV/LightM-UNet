"""Compute component-overlap / fragmentation / skeleton-connectivity
diagnostics (see segmentation_topology.py) for one or all labelsPr_*
prediction folders in Dataset501_ARCADE, writing per-case and per-model
summary CSVs into results/ -- same layout as the dice/precision/recall
metrics produced by preview_results.ipynb.

Usage:
    python compute_topology_metrics.py                  # all labelsPr_* dirs
    python compute_topology_metrics.py nnUNetTrainerENet_E1 nnUNetTrainerENet_Original
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segmentation_topology as topo

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "data" / "nnUNet_raw" / "Dataset501_ARCADE"
LABELS_TS_DIR = DATASET_DIR / "labelsTs"
METRICS_DIR = Path(__file__).resolve().parent / "results"


def discover_net_names() -> list[str]:
    return sorted(
        p.name[len("labelsPr_"):]
        for p in DATASET_DIR.glob("labelsPr_*")
        if p.is_dir()
    )


def run_for_model(net_name: str) -> pd.DataFrame | None:
    pred_dir = DATASET_DIR / f"labelsPr_{net_name}"
    if not pred_dir.exists():
        print(f"Skipping {net_name}: {pred_dir} does not exist")
        return None

    rows = [
        topo.evaluate_case(case_id, gt, pred)
        for case_id, gt, pred in topo.iter_matched_cases(LABELS_TS_DIR, pred_dir)
    ]

    if not rows:
        print(f"Skipping {net_name}: no matched cases")
        return None

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / f"{net_name}_topology_metrics.csv", index=False)

    summary = pd.DataFrame([{
        "model": net_name,
        "n_cases": len(df),
        "mean_component_overlap": df["mean_component_overlap"].mean(),
        "area_weighted_component_overlap": df["area_weighted_component_overlap"].mean(),
        "cases_with_pure_fp_component": int((df["n_pure_fp_components"] > 0).sum()),
        "mean_pure_fp_pixel_frac": df["pure_fp_pixel_frac"].mean(),
        "mean_pred_components": df["pred_components"].mean(),
        "mean_gt_components": df["gt_components"].mean(),
        "mean_component_ratio": df["component_ratio"].replace([np.inf, -np.inf], np.nan).mean(),
        "mean_noise_islands": df["pred_noise_islands"].mean(),
        "mean_gt_skeleton_endpoints": df["gt_skeleton_n_endpoints"].mean(),
        "mean_pred_skeleton_endpoints": df["pred_skeleton_n_endpoints"].mean(),
        "mean_gt_skeleton_branch_points": df["gt_skeleton_n_branch_points"].mean(),
        "mean_pred_skeleton_branch_points": df["pred_skeleton_n_branch_points"].mean(),
        "mean_gt_skeleton_components": df["gt_skeleton_n_skeleton_components"].mean(),
        "mean_pred_skeleton_components": df["pred_skeleton_n_skeleton_components"].mean(),
    }])
    summary.to_csv(METRICS_DIR / f"{net_name}_topology_summary.csv", index=False)
    print(f"{net_name}: {len(df)} cases -> {net_name}_topology_metrics.csv / {net_name}_topology_summary.csv")
    return summary


def main() -> None:
    METRICS_DIR.mkdir(exist_ok=True)
    net_names = sys.argv[1:] or discover_net_names()
    summaries = [s for s in (run_for_model(name) for name in net_names) if s is not None]
    if not summaries:
        print("No models produced results.")
        return
    combined = pd.concat(summaries, ignore_index=True)
    combined = combined.sort_values("area_weighted_component_overlap", ascending=True, ignore_index=True)
    combined.to_csv(METRICS_DIR / "summary_topology_metrics.csv", index=False)
    print(f"\nWrote combined summary for {len(combined)} models -> summary_topology_metrics.csv")


if __name__ == "__main__":
    main()

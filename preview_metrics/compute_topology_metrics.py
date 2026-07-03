"""Compute class-purity / fragmentation / anatomical-consistency diagnostics
(see segmentation_topology.py) for one or all labelsPr_* prediction folders
in Dataset501_ARCADE, writing per-case and per-model summary CSVs into this
directory -- same layout as the dice/precision/recall metrics produced by
preview_results.ipynb.

Usage:
    python compute_topology_metrics.py                  # all labelsPr_* dirs
    python compute_topology_metrics.py ENetOriginal LMUNet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segmentation_topology as topo

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "data" / "nnUNet_raw" / "Dataset501_ARCADE"
LABELS_TS_DIR = DATASET_DIR / "labelsTs"
METRICS_DIR = Path(__file__).resolve().parent


def load_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


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

    rows = []
    for gt_path in sorted(LABELS_TS_DIR.glob("*.png")):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt = load_mask(gt_path)
        pred = load_mask(pred_path)
        if pred.shape != gt.shape:
            pred = np.asarray(
                Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST),
                dtype=np.uint8,
            )
        rows.append(topo.evaluate_case(gt_path.stem, gt, pred))

    if not rows:
        print(f"Skipping {net_name}: no matched cases")
        return None

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / f"{net_name}_topology_metrics.csv", index=False)

    lad_lcx = df[df["territory"] == "LAD_LCX"]
    rca = df[df["territory"] == "RCA"]
    branch_valid = df[df["branch_n_paths"] > 0]

    summary = pd.DataFrame([{
        "model": net_name,
        "n_cases": len(df),
        "territory_leakage_rate_mean": df["territory_leakage_rate"].mean(),
        "territory_leakage_rate_lad_lcx": lad_lcx["territory_leakage_rate"].mean() if len(lad_lcx) else np.nan,
        "territory_leakage_rate_rca": rca["territory_leakage_rate"].mean() if len(rca) else np.nan,
        "cases_with_leakage": int((df["leaked_px"] > 0).sum()),
        "mean_component_purity": df["mean_purity"].mean(),
        "area_weighted_purity": df["area_weighted_purity"].mean(),
        "cases_with_mixed_component": int((df["n_mixed_components"] > 0).sum()),
        "mean_pred_components": df["pred_components"].mean(),
        "mean_gt_components": df["gt_components"].mean(),
        "mean_component_ratio": df["component_ratio"].replace([np.inf, -np.inf], np.nan).mean(),
        "mean_noise_islands": df["pred_noise_islands"].mean(),
        "n_cases_with_branch_paths": len(branch_valid),
        "branch_inconsistent_path_rate_mean": branch_valid["branch_inconsistent_path_rate"].mean() if len(branch_valid) else np.nan,
        "cases_with_branch_inconsistency": int((df["branch_n_inconsistent_paths"] > 0).sum()),
    }])
    summary.to_csv(METRICS_DIR / f"{net_name}_topology_summary.csv", index=False)
    print(f"{net_name}: {len(df)} cases -> {net_name}_topology_metrics.csv / {net_name}_topology_summary.csv")
    return summary


def main() -> None:
    net_names = sys.argv[1:] or discover_net_names()
    summaries = [s for s in (run_for_model(name) for name in net_names) if s is not None]
    if not summaries:
        print("No models produced results.")
        return
    combined = pd.concat(summaries, ignore_index=True)
    combined = combined.sort_values("area_weighted_purity", ascending=True, ignore_index=True)
    combined.to_csv(METRICS_DIR / "summary_topology_metrics.csv", index=False)
    print(f"\nWrote combined summary for {len(combined)} models -> summary_topology_metrics.csv")


if __name__ == "__main__":
    main()

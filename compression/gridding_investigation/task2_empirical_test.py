"""Task 2: empirical gridding test -- checks whether S5-DscNoProjDense's
real validation/test-split errors show spatial periodicity at the periods
Task 1's structural analysis predicts. See gridding_investigation_summary.md.

Reuses the predictions collect_results.py's own test-split inference run
already produced (labelsPr_nnUNetTrainerENet_5_1_dscnoprojection_dense_dilation,
300 held-out test cases -- the same split every Dice number in results.csv
comes from) rather than re-running inference. All images/masks in this
dataset are uniformly 512x512, so no resampling is needed for pixel-aligned
aggregation.

Usage:
    python compression/gridding_investigation/task2_empirical_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "analysis" / "501_ARCADE"))
from nnunetv2.nets.ENet import ENet  # noqa: E402
import segmentation_topology as topo  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "Dataset509_ARCADE_1x1_4c"
DATASET_DIR = REPO_ROOT / "data" / "nnUNet_raw" / DATASET_NAME
LABELS_TS_DIR = DATASET_DIR / "labelsTs"
PREDICTION_DIR = DATASET_DIR / "labelsPr_nnUNetTrainerENet_5_1_dscnoprojection_dense_dilation"
IMAGE_HW = (512, 512)
RATES = (2, 4, 8, 16)  # the schedule the trained checkpoint actually uses (DENSE_DILATION_PATTERN cycles this twice per stage)


def get_downsample_factor() -> int:
    """Same measurement as Task 1's get_stage2_resolution, just returning the
    scalar ratio -- kept as its own function/step per 2.1's spec rather than
    importing Task 1's helper, so this script is runnable standalone."""
    model = ENet(
        in_channels=1, out_channels=5, channels=(4, 16, 32, 16, 4),
        bottlenecks_per_stage=(4, 8, 8, 2, 1), decoder_type="upsample_conv",
        use_dilated=True, use_asymmetric=False, use_strided=True, use_dsc=False,
        context_pattern="dense_dilation", use_prelu=True, dsc_no_projection=True,
    ).eval()
    captured = {}

    def hook(module, inp, out):
        captured["shape"] = inp[0].shape

    handle = model.stage2.register_forward_hook(hook)
    with torch.no_grad():
        model(torch.zeros(1, 1, *IMAGE_HW))
    handle.remove()
    h, w = captured["shape"][-2:]
    assert IMAGE_HW[0] % h == 0 and IMAGE_HW[1] % w == 0, "non-integer downsample factor"
    factor_h, factor_w = IMAGE_HW[0] // h, IMAGE_HW[1] // w
    assert factor_h == factor_w, f"anisotropic downsample factor: {factor_h} vs {factor_w}"
    return factor_h


def compute_error_masks() -> tuple[np.ndarray, np.ndarray, int]:
    img_files = topo.image_files(DATASET_DIR / "imagesTs")
    gt_files = {p.stem: p for p in topo.image_files(LABELS_TS_DIR)}
    pred_files = {p.stem: p for p in topo.image_files(PREDICTION_DIR)}

    fn_aggregate = np.zeros(IMAGE_HW, dtype=np.float64)
    fp_aggregate = np.zeros(IMAGE_HW, dtype=np.float64)
    n = 0
    for img in img_files:
        cid = topo.case_id_from_image(img)
        if cid not in gt_files or cid not in pred_files:
            continue
        gt = topo.load_class_id_mask(gt_files[cid], binarize=False)
        pred = topo.load_class_id_mask(pred_files[cid], binarize=False)
        pred = topo.resize_mask_to(pred, gt.shape)
        if gt.shape != IMAGE_HW:
            continue  # skip any non-uniform outlier rather than silently misaligning the sum
        gt_fg = gt > 0
        pred_fg = pred > 0
        fn_mask = gt_fg & (pred != gt)   # GT vessel, pred background OR wrong class
        fp_mask = (~gt_fg) & pred_fg     # GT background, pred any vessel class
        fn_aggregate += fn_mask
        fp_aggregate += fp_mask
        n += 1
    return fn_aggregate, fp_aggregate, n


def peak_to_background(mag: np.ndarray, cy: int, cx: int, dy: int, dx: int,
                        ring_width: int = 3, exclude_radius: int = 2) -> dict:
    ty, tx = cy + dy, cx + dx
    if not (0 <= ty < mag.shape[0] and 0 <= tx < mag.shape[1]):
        return {"peak": None, "background": None, "ratio": None}
    peak_val = float(mag[ty, tx])
    r = float(np.hypot(dx, dy))
    yy, xx = np.mgrid[0:mag.shape[0], 0:mag.shape[1]]
    dist = np.hypot(xx - cx, yy - cy)
    ring_mask = (dist >= r - ring_width) & (dist <= r + ring_width)
    close_to_target = (np.abs(xx - tx) <= exclude_radius) & (np.abs(yy - ty) <= exclude_radius)
    ring_mask &= ~close_to_target
    background = float(np.median(mag[ring_mask])) if ring_mask.any() else float("nan")
    ratio = peak_val / background if background > 0 else float("inf")
    return {"peak": peak_val, "background": background, "ratio": ratio}


def analyze_map(name: str, aggregate: np.ndarray, downsample_factor: int) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heatmap_path = OUT_DIR / f"{name}_aggregate.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(aggregate, cmap="inferno")
    ax.set_title(f"{name} aggregate (n=300 test cases)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fft = np.fft.fft2(aggregate)
    fft_shifted = np.fft.fftshift(fft)
    mag = np.abs(fft_shifted)
    cy, cx = mag.shape[0] // 2, mag.shape[1] // 2

    spectrum_path = OUT_DIR / f"{name}_fft_spectrum.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.log1p(mag), cmap="viridis")
    ax.set_title(f"{name}: log FFT magnitude spectrum")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(spectrum_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    per_rate = {}
    for rate in RATES:
        period_px = rate * downsample_factor
        freq_bin = int(round(aggregate.shape[0] / period_px))
        horizontal = peak_to_background(mag, cy, cx, dy=0, dx=freq_bin)
        vertical = peak_to_background(mag, cy, cx, dy=freq_bin, dx=0)
        per_rate[str(rate)] = {
            "period_px": period_px,
            "freq_bin": freq_bin,
            "horizontal": horizontal,
            "vertical": vertical,
        }

    return {
        "aggregate_heatmap": str(heatmap_path.relative_to(REPO_ROOT)),
        "fft_spectrum": str(spectrum_path.relative_to(REPO_ROOT)),
        "per_rate": per_rate,
    }


def main() -> int:
    downsample_factor = get_downsample_factor()
    expected_periods = {rate: rate * downsample_factor for rate in RATES}
    print(f"Cumulative downsample factor (input -> stage2/3): {downsample_factor}x")
    print(f"Expected grid periods (input-image pixels): {expected_periods}")

    fn_aggregate, fp_aggregate, n = compute_error_masks()
    print(f"Aggregated FN/FP masks over {n} matched test cases.")

    fn_results = analyze_map("fn", fn_aggregate, downsample_factor)
    fp_results = analyze_map("fp", fp_aggregate, downsample_factor)

    results = {
        "downsample_factor": downsample_factor,
        "expected_periods_px": expected_periods,
        "n_cases": n,
        "fn": fn_results,
        "fp": fp_results,
    }
    out_json = Path(__file__).resolve().parent / "task2_empirical_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")

    print("\nPeak-to-background ratios:")
    for kind, res in (("FN", fn_results), ("FP", fp_results)):
        for rate, entry in res["per_rate"].items():
            h, v = entry["horizontal"]["ratio"], entry["vertical"]["ratio"]
            print(f"  {kind} rate={rate} period={entry['period_px']}px: horizontal={h:.2f}x vertical={v:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

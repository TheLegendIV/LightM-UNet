"""Task 1: structural (impulse-response) gridding test for stacked dilated
convolutions -- see gridding_investigation_summary.md for the full writeup.

Purely architectural: no trained weights, no GPU needed. Determines the
real stage2/3 spatial resolution from the actual S5-DscNoProjDense model
(the "dense dilation, no projection" checkpoint this investigation is
about) via a forward hook, rather than hardcoding it, then builds a raw
4-layer Conv2d stack per candidate schedule and traces which input pixels
are reachable from a single impulse.

Usage:
    python compression/gridding_investigation/task1_structural_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
from nnunetv2.nets.ENet import ENet  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCHEDULES = {
    "current_2_4_8_16": (2, 4, 8, 16),
    "schedule_A_1_5_7_17": (1, 5, 7, 17),
    "schedule_B_1_4_9_16": (1, 4, 9, 16),
}


def get_stage2_resolution(input_hw: tuple[int, int] = (512, 512)) -> tuple[int, int]:
    """Builds the real S5-DscNoProjDense architecture (channels=4,16,32,16,4,
    dsc_no_projection=1, context_pattern=dense_dilation, use_asymmetric=0,
    use_prelu=1 -- exactly compression/slurm/stage_5_arch_probe_pairs_array.job
    task 0's config) and hooks stage2's input to read its real spatial shape,
    instead of assuming a fixed downsample factor."""
    model = ENet(
        in_channels=1, out_channels=5, channels=(4, 16, 32, 16, 4),
        bottlenecks_per_stage=(4, 8, 8, 2, 1), decoder_type="upsample_conv",
        use_dilated=True, use_asymmetric=False, use_strided=True, use_dsc=False,
        context_pattern="dense_dilation", use_prelu=True, dsc_no_projection=True,
    ).eval()

    captured: dict[str, torch.Size] = {}

    def hook(module, inp, out):
        captured["shape"] = inp[0].shape

    handle = model.stage2.register_forward_hook(hook)
    dummy = torch.zeros(1, 1, *input_hw)
    with torch.no_grad():
        model(dummy)
    handle.remove()

    _, _, h, w = captured["shape"]
    return int(h), int(w)


def build_raw_stack(rates: tuple[int, ...]) -> nn.Sequential:
    """4 raw Conv2d layers, kernel=3, 'same' padding (padding=dilation),
    weight=1 / bias=0, single channel -- no BN/activation/residual, so the
    only thing that can produce a non-zero output is spatial reachability,
    not channel mixing or learned weighting."""
    layers = []
    for rate in rates:
        conv = nn.Conv2d(1, 1, kernel_size=3, padding=rate, dilation=rate, bias=True)
        with torch.no_grad():
            conv.weight.fill_(1.0)
            conv.bias.fill_(0.0)
        layers.append(conv)
    return nn.Sequential(*layers)


def impulse_response(rates: tuple[int, ...], canvas_hw: tuple[int, int]) -> np.ndarray:
    stack = build_raw_stack(rates)
    h, w = canvas_hw
    x = torch.zeros(1, 1, h, w)
    x[0, 0, h // 2, w // 2] = 1.0
    with torch.no_grad():
        out = stack(x)
    response = out[0].sum(dim=0).numpy()  # sum across output channels (only 1 here)
    return response


def crop_to_rf(response: np.ndarray, center: tuple[int, int], rf_half: int) -> np.ndarray:
    cy, cx = center
    y0, y1 = max(0, cy - rf_half), min(response.shape[0], cy + rf_half + 1)
    x0, x1 = max(0, cx - rf_half), min(response.shape[1], cx + rf_half + 1)
    return response[y0:y1, x0:x1]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canvas_hw = get_stage2_resolution()
    print(f"Real stage2/3 spatial resolution (from S5-DscNoProjDense, 512x512 input): {canvas_hw}")

    results = []
    for name, rates in SCHEDULES.items():
        rf_size = 1 + 2 * sum(rates)
        rf_half = sum(rates)
        response = impulse_response(rates, canvas_hw)
        center = (canvas_hw[0] // 2, canvas_hw[1] // 2)
        cropped = crop_to_rf(response, center, rf_half)
        reached = cropped > 1e-6
        coverage_ratio = float(reached.sum()) / reached.size

        heatmap_path = OUT_DIR / f"gridding_impulse_{name}.png"
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(reached, cmap="gray_r", interpolation="nearest")
        ax.set_title(f"{name}: rates={rates}\ncoverage={coverage_ratio:.3f} (crop {cropped.shape[0]}x{cropped.shape[1]}, theoretical RF {rf_size}x{rf_size})")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"{name}: rates={rates} RF={rf_size}x{rf_size} crop_shape={cropped.shape} "
              f"coverage_ratio={coverage_ratio:.4f} -> {heatmap_path.name}")

        results.append({
            "schedule_name": name,
            "rates": list(rates),
            "receptive_field_size": rf_size,
            "canvas_resolution": list(canvas_hw),
            "crop_shape": list(cropped.shape),
            "coverage_ratio": coverage_ratio,
            "heatmap_path": str(heatmap_path.relative_to(REPO_ROOT)),
        })

    out_json = Path(__file__).resolve().parent / "task1_structural_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    cur = next(r for r in results if r["schedule_name"] == "current_2_4_8_16")
    a = next(r for r in results if r["schedule_name"] == "schedule_A_1_5_7_17")
    b = next(r for r in results if r["schedule_name"] == "schedule_B_1_4_9_16")
    gap_cur_a = (a["coverage_ratio"] - cur["coverage_ratio"]) * 100
    gap_b_a = abs(b["coverage_ratio"] - a["coverage_ratio"]) * 100
    gap_b_cur = abs(b["coverage_ratio"] - cur["coverage_ratio"]) * 100
    print(f"\nCoverage gap (A - current) = {gap_cur_a:.1f} pp")
    print(f"Coverage gap (B vs A) = {gap_b_a:.1f} pp, (B vs current) = {gap_b_cur:.1f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Plots every PReLU module's LEARNED slope (the per-channel `weight` a
trained PReLU(x) = max(0,x) + weight*min(0,x) ends up with) against network
depth, for two trained checkpoints: S5-SeparableDense (dense_dilation +
separable_dilated, projection kept) and S6-RegInterleaved (dense_dilation_
reg_interleaved + dsc_no_projection, the config S8-ReLU is a plain-ReLU
ablation of -- picked here specifically because it's the PReLU sibling, so
there's an actual learned slope to plot).

"Depth" = sequential index into model.named_modules() restricted to
nn.PReLU instances, which follows forward-pass order for this codebase
(attributes are assigned in ENet.__init__ in the same order they're
applied in forward()) -- not a designed axis, just "how far into the
network this activation sits." A weight near 1 means the learned PReLU is
nearly linear/identity for negative inputs; near 0 means it's nearly a
plain ReLU (kills negative inputs); each point's vertical spread (min/max
band) is the per-channel range at that one activation site.

Usage:
    python compression/plot_prelu_activations.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402

NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"
DATASET_NAME = "Dataset509_ARCADE_1x1_4c"

INK, SECONDARY_INK, MUTED, GRID, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb",
)
STAGE_ORDER = [
    "initial", "down1", "regular1", "down2", "stage2", "proj2_to_3", "stage3",
    "up4", "regular4", "up5", "regular5", "final",
]
STAGE_BG = {
    "initial": "#eef1f2", "down1": "#e7edf7", "regular1": "#eef1f2",
    "down2": "#e7edf7", "stage2": "#f3ecf7", "proj2_to_3": "#eef1f2",
    "stage3": "#f3ecf7", "up4": "#e7f7f0", "regular4": "#eef1f2",
    "up5": "#e7f7f0", "regular5": "#eef1f2", "final": "#f7f0e7",
}

CONFIGS = {
    "5_6_separable_dense_dilation": dict(
        label="S5-SeparableDense",
        bottlenecks_per_stage=(4, 8, 8, 2, 1),
        context_pattern="dense_dilation",
        dsc_no_projection=False,
        separable_dilated=True,
    ),
    "6_1_reg_interleaved": dict(
        label="S6-RegInterleaved",
        bottlenecks_per_stage=(4, 11, 11, 2, 1),
        context_pattern="dense_dilation_reg_interleaved",
        dsc_no_projection=True,
        separable_dilated=False,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "compression" / "results")
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pth")
    return parser.parse_args()


def build_and_load(net_name: str, spec: dict, checkpoint_name: str) -> nn.Module:
    model = ENet(
        in_channels=1, out_channels=5, channels=(4, 16, 32, 16, 4),
        bottlenecks_per_stage=spec["bottlenecks_per_stage"], decoder_type="upsample_conv",
        use_dilated=True, use_asymmetric=False, use_strided=True, use_prelu=True,
        use_dsc=False, separable_dilated=spec["separable_dilated"],
        dsc_no_projection=spec["dsc_no_projection"], context_pattern=spec["context_pattern"],
    )
    ckpt_path = (
        NNUNET_RESULTS / DATASET_NAME / f"nnUNetTrainerENet_{net_name}__nnUNetPlans__2d"
        / "fold_0" / checkpoint_name
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["network_weights"])
    model.eval()
    return model


def collect_prelu_stats(model: nn.Module) -> list[dict]:
    stats = []
    for name, module in model.named_modules():
        if isinstance(module, nn.PReLU):
            w = module.weight.detach()
            stage = name.split(".")[0]
            stats.append(dict(
                name=name, stage=stage,
                mean=float(w.mean()), min=float(w.min()), max=float(w.max()), std=float(w.std()),
                n_channels=w.numel(),
            ))
    return stats


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for net_name, spec in CONFIGS.items():
        model = build_and_load(net_name, spec, args.checkpoint_name)
        stats = collect_prelu_stats(model)
        all_stats[net_name] = stats
        print(f"{spec['label']} ({net_name}): {len(stats)} PReLU sites")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), facecolor=SURFACE, sharey=True)

    for ax, (net_name, spec) in zip(axes, CONFIGS.items()):
        stats = all_stats[net_name]
        ax.set_facecolor(SURFACE)

        # Shade each stage's index span so "depth" has a landmark, even
        # though the two configs have different per-stage block counts
        # (and thus different total PReLU counts -- 83 vs 65, itself a real
        # structural difference: separable_dilated's intermediate act
        # between the (k,1) and (1,k) passes adds MORE activation sites per
        # dilated slot than dsc_no_projection's single-act-per-block DSC).
        stage_spans: dict[str, tuple[int, int]] = {}
        for i, s in enumerate(stats):
            lo, hi = stage_spans.get(s["stage"], (i, i))
            stage_spans[s["stage"]] = (min(lo, i), max(hi, i))
        for stage, (lo, hi) in stage_spans.items():
            ax.axvspan(lo - 0.5, hi + 0.5, color=STAGE_BG.get(stage, "#eeeeee"), zorder=0)
            ax.text((lo + hi) / 2, 1.03, stage, transform=ax.get_xaxis_transform(),
                     ha="left", va="bottom", fontsize=7.5, color=SECONDARY_INK, rotation=40)

        idx = list(range(len(stats)))
        means = [s["mean"] for s in stats]
        mins = [s["min"] for s in stats]
        maxs = [s["max"] for s in stats]

        ax.fill_between(idx, mins, maxs, color="#8e44ad", alpha=0.18, zorder=1, label="per-channel min/max range")
        ax.plot(idx, means, color="#8e44ad", linewidth=1.6, marker="o", markersize=3, zorder=2,
                 label="per-channel mean slope")
        ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=":", zorder=1)
        ax.axhline(0.0, color=MUTED, linewidth=0.8, linestyle=":", zorder=1)
        ax.text(len(stats) - 1, 1.0, " identity (a=1)", fontsize=7.5, color=MUTED, va="center")
        ax.text(len(stats) - 1, 0.0, " ReLU-like (a=0)", fontsize=7.5, color=MUTED, va="center")

        ax.set_title(f"{spec['label']} ({net_name}) -- {len(stats)} PReLU sites", color=INK, fontsize=12)
        ax.set_ylabel("Learned PReLU slope (a)", color=SECONDARY_INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK, loc="lower right")

    axes[-1].set_xlabel("Depth (sequential index into forward-pass order)", color=SECONDARY_INK, fontsize=10)
    fig.suptitle("Learned PReLU slope vs. network depth", color=INK, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = args.out_dir / "prelu_activations_vs_depth.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")

    for net_name, spec in CONFIGS.items():
        stats = all_stats[net_name]
        overall_mean = sum(s["mean"] for s in stats) / len(stats)
        print(f"\n{spec['label']}: overall mean slope across all sites = {overall_mean:.4f}")
        by_stage: dict[str, list[float]] = {}
        for s in stats:
            by_stage.setdefault(s["stage"], []).append(s["mean"])
        for stage in STAGE_ORDER:
            if stage in by_stage:
                vals = by_stage[stage]
                print(f"  {stage:14s} n={len(vals):3d}  mean={sum(vals)/len(vals):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

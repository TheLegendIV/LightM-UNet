"""Per-epoch train/val loss and validation pseudo-Dice curves for a
transfer-learned run vs. a from-scratch run, read directly from each
checkpoint's nnU-Net `logging` dict (train_losses/val_losses/mean_fg_dice/
ema_fg_dice -- one entry per epoch, already recorded during training, no
re-inference needed).

Defaults compare stage_3_transfer_original (fine-tuned from the retired
binary Dataset501 checkpoint) against stage_1_naive_baseline's Base config
(same paper channel shape, trained from scratch) -- the closest
apples-to-apples pair available. Note S3-Transfer also differs in
decoder_type (max_unpool vs. upsample_conv) and use_prelu (1 vs. 0) from
Base, per stage_3_transfer_original.job -- so this isolates "transfer vs.
scratch" only loosely, not as a single-variable ablation.

Usage:
    python compression/plot_transfer_vs_scratch.py
    python compression/plot_transfer_vs_scratch.py --ckpt-a path/to/a/checkpoint_final.pth --label-a "..." \
                                                     --ckpt-b path/to/b/checkpoint_final.pth --label-b "..."
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_A = (
    REPO_ROOT / "data" / "nnUNet_results" / "Dataset509_ARCADE_1x1_4c"
    / "nnUNetTrainerENet_1_naive_baseline_Baseline__nnUNetPlans__2d" / "fold_0" / "checkpoint_final.pth"
)
DEFAULT_B = (
    REPO_ROOT / "data" / "nnUNet_results" / "Dataset509_ARCADE_1x1_4c"
    / "nnUNetTrainerENet_3_transfer_original__nnUNetPlans__2d" / "fold_0" / "checkpoint_final.pth"
)

INK, SECONDARY_INK, MUTED, GRID, SPINE, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)
COLOR_A, COLOR_B = "#2a78d6", "#1baf7a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-a", type=Path, default=DEFAULT_A)
    parser.add_argument("--label-a", default="Base (from scratch)")
    parser.add_argument("--ckpt-b", type=Path, default=DEFAULT_B)
    parser.add_argument("--label-b", default="S3-Transfer (from Dataset501 binary ckpt)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "compression" / "results")
    return parser.parse_args()


def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
    ax.tick_params(colors=MUTED, labelsize=9)


def main() -> int:
    args = parse_args()
    if not args.ckpt_a.exists() or not args.ckpt_b.exists():
        print(f"Checkpoint not found: {args.ckpt_a if not args.ckpt_a.exists() else args.ckpt_b}")
        return 1

    runs = {
        args.label_a: (COLOR_A, torch.load(args.ckpt_a, map_location="cpu", weights_only=False)["logging"]),
        args.label_b: (COLOR_B, torch.load(args.ckpt_b, map_location="cpu", weights_only=False)["logging"]),
    }

    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(14, 6), facecolor=SURFACE)

    _style_axis(ax_loss)
    for label, (color, log) in runs.items():
        epochs = list(range(len(log["train_losses"])))
        ax_loss.plot(epochs, log["train_losses"], color=color, linewidth=1.3, label=f"{label} -- train")
        ax_loss.plot(epochs, log["val_losses"], color=color, linewidth=1.3, linestyle="--", label=f"{label} -- val")
    ax_loss.set_xlabel("Epoch", color=SECONDARY_INK, fontsize=10)
    ax_loss.set_ylabel("Loss", color=SECONDARY_INK, fontsize=10)
    ax_loss.set_title("Train/val loss: transfer vs. from-scratch", color=INK, fontsize=12)
    ax_loss.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK, loc="best")

    _style_axis(ax_dice)
    for label, (color, log) in runs.items():
        epochs = list(range(len(log["mean_fg_dice"])))
        ax_dice.plot(epochs, log["mean_fg_dice"], color=color, linewidth=0.8, alpha=0.4)
        ax_dice.plot(epochs, log["ema_fg_dice"], color=color, linewidth=1.6, label=f"{label} -- EMA pseudo dice")
    ax_dice.set_xlabel("Epoch", color=SECONDARY_INK, fontsize=10)
    ax_dice.set_ylabel("Pseudo (validation) foreground Dice", color=SECONDARY_INK, fontsize=10)
    ax_dice.set_title("Validation pseudo-Dice: transfer vs. from-scratch", color=INK, fontsize=12)
    ax_dice.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK, loc="lower right")

    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "transfer_vs_scratch_training_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Wrote {out_path}")

    for label, (_, log) in runs.items():
        tl, vl, dice, ema = log["train_losses"], log["val_losses"], log["mean_fg_dice"], log["ema_fg_dice"]
        n = len(tl)
        best_epoch = max(range(len(ema)), key=lambda i: ema[i])
        print(f"\n{label}:")
        print(f"  epoch 0:      train_loss={tl[0]:.4f} val_loss={vl[0]:.4f} mean_fg_dice={dice[0]:.4f} ema={ema[0]:.4f}")
        print(f"  epoch {n - 1}:    train_loss={tl[-1]:.4f} val_loss={vl[-1]:.4f} mean_fg_dice={dice[-1]:.4f} ema={ema[-1]:.4f}")
        print(f"  best EMA dice: {ema[best_epoch]:.4f} at epoch {best_epoch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

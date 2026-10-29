"""Learned PReLU alpha (y) vs PReLU index (x) for a trained S12 dense / bilinear / U4 PReLU checkpoint.

x = sequential index of every nn.PReLU module in model.named_modules() order (= forward-pass order for ENet), y = that
module's per-channel `weight` (PReLU(x) = max(0,x) + alpha*min(0,x)). A line through the per-site mean. Alpha <= 0 (V-shaped, non-monotone) and alpha >= 1 are the values that cannot map to a single FINN
MultiThreshold, so the count of each is printed and the dashed lines mark 0 and 1. The grey band is the per-site
min-max alpha range.

Writes <out-dir>/<run-name>_alpha.png and <run-name>_alpha.csv (site, module, stage, channel, alpha).

    docker exec lightmunet_dev python3 /workspace/LightM-UNet/report/hw/activation/plot_prelu_alpha.py
    # all-PReLU run (once trained):
    docker exec lightmunet_dev python3 .../plot_prelu_alpha.py --run-name 12_dense_prelu_all --decoder-prelu
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "enet"))
from nnunetv2.nets.ENet import ENet  # noqa: E402

NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results" / "Dataset509_ARCADE_1x1_4c"
BLUE, INK, GRID = "#2a78d6", "#0b0b0b", "#c3c2b7"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="4_8_dense_dilation", help="config name without the nnUNetTrainerENet_ prefix")
    p.add_argument("--decoder-prelu", action="store_true", help="checkpoint was trained with PReLU in the decoder too")
    p.add_argument("--checkpoint-name", default="checkpoint_best.pth")
    p.add_argument("--out-dir", type=Path, default=HERE)
    return p.parse_args()


def load_model(run_name: str, decoder_prelu: bool, checkpoint_name: str) -> nn.Module:
    model = ENet(
        in_channels=1, out_channels=5, channels=(4, 16, 32, 32, 16, 4), bottlenecks_per_stage=(4, 8, 8, 2, 1),
        decoder_type="upsample_conv", use_dilated=True, use_asymmetric=False, use_strided=True, use_dsc=False,
        context_pattern="dense_dilation", use_prelu=True, decoder_prelu=decoder_prelu,
    )
    ckpt_path = NNUNET_RESULTS / f"nnUNetTrainerENet_{run_name}__nnUNetPlans__2d" / "fold_0" / checkpoint_name
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["network_weights"])
    return model.eval()


def main() -> int:
    args = parse_args()
    model = load_model(args.run_name, args.decoder_prelu, args.checkpoint_name)
    sites = [(n, m.weight.detach().flatten().tolist()) for n, m in model.named_modules() if isinstance(m, nn.PReLU)]
    print(f"{args.run_name}: {len(sites)} PReLU sites, {sum(len(w) for _, w in sites)} alphas")

    out_csv = args.out_dir / f"{args.run_name}_alpha.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site", "module", "stage", "channel", "alpha"])
        for i, (name, ws) in enumerate(sites):
            for c, a in enumerate(ws):
                w.writerow([i, name, name.split(".")[0], c, f"{a:.6f}"])

    all_a = [a for _, ws in sites for a in ws]
    n_neg = sum(a <= 0 for a in all_a)
    n_hi = sum(a >= 1 for a in all_a)
    print(f"min {min(all_a):.4f}  max {max(all_a):.4f}  mean {sum(all_a) / len(all_a):.4f}  "
          f"alpha<=0: {n_neg}  alpha>=1: {n_hi}  (of {len(all_a)})")
    sites_out = sorted({i for i, (_, ws) in enumerate(sites) for a in ws if a <= 0 or a >= 1})
    print(f"sites with any alpha outside (0,1): {len(sites_out)}/{len(sites)} -> {sites_out}")

    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold"})
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(range(len(sites)), [sum(ws) / len(ws) for _, ws in sites], color=BLUE, linewidth=1.8, marker="o",
            markersize=4, label="Site Mean", zorder=3)
    idx = list(range(len(sites)))
    ax.fill_between(idx, [min(ws) for _, ws in sites], [max(ws) for _, ws in sites], color="#c9c8c1", alpha=0.6,
                    linewidth=0, label="Site Min-Max", zorder=1)
    for y in (0, 1):
        ax.axhline(y, color="#898781", linestyle="--", linewidth=1.2, zorder=1)
    ax.set_xlim(-1, len(sites))
    ax.set_xlabel("PReLU Index", fontsize=14)
    ax.set_ylabel("Alpha", fontsize=14)
    ax.set_title("Learned PReLU Alpha", fontsize=16)
    ax.tick_params(labelsize=12)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.grid(True, color="#e8e7e0", linewidth=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="best", frameon=True, edgecolor=GRID, prop={"weight": "bold", "size": 12})
    fig.tight_layout()
    out_png = args.out_dir / f"{args.run_name}_alpha.png"
    fig.savefig(out_png, dpi=200)
    print(f"Wrote {out_png}\nWrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

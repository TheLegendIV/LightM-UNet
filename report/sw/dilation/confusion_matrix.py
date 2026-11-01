"""Pixel-level class confusion matrix (test split): does removing dilation hurt vessel detection or artery labelling?

All nets are ReLU + bilinear (upsample_conv) + U4 (channels 4,16,32,16,4), 150-epoch cold-start, checkpoint_best:
  S12            12_dense_relu              dense dilation 2,4,8,16 x2 per context stage, no asymmetric ("Our Context")
  ENet Context   1_naive_baseline_U4        ENet default context block (dilated + asymmetric)
  No Dilated     2_special_ops_no_dilated   ENet Context with the dilated convs made plain 3x3 (the only trained no-dilation net)
No Dilated is compared against ENet Context (the one-knob-off reference, as in report/sw/ablation.py), not against S12.

Writes confusion_counts.csv (raw pixels), per_case_stats.csv (spurious / missed per class), summary.csv
(detection vs labelling split) and confusion_dilation.png (row-normalised %, plus No Dilated - ENet Context).
Run inside the dev container (host python has no torch):
  docker exec -e HOME=/tmp/home_dir lightmunet_dev python3 \
      /workspace/LightM-UNet/report/sw/dilation/confusion_matrix.py [--plot-only]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SW = HERE.parent
sys.path.insert(0, str(SW / "stage_1" / "confusion"))
import confusion_matrix as cm1  # noqa: E402  (reuses confusion_counts / style; imports torch + nnunetv2)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ua, ns = cm1.ua, cm1.ns
CLASSES, K = cm1.CLASSES, cm1.K

# (key, plot title, run name, ENET_CONTEXT_PATTERN, use_dilated, use_asymmetric)
NETS = [
    ("s12", "S12 (Our Context)", "12_dense_relu", "dense_dilation", "1", "0"),
    ("enet_ctx", "ENet Context", "1_naive_baseline_U4", "default", "1", "1"),
    ("no_dilated", "ENet Context, No Dilated", "2_special_ops_no_dilated", "default", "0", "1"),
]
DIFF = ("no_dilated", "enet_ctx")  # (minus, from)


def build_predictor(run_name: str, pattern: str, dilated: str, asym: str):
    os.environ.update({
        "ENET_CHANNELS": "4,16,32,16,4", "ENET_BOTTLENECKS": "4,8,8,2,1", "ENET_DECODER_TYPE": "upsample_conv",
        "ENET_CONTEXT_PATTERN": pattern, "ENET_USE_DILATED": dilated, "ENET_USE_ASYMMETRIC": asym,
        "ENET_USE_STRIDED": "1", "ENET_USE_DSC": "0", "ENET_USE_PRELU": "0",
    })
    folder = ua.NNUNET_RESULTS / ua.DATASET / f"nnUNetTrainerENet_{run_name}__nnUNetPlans__2d"
    predictor = ua.nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                   perform_everything_on_device=True, device=torch.device("cuda"),
                                   verbose=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(folder), use_folds=(0,),
                                                   checkpoint_name="checkpoint_best.pth")
    predictor.network = predictor.network.to(predictor.device).eval()
    return predictor


def summarise(mat: np.ndarray) -> dict:
    """Split foreground error into detection (fg predicted as BG) and labelling (fg predicted as another fg class)."""
    fg = mat[1:]
    n_fg = fg.sum()
    detected = fg[:, 1:].sum()
    correct = np.trace(mat[1:, 1:])
    return {
        "fg_recall": detected / n_fg,                      # fg pixels predicted as any vessel class
        "label_acc_given_detected": correct / detected,    # of those, right class
        "fg_to_bg": fg[:, 0].sum() / n_fg,
        "fg_wrong_class": (detected - correct) / n_fg,
        "bg_to_fg": mat[0, 1:].sum() / mat[0].sum(),
        "binary_dice": 2 * detected / (2 * detected + fg[:, 0].sum() + mat[0, 1:].sum()),
    }


def plot(mats: dict[str, np.ndarray]) -> Path:
    pcts = {k: m / np.maximum(m.sum(axis=1, keepdims=True), 1) * 100 for k, m in mats.items()}
    titles = {k: t for k, t, *_ in NETS}
    panels = [k for k, *_ in NETS if k in pcts]
    has_diff = all(k in pcts for k in DIFF)
    n = len(panels) + int(has_diff)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.2), facecolor=ns.SURFACE)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, panels):
        pct = pcts[k]
        ax.imshow(pct, cmap="Blues", norm=matplotlib.colors.PowerNorm(gamma=0.5, vmin=0, vmax=100))
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{pct[i, j]:.1f}", ha="center", va="center", fontsize=8, fontweight="bold",
                        color="white" if pct[i, j] > 40 else ns.INK)
        cm1._style(ax, titles[k])
    if has_diff:
        diff = pcts[DIFF[0]] - pcts[DIFF[1]]
        ax, lim = axes[-1], np.abs(pcts[DIFF[0]] - pcts[DIFF[1]]).max()
        ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{(round(float(diff[i, j]), 1) or 0.0):+.1f}", ha="center", va="center", fontsize=8,
                        fontweight="bold", color="white" if abs(diff[i, j]) > 0.55 * lim else ns.INK)
        cm1._style(ax, "No Dilated − ENet Context (pts)")
    fig.tight_layout()
    out = HERE / "confusion_dilation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=ns.SURFACE)
    plt.close(fig)
    return out


def write_summary(mats: dict[str, np.ndarray]) -> None:
    fields = ["net", "fg_recall", "label_acc_given_detected", "fg_to_bg", "fg_wrong_class", "bg_to_fg", "binary_dice"]
    with open(HERE / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k, *_ in NETS:
            if k in mats:
                w.writerow({"net": k, **{n: round(float(v), 4) for n, v in summarise(mats[k]).items()}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--plot-only", action="store_true", help="re-plot from confusion_counts.csv, no inference")
    args = ap.parse_args()
    out_csv, case_csv = HERE / "confusion_counts.csv", HERE / "per_case_stats.csv"
    rows = ua.load_rows(out_csv)

    if args.plot_only:
        mats = {}
        for k, *_ in NETS:
            sel = sorted((r for r in rows if r["net"] == k), key=lambda r: CLASSES.index(r["gt"]))
            if sel:
                mats[k] = np.array([[int(r[f"pred_{c}"]) for c in CLASSES] for r in sel], dtype=np.int64)
        write_summary(mats)
        print("Wrote", plot(mats))
        return

    ids, images_dir, labels_dir = ua.case_ids("test", None)
    print(f"test cases={len(ids)}", flush=True)
    case_rows = ua.load_rows(case_csv)
    mats = {}
    for key, _title, run_name, pattern, dilated, asym in NETS:
        predictor = build_predictor(run_name, pattern, dilated, asym)
        pred_dir = Path("/tmp/dilation_confusion") / key
        shutil.rmtree(pred_dir, ignore_errors=True)
        pred_dir.mkdir(parents=True)
        predictor.predict_from_files(
            [[str(images_dir / f"{cid}_0000.png")] for cid in ids],
            [str(pred_dir / cid) for cid in ids],
            save_probabilities=False, overwrite=True,
            num_processes_preprocessing=args.workers, num_processes_segmentation_export=args.workers,
            folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
        mat, cases = cm1.confusion_counts(labels_dir, pred_dir)
        shutil.rmtree(pred_dir, ignore_errors=True)
        mats[key] = mat
        rows = [r for r in rows if r["net"] != key] + [
            {"net": key, "run_name": run_name, "gt": g, **{f"pred_{c}": int(mat[i, j]) for j, c in enumerate(CLASSES)}}
            for i, g in enumerate(CLASSES)]
        case_rows = [r for r in case_rows if r["net"] != key]
        print(f"[{key}] class | GT-empty cases | spurious (GT empty, pred present) | missed | Dice GT-present | mean Dice (as scored)",
              flush=True)
        for c, s in cases.items():
            dp = float(np.mean(s["dice_present"])) if s["dice_present"] else float("nan")
            da = float(np.mean(s["dice_all"]))
            case_rows.append({"net": key, "run_name": run_name, "class": c, "n_cases": s["n"], "gt_empty": s["gt_empty"],
                              "spurious": s["fp_case"], "missed": s["miss_case"], "dice_gt_present": round(dp, 4),
                              "dice_mean": round(da, 4)})
            print(f"  {c:>3} | {s['gt_empty']:3d} | {s['fp_case']:3d} | {s['miss_case']:3d} | {dp:.3f} | {da:.3f}", flush=True)
        print(f"[{key}] row-normalised % (GT rows, pred cols {CLASSES}):", flush=True)
        for i, g in enumerate(CLASSES):
            print(f"  {g:>3}", " ".join(f"{v:6.1f}" for v in mat[i] / max(mat[i].sum(), 1) * 100), flush=True)
        del predictor
        torch.cuda.empty_cache()
    ua.write_rows(out_csv, ["net", "run_name", "gt"] + [f"pred_{c}" for c in CLASSES], rows)
    ua.write_rows(case_csv, ["net", "run_name", "class", "n_cases", "gt_empty", "spurious", "missed",
                             "dice_gt_present", "dice_mean"], case_rows)
    write_summary(mats)
    print("Wrote", plot(mats), flush=True)


if __name__ == "__main__":
    main()

"""Pixel-level class confusion matrix (test split) for the Stage 1 PReLU max-unpool
and bilinear (upsample_conv) nets at the chosen width divisor(s). Rows = ground truth,
columns = prediction, classes = background + LAD/RCA/LCX/LM.

Writes confusion_counts.csv (raw pixel counts) and confusion_U<d>.png (row-normalised %).
Run inside the dev container (host python has no torch):
  docker exec -e HOME=/tmp/home_dir lightmunet_dev python3 \
      /workspace/LightM-UNet/report/sw/stage_1/confusion/confusion_matrix.py [--divisors 4 8]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "maxUnpool" / "skip_ablation"))
import skip_ablation as sa  # noqa: E402  (also imports unpool_ablation -> torch/nnunetv2)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ua = sa.ua
sys.path.insert(0, str(HERE.parents[1]))  # report/sw
import nas_stages as ns  # noqa: E402

CLASSES = ["BG", "LAD", "RCA", "LCX", "LM"]
K = len(CLASSES)
TITLES = {"maxunpool": "Max-Unpool", "bilinear": "Bilinear"}


def confusion_counts(labels_dir: Path, pred_dir: Path) -> tuple[np.ndarray, dict]:
    """Pixel confusion matrix plus per-case, per-class breakdown (the metric in
    collect_results.py is a per-case mean, with empty-vs-empty scoring 1.0)."""
    cm = np.zeros((K, K), dtype=np.int64)
    cases = {c: {"n": 0, "gt_empty": 0, "fp_case": 0, "miss_case": 0, "dice_present": [], "dice_all": []}
             for c in CLASSES[1:]}
    for _, gt, pred in ua.topo.iter_matched_cases(labels_dir, pred_dir, binarize=False):
        cm += np.bincount(gt.astype(np.int64).ravel() * K + pred.astype(np.int64).ravel(),
                          minlength=K * K).reshape(K, K)
        for k, c in enumerate(CLASSES[1:], start=1):
            g, p = gt == k, pred == k
            d = ua.topo.dice_score(g, p)
            s = cases[c]
            s["n"] += 1
            s["dice_all"].append(d)
            if not g.any():
                s["gt_empty"] += 1
                s["fp_case"] += int(p.any())  # GT absent, net predicts it: scores 0 instead of 1
            else:
                s["dice_present"].append(d)
                s["miss_case"] += int(not p.any())
    return cm, cases


def _style(ax, title: str) -> None:
    ax.set_xticks(range(K)); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(K)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted", fontsize=10, fontweight="bold", color=ns.INK)
    ax.set_ylabel("Ground Truth", fontsize=10, fontweight="bold", color=ns.INK)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=ns.INK)
    ax.tick_params(colors=ns.INK, labelsize=8.5)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def plot(div: int, mats: dict[str, np.ndarray]) -> Path:
    """Row-normalised % with one decimal; sqrt colour scale so small confusions are visible.
    Third panel (if both curves present) = max-unpool minus bilinear, in percentage points."""
    pcts = {c: cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100 for c, cm in mats.items()}
    n = len(pcts) + int(len(pcts) == 2)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.2), facecolor=ns.SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (curve, pct) in zip(axes, pcts.items()):
        ax.imshow(pct, cmap="Blues", norm=matplotlib.colors.PowerNorm(gamma=0.5, vmin=0, vmax=100))
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{pct[i, j]:.1f}", ha="center", va="center", fontsize=8, fontweight="bold",
                        color="white" if pct[i, j] > 40 else ns.INK)
        _style(ax, TITLES[curve])
    if len(pcts) == 2:
        diff = pcts["maxunpool"] - pcts["bilinear"]
        ax = axes[2]
        lim = np.abs(diff).max()
        ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{(round(float(diff[i, j]), 1) or 0.0):+.1f}", ha="center", va="center", fontsize=8, fontweight="bold",
                        color="white" if abs(diff[i, j]) > 0.55 * lim else ns.INK)
        _style(ax, "Max-Unpool \u2212 Bilinear (pts)")
    fig.tight_layout()
    out = HERE / f"confusion_U{div}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=ns.SURFACE)
    plt.close(fig)
    return out


def mats_from_csv(rows: list[dict], div: int) -> dict[str, np.ndarray]:
    mats: dict[str, np.ndarray] = {}
    for curve in ("maxunpool", "bilinear"):
        sel = [r for r in rows if r["curve"] == curve and r["divisor"] == str(div)]
        if sel:
            order = {g: i for i, g in enumerate(CLASSES)}
            sel.sort(key=lambda r: order[r["gt"]])
            mats[curve] = np.array([[int(r[f"pred_{c}"]) for c in CLASSES] for r in sel], dtype=np.int64)
    return mats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--divisors", type=int, nargs="+", default=[4])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--plot-only", action="store_true", help="re-plot from confusion_counts.csv, no inference")
    args = ap.parse_args()

    if args.plot_only:
        rows = ua.load_rows(HERE / "confusion_counts.csv")
        for div in args.divisors:
            print("Wrote", plot(div, mats_from_csv(rows, div)))
        return

    ids, images_dir, labels_dir = ua.case_ids("test", None)
    print(f"test cases={len(ids)}", flush=True)
    out_csv = HERE / "confusion_counts.csv"
    rows = ua.load_rows(out_csv)
    case_csv = HERE / "per_case_stats.csv"
    case_rows = ua.load_rows(case_csv)
    for div in args.divisors:
        mats = {}
        for curve, d, run_name, channels, decoder in sa.NETS:
            if d != div:
                continue
            predictor = sa.build_predictor(run_name, channels, decoder)
            pred_dir = Path("/tmp/stage1_confusion") / f"{curve}_{div}"
            shutil.rmtree(pred_dir, ignore_errors=True)
            pred_dir.mkdir(parents=True)
            predictor.predict_from_files(
                [[str(images_dir / f"{cid}_0000.png")] for cid in ids],
                [str(pred_dir / cid) for cid in ids],
                save_probabilities=False, overwrite=True,
                num_processes_preprocessing=args.workers, num_processes_segmentation_export=args.workers,
                folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
            cm, cases = confusion_counts(labels_dir, pred_dir)
            shutil.rmtree(pred_dir, ignore_errors=True)
            case_rows = [c for c in case_rows if not (c["curve"] == curve and c["divisor"] == str(div))]
            print(f"[{curve} U{div}] per-case: class | GT-empty cases | spurious (GT empty, pred present) | "
                  f"missed (GT present, pred empty) | Dice on GT-present | mean Dice (as scored)", flush=True)
            for c, s in cases.items():
                dp = float(np.mean(s["dice_present"])) if s["dice_present"] else float("nan")
                da = float(np.mean(s["dice_all"]))
                case_rows.append({"curve": curve, "divisor": div, "run_name": run_name, "class": c,
                                  "n_cases": s["n"], "gt_empty": s["gt_empty"], "spurious": s["fp_case"],
                                  "missed": s["miss_case"], "dice_gt_present": round(dp, 4),
                                  "dice_mean": round(da, 4)})
                print(f"  {c:>3} | {s['gt_empty']:3d} | {s['fp_case']:3d} | {s['miss_case']:3d} | "
                      f"{dp:.3f} | {da:.3f}", flush=True)
            mats[curve] = cm
            rows = [r for r in rows if not (r["curve"] == curve and r["divisor"] == str(div))]
            for i, g in enumerate(CLASSES):
                rows.append({"curve": curve, "divisor": div, "run_name": run_name, "gt": g,
                             **{f"pred_{c}": int(cm[i, j]) for j, c in enumerate(CLASSES)}})
            print(f"[{curve} U{div}] row-normalised % (GT rows, pred cols {CLASSES}):", flush=True)
            for i, g in enumerate(CLASSES):
                print(f"  {g:>3}", " ".join(f"{v:6.1f}" for v in cm[i] / max(cm[i].sum(), 1) * 100), flush=True)
            del predictor
        ua.write_rows(out_csv, ["curve", "divisor", "run_name", "gt"] + [f"pred_{c}" for c in CLASSES], rows)
        ua.write_rows(case_csv, ["curve", "divisor", "run_name", "class", "n_cases", "gt_empty", "spurious",
                                 "missed", "dice_gt_present", "dice_mean"], case_rows)
        print("Wrote", plot(div, mats), flush=True)


if __name__ == "__main__":
    main()

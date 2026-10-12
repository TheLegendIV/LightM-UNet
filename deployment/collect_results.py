"""Stream a whole nnU-Net test set through the FPGA board and record Dice in deployment/results.csv.

Per image of data/nnUNet_raw/<dataset>/imagesTs:
  1. preprocess (per-image z-score + the model's input quantizer + 128 -> board UINT8 input; scale per build in
     img_intf/utils/input_profiles.json, see img_intf/utils/preprocess.py) into data/nnUNet_preprocessed/<dataset>/img_intf_<model-name>/<case>.raw
  2. send it to the board (v3 batch protocol, img_intf/interfaces/send_image.py) and store the returned
     512x512 class-id map in deployment/img_intf/output/<model-name>/raw/<case>.raw
  3. score the maps against data/nnUNet_raw/<dataset>/labelsTs (same per-class dice / clDice / dice_binary
     definition as compression/collect_results.py: mean over images, 'dice' = mean of the 4 foreground classes)
and upsert one row keyed by (model_name, config) into deployment/results.csv (idempotent: re-running replaces
the row). Already-received outputs are not re-sent unless --resend; --no-send only re-scores stored outputs.

  python deployment/collect_results.py                       # model-name s12_512_nn, whole imagesTs
  python deployment/collect_results.py --limit 10            # smoke test
  python deployment/collect_results.py --no-send             # re-evaluate stored outputs
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

DEPLOYMENT = Path(__file__).resolve().parent
REPO_ROOT = DEPLOYMENT.parent
sys.path.insert(0, str(DEPLOYMENT))
sys.path.insert(0, str(REPO_ROOT / "analysis" / "501_ARCADE"))
import segmentation_topology as topo  # noqa: E402
from img_intf.interfaces.send_image import BatchImageStreamer, IMAGE_H, IMAGE_W  # noqa: E402
from img_intf.utils.preprocess import case_id, png_to_u8  # noqa: E402

RESULTS_CSV = DEPLOYMENT / "results.csv"
NNUNET_RAW = REPO_ROOT / "data" / "nnUNet_raw"
NNUNET_PREPROCESSED = REPO_ROOT / "data" / "nnUNet_preprocessed"
OUTPUT_ROOT = DEPLOYMENT / "img_intf" / "output"
CLASSES = ["LAD", "RCA", "LCX", "LM"]  # label ids 1..4 of Dataset509_ARCADE_1x1_4c

COLUMNS = (
    ["model_name", "config", "dataset", "n_images", "n_scored", "dice", "dice_binary"]
    + [f"dice_{c}" for c in CLASSES] + ["cldice"] + [f"cldice_{c}" for c in CLASSES]
    + ["send_seconds", "timestamp"]
)


def preprocess(images: list[Path], out_dir: Path, profile: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for png in images:
        raw = out_dir / f"{png.stem}.raw"
        if not raw.is_file() or raw.stat().st_size != IMAGE_H * IMAGE_W:
            png_to_u8(png, profile).tofile(raw)
        paths.append(raw)
    return paths


def evaluate(raw_dir: Path, labels_dir: Path, case_ids: list[str]) -> dict:
    per_class = {c: [] for c in CLASSES}
    per_class_cl = {c: [] for c in CLASSES}
    binary = []
    for stem in case_ids:
        pred_path, gt_path = raw_dir / f"{stem}.raw", labels_dir / f"{case_id(stem)}.png"
        if not (pred_path.is_file() and gt_path.is_file()):
            continue
        gt = topo.load_class_id_mask(gt_path, binarize=False)
        pred = topo.resize_mask_to(np.fromfile(pred_path, dtype=np.uint8).reshape(IMAGE_H, IMAGE_W), gt.shape)
        for class_id, name in enumerate(CLASSES, start=1):
            per_class[name].append(topo.dice_score(gt == class_id, pred == class_id))
            per_class_cl[name].append(topo.cldice_score(gt == class_id, pred == class_id))
        binary.append(topo.dice_score(gt > 0, pred > 0))
    n = len(binary)
    result = {"n_scored": n}
    if n == 0:
        return result
    for name in CLASSES:
        result[f"dice_{name}"] = float(np.mean(per_class[name]))
        result[f"cldice_{name}"] = float(np.mean(per_class_cl[name]))
    result["dice"] = float(np.mean([result[f"dice_{c}"] for c in CLASSES]))
    result["cldice"] = float(np.mean([result[f"cldice_{c}"] for c in CLASSES]))
    result["dice_binary"] = float(np.mean(binary))
    return result


def upsert_row(row: dict) -> None:
    rows = []
    if RESULTS_CSV.is_file():
        with open(RESULTS_CSV, newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if (r["model_name"], r["config"]) != (row["model_name"], row["config"])]
    rows.append({c: ("" if row.get(c) is None or row.get(c) != row.get(c) else row[c]) for c in COLUMNS})
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="s12_512_nn", help="build id; names output/<model-name> and the preprocessed dir")
    ap.add_argument("--input-profile", help="input scale profile in img_intf/utils/input_profiles.json (default: --model-name)")
    ap.add_argument("--config", default="board", help="second half of the results.csv key")
    ap.add_argument("--dataset", default="Dataset509_ARCADE_1x1_4c")
    ap.add_argument("--labels-dir", help="default: data/nnUNet_raw/<dataset>/labelsTs")
    ap.add_argument("--ip", default="192.168.0.111")
    ap.add_argument("--port", type=int, default=7)
    ap.add_argument("--limit", type=int, help="only the first N images (smoke test)")
    ap.add_argument("--no-send", action="store_true", help="skip the board; score stored outputs only")
    ap.add_argument("--resend", action="store_true", help="send every image again, even if an output exists")
    a = ap.parse_args()

    images = sorted((NNUNET_RAW / a.dataset / "imagesTs").glob("*.png"))[: a.limit]
    if not images:
        raise FileNotFoundError(f"no PNGs in {NNUNET_RAW / a.dataset / 'imagesTs'}")
    stems = [p.stem for p in images]
    raw_dir = OUTPUT_ROOT / a.model_name / "raw"
    inputs = preprocess(images, NNUNET_PREPROCESSED / a.dataset / f"img_intf_{a.model_name}",
                        a.input_profile or a.model_name)
    print(f"[INFO] {len(inputs)} preprocessed inputs ready")

    send_seconds = None
    if not a.no_send:
        todo = [p for p in inputs if a.resend or not (raw_dir / p.name).is_file()]
        if todo:
            t0 = time.perf_counter()
            BatchImageStreamer(a.ip, a.port, None, str(raw_dir), paths=[str(p) for p in todo]).run()
            send_seconds = round(time.perf_counter() - t0, 1)
        else:
            print("[INFO] all outputs already stored; use --resend to send again")

    labels_dir = Path(a.labels_dir) if a.labels_dir else NNUNET_RAW / a.dataset / "labelsTs"
    if not labels_dir.is_dir():
        print(f"[WARN] {labels_dir} not found: outputs stored, Dice left blank")
        metrics = {"n_scored": 0}
    else:
        metrics = evaluate(raw_dir, labels_dir, stems)
    row = {"model_name": a.model_name, "config": a.config, "dataset": a.dataset, "n_images": len(stems),
           "send_seconds": send_seconds, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **metrics}
    if send_seconds is None and RESULTS_CSV.is_file():  # keep the earlier send time on a re-score
        with open(RESULTS_CSV, newline="") as f:
            for r in csv.DictReader(f):
                if (r["model_name"], r["config"]) == (a.model_name, a.config):
                    row["send_seconds"] = r["send_seconds"] or None
    upsert_row(row)
    print(f"[INFO] wrote {RESULTS_CSV.name} row ({a.model_name}, {a.config}): "
          + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float) and k in ("dice", "dice_binary")))


if __name__ == "__main__":
    main()

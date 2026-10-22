"""Run a whole nnU-Net test set on the FPGA board and record Dice + test-split inference speed in deployment/results.csv.

Needs the test-split benchmark server on the board (FINN_interface/lwip_collect_results, port 7), NOT the live
streaming server. Live-stream metrics (fps over an endless stream, turnaround, stalls) come from
img_intf/interfaces/stream_images.py with lwip_stream.

Per image of data/nnUNet_raw/<dataset>/imagesTs:
  1. preprocess (per-image z-score + the model's input quantizer + 128 -> board UINT8 input; scale per build in
     img_intf/utils/input_profiles.json, see img_intf/utils/preprocess.py) into data/nnUNet_preprocessed/<dataset>/img_intf_<model-name>/<case>.raw
  2. one board session (img_intf/interfaces/collect_split.py): upload all images into board DDR, per-image latency pass,
     --repeats timed passes over the whole split (unlimited source, unlimited sink, no network in the timed part),
     download the 512x512 class-id maps into deployment/img_intf/output/<model-name>/raw/<case>.raw
  3. score the maps against data/nnUNet_raw/<dataset>/labelsTs (same per-class dice / clDice / dice_binary
     definition as compression/collect_results.py: mean over images, 'dice' = mean of the 4 foreground classes)
and upsert one row keyed by (model_name, config) into deployment/results.csv (idempotent: re-running replaces
the row). --no-send only re-scores stored outputs and keeps the earlier speed columns.

Speed columns, all timed in PL clock ticks by the axi_late_mon block (no CPU timer, no network), converted at
PL_CLOCK_MHZ = 187.481262 MHz:
  split_seconds (+_std, split_repeats)  first input beat -> last output beat of the whole split, mean over the passes
  split_fps                             images / split_seconds (includes pipeline fill)
  hw_infer_ms / hw_infer_fps            steady-state time per image: (split time - latency of one image) / (n - 1)
  hw_first_output_ms                    mean over all images of input_first -> output_first (one image alone)
  hw_latency_ms / hw_latency_max_ms     mean / slowest single-image latency (input_first -> output_last)
  milp_bottleneck_node / milp_pred_interval_ms / milp_pred_error_pct   only with --milp-json: the MILP's predicted slowest node, its cycles per frame at PL_CLOCK_MHZ
                                        and how far that is from hw_infer_ms (the model-vs-hardware check)
(send_seconds is a legacy column: host time of the older lockstep runs; it is kept, not written any more.)

Board power (--power; needs the lwip_collect_results build with the POWER command): the 18 INA226 rail monitors of the
ZCU106 are sampled over the PS I2C bus (about 90 sweeps/s) in three windows of the same session and written to
deployment/power_results.csv (one row per condition, key (model_name, config, condition, source)) and, per rail, to
deployment/power_rails.csv:
  Idle                          accelerator idle (before the images are loaded)             --power-idle-s
  Inference (back-to-back)      the split runs back-to-back, after --power-warmup-s of unmeasured warm-up, for --power-load-s
  Idle (after inference)        accelerator idle again (thermal drift check)                 (--no-power-idle-after skips it)
  board_power_w                 SUM OF THE MONITORED RAILS (PS + PL, after the regulators: no conversion losses, no fan /
                                clocks / PHYs). Differences between conditions on the same board are exact; the absolute
                                value is not the 12 V input power.
  increase_above_idle_w         board_power_w - board_power_w(Idle)
  throughput_fps                images / (sum of the PL-tick pass times) of the passes inside the measured window
  energy_per_image_mj           board_power_w / throughput_fps; dynamic_energy_per_image_mj uses the increase above idle
For the true input power of the board read a plug / 12 V meter and pass --ext-power-idle-w / --ext-power-load-w: the same
rows are added with source=external_meter (throughput from the same run).

  python deployment/collect_results.py                       # whole imagesTs, default build s12_512_bilinear_int8_maxfps
  python deployment/collect_results.py --limit 10            # smoke test
  python deployment/collect_results.py --power               # + board power: idle / inference / idle after (about 1 min more)
  python deployment/collect_results.py --dma-mode frame      # one DMA start per image instead of one for the split
  python deployment/collect_results.py --no-send             # re-evaluate stored outputs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DEPLOYMENT = Path(__file__).resolve().parent
REPO_ROOT = DEPLOYMENT.parent
sys.path.insert(0, str(DEPLOYMENT))
sys.path.insert(0, str(REPO_ROOT / "analysis" / "501_ARCADE"))
import segmentation_topology as topo  # noqa: E402
from img_intf.interfaces.collect_split import (  # noqa: E402
    DMA_MODE_BATCH, DMA_MODE_FRAME, HandshakeError, PowerPlan, run_split_benchmark, ticks_to_intervals,
)
from img_intf.utils.power_report import (  # noqa: E402
    PL_CLOCK_MHZ, POWER_COLUMNS, POWER_KEY, POWER_RAIL_COLUMNS, POWER_RAIL_KEY, external_power_rows, power_tables,
    print_power_table, print_rail_table, upsert_csv,
)
from img_intf.utils.preprocess import case_id, png_to_u8  # noqa: E402

IMAGE_H = int(os.getenv("IMAGE_H", "512"))
IMAGE_W = int(os.getenv("IMAGE_W", "512"))
RESULTS_CSV = DEPLOYMENT / "results.csv"
POWER_CSV = DEPLOYMENT / "power_results.csv"
POWER_RAILS_CSV = DEPLOYMENT / "power_rails.csv"
NNUNET_RAW = REPO_ROOT / "data" / "nnUNet_raw"
NNUNET_PREPROCESSED = REPO_ROOT / "data" / "nnUNet_preprocessed"
OUTPUT_ROOT = DEPLOYMENT / "img_intf" / "output"
CLASSES = ["LAD", "RCA", "LCX", "LM"]  # label ids 1..4 of Dataset509_ARCADE_1x1_4c
# PL_CLOCK_MHZ (accelerator clock of the current bitstream, converts PL ticks to time) is defined in img_intf/utils/power_report.py

PERF_COLUMNS = ["split_repeats", "split_seconds", "split_seconds_std", "split_fps", "hw_infer_ms", "hw_infer_fps",
                "hw_first_output_ms", "hw_latency_ms", "hw_latency_max_ms",
                "milp_bottleneck_node", "milp_pred_interval_ms", "milp_pred_error_pct"]
LEGACY_COLUMNS = ["send_seconds"]  # written by the older lockstep runs; carried over, never overwritten
COLUMNS = (
    ["model_name", "config", "dataset", "n_images", "n_scored", "dice", "dice_binary"]
    + [f"dice_{c}" for c in CLASSES] + ["cldice"] + [f"cldice_{c}" for c in CLASSES]
    + LEGACY_COLUMNS + PERF_COLUMNS + ["timestamp"]
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
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")  # drops columns of older versions of this script
        w.writeheader()
        w.writerows(rows)


def run_on_board(inputs: list[Path], raw_dir: Path, ip: str, port: int, dma_mode: int, repeats: int,
                 power_plan: PowerPlan | None = None) -> tuple[dict, dict | None]:
    """One session on lwip_collect_results: upload the split, per-image latency pass, `repeats` timed passes over the
    whole split, download the class maps (stored as <raw_dir>/<input stem>.raw). With `power_plan` the board's rails are
    sampled around it too. Returns (the speed columns, the raw POWER replies or None)."""
    frames = [p.read_bytes() for p in inputs]
    n = len(frames)
    try:
        r = run_split_benchmark(ip, port, frames, dma_mode=dma_mode, repeats=repeats, width=IMAGE_W, height=IMAGE_H,
                                power=power_plan)
    except HandshakeError:
        raise
    except (ConnectionError, OSError) as exc:
        if power_plan is None:
            raise
        raise RuntimeError(f"the session failed during a --power run ({exc}). Either the firmware has no POWER command (an "
                           "lwip_collect_results build without it drops the session on the first POWER command: rebuild with "
                           "src/power_mon.c + src/run.c), or the board hung / reset (check its UART log). Run without --power to "
                           "separate the two") from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    for p, out in zip(inputs, r["outputs"]):
        (raw_dir / p.name).write_bytes(out)
    if n > 1 and len(set(r["outputs"])) == 1 and len(set(frames)) > 1:
        print("[WARN] every returned class map is identical although the inputs differ: with --dma-mode batch the "
              "DMA may not advance its buffer between reps; rerun with --dma-mode frame")

    ms = lambda t: np.asarray(t, dtype=np.float64) / PL_CLOCK_MHZ / 1000  # noqa: E731  PL ticks -> ms
    perf: dict = {}

    lat_first, lat_full = ticks_to_intervals(r["latency_ticks"])
    if len(lat_full) and lat_full.min() > 0:
        perf.update({"hw_first_output_ms": round(float(ms(lat_first.mean())), 3),
                     "hw_latency_ms": round(float(ms(lat_full.mean())), 3),
                     "hw_latency_max_ms": round(float(ms(lat_full.max())), 3)})
        print(f"[BENCH] single-image hardware latency over {len(lat_full)} images @ {PL_CLOCK_MHZ} MHz: "
              f"full image mean {ms(lat_full.mean()):.3f} ms (min {ms(lat_full.min()):.3f}, "
              f"max {ms(lat_full.max()):.3f}); first output mean {ms(lat_first.mean()):.3f} ms")
    else:
        print("[WARN] no valid latency ticks from the board (latency monitor missing or not armed)")

    _, run_full = ticks_to_intervals(r["run_ticks"])  # per pass: first input beat -> last output beat of the split
    if len(run_full) and run_full.min() > 0:
        secs = run_full / (PL_CLOCK_MHZ * 1e6)
        perf.update({"split_repeats": len(secs), "split_seconds": round(float(secs.mean()), 4),
                     "split_seconds_std": round(float(secs.std()), 5), "split_fps": round(float(n / secs.mean()), 1)})
        print(f"[BENCH] {n} images through the accelerator, {len(secs)} pass(es), no network: "
              f"{secs.mean():.4f} s +- {secs.std():.4f} s (min {secs.min():.4f}, max {secs.max():.4f}) "
              f"-> {n / secs.mean():.1f} fps ({secs.mean() / n * 1000:.3f} ms/image incl. pipeline fill)")
        if n > 1 and "hw_latency_ms" in perf:
            # run = latency of the first image + (n-1) steady-state intervals
            steady_ms = float(ms(run_full.mean() - lat_full.mean()) / (n - 1))
            perf.update({"hw_infer_ms": round(steady_ms, 4), "hw_infer_fps": round(1000 / steady_ms, 1)})
            print(f"[BENCH] steady state: {steady_ms:.4f} ms/image -> {1000 / steady_ms:.1f} fps")
    else:
        print("[WARN] no valid split timing from the board (latency monitor missing or not armed)")
    print(f"[INFO] network: upload + latency pass {r['upload_s']:.1f} s, download {r['fetch_s']:.1f} s (not part of the "
          "inference numbers)")
    return perf, r["power"]


def compare_with_milp(milp_json: Path, measured_interval_ms: float) -> dict:
    """The MILP's predicted bottleneck (cycles per frame of its slowest node at PL_CLOCK_MHZ) next to the measured steady-state interval: the model-vs-hardware check of the iteration loop."""
    with open(milp_json) as f:
        diag = json.load(f)["_diagnostics"]
    cycles, node = diag["bottleneck_cycles"], diag.get("bottleneck_node")
    pred_ms = cycles / (PL_CLOCK_MHZ * 1e3)
    err = (pred_ms - measured_interval_ms) / measured_interval_ms * 100
    print(f"[MILP] predicted bottleneck {node}: {cycles} cycles = {pred_ms:.4f} ms ({1000 / pred_ms:.1f} fps) vs measured steady state "
          f"{measured_interval_ms:.4f} ms ({1000 / measured_interval_ms:.1f} fps): model {err:+.2f} %")
    return {"milp_bottleneck_node": node, "milp_pred_interval_ms": round(pred_ms, 4), "milp_pred_error_pct": round(err, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="s12_512_bilinear_int8_maxfps", help="build id; names output/<model-name> and the preprocessed dir")
    ap.add_argument("--input-profile", help="input scale profile in img_intf/utils/input_profiles.json (default: --model-name)")
    ap.add_argument("--config", default="board", help="second half of the results.csv key")
    ap.add_argument("--dataset", default="Dataset509_ARCADE_1x1_4c")
    ap.add_argument("--labels-dir", help="default: data/nnUNet_raw/<dataset>/labelsTs")
    ap.add_argument("--ip", default="192.168.0.111")
    ap.add_argument("--port", type=int, default=7)
    ap.add_argument("--limit", type=int, help="only the first N images (smoke test)")
    ap.add_argument("--no-send", action="store_true", help="skip the board; score stored outputs only")
    ap.add_argument("--repeats", type=int, default=10, help="timed passes over the whole split (more = more accurate)")
    ap.add_argument("--dma-mode", choices=["batch", "frame"], default="batch",
                    help="batch: one idma/odma start for the whole split (numReps = n); frame: one start per image")
    ap.add_argument("--milp-json", type=Path,
                    help="MILP layer_bits_folding_*.json of this build: records its predicted bottleneck (cycles at the PL clock) and the error against the measured steady-state interval")
    ap.add_argument("--power", action="store_true",
                    help="also measure the board's rail power (INA226 monitors): idle / inference / idle after -> power_results.csv, power_rails.csv")
    ap.add_argument("--power-idle-s", type=int, default=10, help="length of each idle window (default 10)")
    ap.add_argument("--power-warmup-s", type=int, default=5, help="inference warm-up before the measured window, run but not measured (default 5)")
    ap.add_argument("--power-load-s", type=int, default=20, help="measured inference window (default 20)")
    ap.add_argument("--no-power-idle-after", action="store_true", help="skip the second idle window after the inference window")
    ap.add_argument("--ext-power-idle-w", type=float, help="board input power read from an external meter with the accelerator idle (W); needs --ext-power-load-w")
    ap.add_argument("--ext-power-load-w", type=float, help="board input power from the external meter while the split runs back-to-back (W)")
    a = ap.parse_args()
    if (a.ext_power_idle_w is None) != (a.ext_power_load_w is None):
        ap.error("--ext-power-idle-w and --ext-power-load-w go together")
    if a.power and a.no_send:
        ap.error("--power needs the board (drop --no-send)")
    power_plan = (PowerPlan(idle_s=a.power_idle_s, warmup_s=a.power_warmup_s, load_s=a.power_load_s,
                            idle_after=not a.no_power_idle_after) if a.power else None)

    images = sorted((NNUNET_RAW / a.dataset / "imagesTs").glob("*.png"))[: a.limit]
    if not images:
        raise FileNotFoundError(f"no PNGs in {NNUNET_RAW / a.dataset / 'imagesTs'}")
    stems = [p.stem for p in images]
    raw_dir = OUTPUT_ROOT / a.model_name / "raw"
    inputs = preprocess(images, NNUNET_PREPROCESSED / a.dataset / f"img_intf_{a.model_name}",
                        a.input_profile or a.model_name)
    print(f"[INFO] {len(inputs)} preprocessed inputs ready")

    perf: dict = {}
    power_raw = None
    if not a.no_send:
        perf, power_raw = run_on_board(inputs, raw_dir, a.ip, a.port,
                                       DMA_MODE_BATCH if a.dma_mode == "batch" else DMA_MODE_FRAME, a.repeats, power_plan)

    labels_dir = Path(a.labels_dir) if a.labels_dir else NNUNET_RAW / a.dataset / "labelsTs"
    if not labels_dir.is_dir():
        print(f"[WARN] {labels_dir} not found: outputs stored, Dice left blank")
        metrics = {"n_scored": 0}
    else:
        metrics = evaluate(raw_dir, labels_dir, stems)
    row = {"model_name": a.model_name, "config": a.config, "dataset": a.dataset, "n_images": len(stems),
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **metrics}
    if RESULTS_CSV.is_file():  # columns this run did not measure keep their earlier values
        with open(RESULTS_CSV, newline="") as f:
            for r in csv.DictReader(f):
                if (r["model_name"], r["config"]) == (a.model_name, a.config):
                    row.update({c: r.get(c) or None for c in PERF_COLUMNS})
    row.update(perf)
    if a.milp_json is not None and row.get("hw_infer_ms") not in (None, ""):
        row.update(compare_with_milp(a.milp_json, float(row["hw_infer_ms"])))
    upsert_row(row)
    print(f"[INFO] wrote {RESULTS_CSV.name} row ({a.model_name}, {a.config}): "
          + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float) and k in ("dice", "dice_binary")))

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    power_rows: list = []
    failure = None
    if power_raw is not None:
        try:
            power_rows, rail_rows = power_tables(a.model_name, a.config, power_raw, stamp)
            upsert_csv(POWER_RAILS_CSV, POWER_RAIL_COLUMNS, POWER_RAIL_KEY, rail_rows)
            print(f"[INFO] wrote {len(rail_rows)} rail rows to {POWER_RAILS_CSV.name}")
            print_rail_table(rail_rows)
        except RuntimeError as exc:
            failure = exc
    if a.ext_power_idle_w is not None:
        fps = next((r["throughput_fps"] for r in power_rows if r.get("throughput_fps")), None) or float(row.get("split_fps") or 0)
        if fps > 0:
            power_rows += external_power_rows(a.model_name, a.config, a.ext_power_idle_w, a.ext_power_load_w, fps, stamp)
        else:
            print("[WARN] no throughput known for this build (run without --no-send first): external meter rows not written")
    if power_rows:
        upsert_csv(POWER_CSV, POWER_COLUMNS, POWER_KEY, power_rows)
        print(f"[INFO] wrote {len(power_rows)} row(s) to {POWER_CSV.name}")
        print_power_table(power_rows)
    if failure is not None:
        print(f"[ERROR] {failure}")
        sys.exit(1)


if __name__ == "__main__":
    main()

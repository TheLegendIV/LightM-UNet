"""Inference-time max-unpool skip ablations on the trained PReLU+max_unpool
ENet width sweep (divisors 1/2/4/8/16).

Why: bilinear (upsample_conv) beats max_unpool at low widths and we want to
know why. This script changes ONLY the skip branch of the two
UpsamplingBottleneck blocks (up4, up5) of the already-trained nets -- no
retraining -- and measures validation Dice plus skip diagnostics.

Experiments (the `exp` column):
  control          real indices (the trained net, unchanged)            [exp 3]
  random_shared    one fixed random slot per 2x2 window, the SAME map for
                   every channel (seeded; `--random-mode per_channel` draws
                   an independent map per channel instead)              [exp 1]
  nearest_quarter  no indices at all: v/4 written into all 4 sub-pixels
                   (= E[unpool over random indices], same mass per window,
                   position info removed, full coverage)               [exp 2]
  nearest_half     no indices at all: v/2 written into all 4 sub-pixels.
                   Same L2 norm per window as the real skip (4*(v/2)^2 = v^2),
                   so it removes the norm confound of nearest_quarter (whose
                   norm is halved); the window SUM is doubled instead.

Per (width, exp) it records:
  * Dice on the validation split (nnU-Net fold-0 `val` list), same metric as
    compression/collect_results.py (mean over the 4 classes of per-case
    Dice) -- that script scores the TEST split, this one defaults to val.
  * per decoder stage (up4, up5): log10(||main|| / ||out||), "pixel fill"
    (fraction of high-res pixels with >= 1 non-zero skip channel, plus the
    count of non-zero skip channels per pixel), and mean/std/min/max of the
    block output (post out_act).

Run inside the dev container (host python has no torch):
  docker exec -e HOME=/tmp/home_dir lightmunet_dev python3 \
      /workspace/LightM-UNet/analysis/report/model_search/maxUnpool/unpool_ablation.py
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DATASET = "Dataset509_ARCADE_1x1_4c"
OUT_DIR_DEFAULT = Path(__file__).resolve().parent

# nnunetv2 reads these at import time.
os.environ.setdefault("nnUNet_raw", str(REPO / "data" / "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(REPO / "data" / "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", str(REPO / "data" / "nnUNet_results"))

sys.path.insert(0, str(REPO / "enet"))
sys.path.insert(0, str(REPO / "analysis" / "501_ARCADE"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import segmentation_topology as topo  # noqa: E402
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402

NNUNET_RAW = Path(os.environ["nnUNet_raw"])
NNUNET_RESULTS = Path(os.environ["nnUNet_results"])

# (divisor, run name, ENET_CHANNELS) -- same values as the Slurm jobs
# stage_enet_original_prelu_maxunpool_4c.job and
# stage_1_naive_baseline_prelu_maxunpool_array.job.
WIDTHS = [
    (1, "enet_original_prelu_maxunpool_4c", "16,64,128,128,64,16"),
    (2, "1_naive_baseline_prelu_maxunpool_U2", "8,32,64,32,8"),
    (4, "1_naive_baseline_prelu_maxunpool_U4", "4,16,32,16,4"),
    (8, "1_naive_baseline_prelu_maxunpool_U8", "4,8,16,8,4"),
    (16, "1_naive_baseline_prelu_maxunpool_U16", "4,4,8,4,4"),
]
STAGES = ("up4", "up5")
CLASS_NAMES = ("LAD", "RCA", "LCX", "LM")

DICE_FIELDS = ["split", "divisor", "run_name", "exp", "seed", "n_cases", "dice", "dice_binary",
               "dice_LAD", "dice_RCA", "dice_LCX", "dice_LM"]
STAGE_FIELDS = ["split", "divisor", "run_name", "exp", "seed", "stage", "skip_channels", "n_patches",
                "log10_main_over_out_mean", "log10_main_over_out_std",
                "main_rms_mean", "out_rms_mean",
                "pixel_fill_mean", "pixel_fill_std", "pixel_fill_theory_indep",
                "nz_channels_mean", "nz_channels_std", "channel_density_mean",
                "act_mean", "act_std", "act_min", "act_max"]


# --------------------------------------------------------------------------- #
# Skip-branch patch
# --------------------------------------------------------------------------- #
class StageStats:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.log_ratio, self.main_rms, self.out_rms = [], [], []
        self.fill, self.nz_mean, self.nz_std, self.density = [], [], [], []
        self.act_n, self.act_sum, self.act_sumsq = 0, 0.0, 0.0
        self.act_min, self.act_max = math.inf, -math.inf

    @torch.no_grad()
    def record(self, main: torch.Tensor, out: torch.Tensor, act: torch.Tensor) -> None:
        # The predictor runs under CUDA autocast: cast before taking norms/sums.
        m, o, a = main.float(), out.float(), act.float()
        nz = m != 0
        count = nz.sum(dim=1).float()  # (N, H, W): non-zero skip channels per pixel
        self.fill.append((count > 0).float().mean().item())
        self.nz_mean.append(count.mean().item())
        self.nz_std.append(count.std().item())
        self.density.append(nz.float().mean().item())
        main_norm, out_norm = m.norm().item(), o.norm().item()
        self.log_ratio.append(math.log10(max(main_norm, 1e-30) / max(out_norm, 1e-30)))
        self.main_rms.append(m.pow(2).mean().sqrt().item())
        self.out_rms.append(o.pow(2).mean().sqrt().item())
        ad = a.double()
        self.act_n += ad.numel()
        self.act_sum += ad.sum().item()
        self.act_sumsq += ad.pow(2).sum().item()
        self.act_min = min(self.act_min, ad.min().item())
        self.act_max = max(self.act_max, ad.max().item())

    def summary(self, skip_channels: int) -> dict:
        def mean(v):
            return float(np.mean(v)) if v else float("nan")

        def std(v):
            return float(np.std(v)) if v else float("nan")

        act_mean = self.act_sum / max(self.act_n, 1)
        act_var = max(self.act_sumsq / max(self.act_n, 1) - act_mean ** 2, 0.0)
        return {
            "skip_channels": skip_channels, "n_patches": len(self.fill),
            "log10_main_over_out_mean": mean(self.log_ratio), "log10_main_over_out_std": std(self.log_ratio),
            "main_rms_mean": mean(self.main_rms), "out_rms_mean": mean(self.out_rms),
            "pixel_fill_mean": mean(self.fill), "pixel_fill_std": std(self.fill),
            "pixel_fill_theory_indep": 1.0 - 0.75 ** skip_channels,
            "nz_channels_mean": mean(self.nz_mean), "nz_channels_std": mean(self.nz_std),
            "channel_density_mean": mean(self.density),
            "act_mean": act_mean, "act_std": math.sqrt(act_var),
            "act_min": self.act_min, "act_max": self.act_max,
        }


class SkipPatch:
    """Replaces UpsamplingBottleneck.forward on the two decoder blocks.
    `mode` is switched between runs; `control` reproduces the original
    max_unpool path exactly (checked against the unpatched forward)."""

    def __init__(self, net, random_mode: str) -> None:
        self.mode = "control"
        self.seed = 0
        self.random_mode = random_mode
        self.stats = {s: StageStats() for s in STAGES}
        self._index_cache: dict = {}
        self.blocks = {"up4": net.up4, "up5": net.up5}
        for stage_id, (name, block) in enumerate(self.blocks.items()):
            assert block.learned_skip_upsample is False and block.skip_resize_conv is None, name
            block.forward = types.MethodType(self._make_forward(name, stage_id), block)

    def _random_indices(self, stage_id: int, main: torch.Tensor, output_size) -> torch.Tensor:
        n, c, h, w = main.shape
        out_h, out_w = int(output_size[2]), int(output_size[3])
        assert out_h == 2 * h and out_w == 2 * w, f"random indices need even 2x output, got {(h, w)} -> {(out_h, out_w)}"
        per_channel = self.random_mode == "per_channel"
        key = (stage_id, self.seed, per_channel, c if per_channel else 1, h, w, str(main.device))
        if key not in self._index_cache:
            gen = torch.Generator().manual_seed(self.seed * 1000 + stage_id)
            slot = torch.randint(0, 4, (c if per_channel else 1, h, w), generator=gen)
            di, dj = slot // 2, slot % 2
            ii = torch.arange(h)[None, :, None]
            jj = torch.arange(w)[None, None, :]
            flat = (2 * ii + di) * out_w + (2 * jj + dj)  # flat position in each channel's out_h*out_w plane
            self._index_cache[key] = flat.to(main.device)
        flat = self._index_cache[key]
        return flat[None].expand(n, c, h, w).contiguous()

    def _make_forward(self, name: str, stage_id: int):
        patch = self

        def forward(block, x, output_size, indices=None):
            assert indices is not None, "max_unpool decoder expects encoder indices"
            main = block.main_proj(x)
            if patch.mode == "control":
                main = block.unpool(main, indices, output_size=output_size)
            elif patch.mode == "random":
                main = block.unpool(main, patch._random_indices(stage_id, main, output_size), output_size=output_size)
            elif patch.mode == "nearest_quarter":
                main = F.interpolate(main, size=output_size[2:], mode="nearest") * 0.25
            elif patch.mode == "nearest_half":
                main = F.interpolate(main, size=output_size[2:], mode="nearest") * 0.5
            else:
                raise ValueError(patch.mode)
            out = block.reduce(x)
            out = block.up(out)
            out = block.dropout(block.expand(out))
            assert out.shape[2:] == main.shape[2:], (out.shape, main.shape)
            act = block.out_act(main + out)
            patch.stats[name].record(main, out, act)
            return act

        return forward


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_dice(labels_dir: Path, pred_dir: Path) -> dict:
    per_class = {name: [] for name in CLASS_NAMES}
    binary = []
    for _, gt, pred in topo.iter_matched_cases(labels_dir, pred_dir, binarize=False):
        for class_id, name in enumerate(CLASS_NAMES, start=1):
            per_class[name].append(topo.dice_score(gt == class_id, pred == class_id))
        binary.append(topo.dice_score(gt > 0, pred > 0))
    if not binary:
        raise FileNotFoundError(f"no matched gt/pred pairs for {pred_dir}")
    row = {f"dice_{n}": float(np.mean(v)) for n, v in per_class.items()}
    row["dice"] = float(np.mean([row[f"dice_{n}"] for n in CLASS_NAMES]))
    row["dice_binary"] = float(np.mean(binary))
    row["n_cases"] = len(binary)
    return row


def case_ids(split: str, limit: int | None) -> tuple[list[str], Path, Path]:
    if split == "val":
        import json
        ids = json.load(open(NNUNET_RAW / DATASET / "splits_final.json"))[0]["val"]
        images, labels = NNUNET_RAW / DATASET / "imagesTr", NNUNET_RAW / DATASET / "labelsTr"
    else:
        labels = NNUNET_RAW / DATASET / "labelsTs"
        images = NNUNET_RAW / DATASET / "imagesTs"
        ids = sorted(p.name[: -len("_0000.png")] for p in images.glob("*_0000.png"))
    ids = sorted(ids)
    return (ids[:limit] if limit else ids), images, labels


# --------------------------------------------------------------------------- #
# CSV helpers (resume-safe)
# --------------------------------------------------------------------------- #
def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key_of(row: dict) -> tuple:
    return (str(row["split"]), str(row["divisor"]), str(row["exp"]), str(row["seed"]))


# --------------------------------------------------------------------------- #
def build_predictor(div: int, run_name: str, channels: str) -> nnUNetPredictor:
    os.environ.update({
        "ENET_CHANNELS": channels, "ENET_BOTTLENECKS": "4,8,8,2,1", "ENET_DECODER_TYPE": "max_unpool",
        "ENET_CONTEXT_PATTERN": "default", "ENET_USE_DILATED": "1", "ENET_USE_ASYMMETRIC": "1",
        "ENET_USE_STRIDED": "1", "ENET_USE_DSC": "0", "ENET_USE_PRELU": "1",
    })
    model_folder = NNUNET_RESULTS / DATASET / f"nnUNetTrainerENet_{run_name}__nnUNetPlans__2d"
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                perform_everything_on_device=True, device=torch.device("cuda"),
                                verbose=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model_folder), use_folds=(0,),
                                                   checkpoint_name="checkpoint_best.pth")
    # The predictor only moves the network to the device lazily, inside its own
    # sliding-window call; the control-equivalence probe needs it there first.
    predictor.network = predictor.network.to(predictor.device).eval()
    return predictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--divisors", type=int, nargs="+", default=[w[0] for w in WIDTHS])
    ap.add_argument("--exps", nargs="+", default=["control", "random", "nearest_quarter", "nearest_half"],
                    choices=["control", "random", "nearest_quarter", "nearest_half"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="seeds for the random-index exp")
    ap.add_argument("--random-mode", choices=["shared", "per_channel"], default="shared")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cases (smoke test)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--force", action="store_true", help="recompute rows that already exist")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dice_csv, stage_csv = args.out_dir / "dice_by_width.csv", args.out_dir / "stage_stats.csv"
    dice_rows, stage_rows = load_rows(dice_csv), load_rows(stage_csv)
    done = {key_of(r) for r in dice_rows}

    ids, images_dir, labels_dir = case_ids(args.split, args.limit)
    print(f"split={args.split} cases={len(ids)} random_mode={args.random_mode}", flush=True)

    for div, run_name, channels in WIDTHS:
        if div not in args.divisors:
            continue
        exp_keys = []
        for exp in args.exps:
            for seed in (args.seeds if exp == "random" else [0]):
                label = f"random_{args.random_mode}" if exp == "random" else exp
                if args.force or (args.split, str(div), label, str(seed)) not in done:
                    exp_keys.append((exp, label, seed))
        if not exp_keys:
            print(f"[U{div}] nothing to do", flush=True)
            continue

        predictor = build_predictor(div, run_name, channels)
        net = predictor.network

        # Unpatched reference output, for the control-equivalence check below.
        torch.manual_seed(0)
        probe = torch.randn(1, 1, 512, 512, device="cuda")
        with torch.no_grad():
            reference = net(probe).float()
        patch = SkipPatch(net, args.random_mode)
        with torch.no_grad():
            patched = net(probe).float()
        max_diff = (reference - patched).abs().max().item()
        assert max_diff < 1e-5, f"patched control forward differs from original: {max_diff}"
        print(f"[U{div}] control forward matches the original (max |diff| {max_diff:.2e})", flush=True)
        skip_channels = {name: blk.main_proj[0].out_channels for name, blk in patch.blocks.items()}

        for exp, label, seed in exp_keys:
            patch.mode, patch.seed = exp, seed
            for s in patch.stats.values():
                s.reset()
            pred_dir = Path("/tmp/unpool_ablation") / f"{args.split}_{div}_{label}_{seed}"
            shutil.rmtree(pred_dir, ignore_errors=True)
            pred_dir.mkdir(parents=True)
            t0 = time.time()
            predictor.predict_from_files(
                [[str(images_dir / f"{cid}_0000.png")] for cid in ids],
                [str(pred_dir / cid) for cid in ids],
                save_probabilities=False, overwrite=True,
                num_processes_preprocessing=args.workers, num_processes_segmentation_export=args.workers,
                folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
            row = {"split": args.split, "divisor": div, "run_name": run_name, "exp": label, "seed": seed,
                   **score_dice(labels_dir, pred_dir)}
            shutil.rmtree(pred_dir, ignore_errors=True)

            n_patches = {n: patch.stats[n].summary(skip_channels[n])["n_patches"] for n in STAGES}
            if any(n != len(ids) for n in n_patches.values()):
                print(f"  WARNING: expected {len(ids)} forward passes per stage, saw {n_patches}", flush=True)

            dice_rows = [r for r in dice_rows if key_of(r) != key_of(row)] + [row]
            stage_rows = [r for r in stage_rows if key_of(r) != key_of(row)]
            for name in STAGES:
                stage_rows.append({"split": args.split, "divisor": div, "run_name": run_name, "exp": label,
                                   "seed": seed, "stage": name, **patch.stats[name].summary(skip_channels[name])})
            write_rows(dice_csv, DICE_FIELDS, dice_rows)
            write_rows(stage_csv, STAGE_FIELDS, stage_rows)
            print(f"[U{div}] {label:<18} seed={seed} dice={row['dice']:.4f} "
                  f"binary={row['dice_binary']:.4f} ({time.time() - t0:.0f}s)", flush=True)

        del predictor, net, patch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

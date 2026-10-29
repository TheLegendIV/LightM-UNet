"""Stage 1 inference-time skip ablation: max-unpool vs bilinear (PReLU) width sweep.

Changes only the skip (main) branch of the two decoder UpsamplingBottlenecks
(up4, up5) of the already-trained nets; no retraining. Scores the TEST split with
the same per-class Dice as nas/collect_results.py (via unpool_ablation.score_dice).

Arms (`exp` column):
  control          unchanged net (both curves)
  noskip           main_proj(x) replaced by zeros -> out_act(0 + residual) (both curves)
  random_pc_s<N>   max-unpool nets only: independent random 2x2 slot per channel and
                   window, seed N. Same per-channel sparsity/coverage as the real
                   indices, only WHERE the value lands is random.
  nearest_quarter  max-unpool nets only: v/4 into all 4 sub-pixels (same per-window sum)
  nearest_half     max-unpool nets only: v/2 into all 4 sub-pixels (same per-window L2 norm)

Run inside the dev container (host python has no torch):
  docker exec -e HOME=/tmp/home_dir lightmunet_dev python3 \
      /workspace/LightM-UNet/report/sw/maxUnpool/skip_ablation/skip_ablation.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # report/sw/maxUnpool (unpool_ablation.py)
import unpool_ablation as ua  # noqa: E402  (sets nnUNet_* env, imports torch/nnunetv2)

import torch  # noqa: E402

# (curve, divisor, run name, ENET_CHANNELS, decoder type)
NETS = [
    ("maxunpool", 1, "enet_original_prelu_maxunpool_4c", "16,64,128,128,64,16", "max_unpool"),
    ("maxunpool", 2, "1_naive_baseline_prelu_maxunpool_U2", "8,32,64,32,8", "max_unpool"),
    ("maxunpool", 4, "1_naive_baseline_prelu_maxunpool_U4", "4,16,32,16,4", "max_unpool"),
    ("maxunpool", 8, "1_naive_baseline_prelu_maxunpool_U8", "4,8,16,8,4", "max_unpool"),
    ("maxunpool", 16, "1_naive_baseline_prelu_maxunpool_U16", "4,4,8,4,4", "max_unpool"),
    ("bilinear", 1, "1_naive_baseline_prelu_upsample_conv_Baseline", "16,64,128,64,16", "upsample_conv"),
    ("bilinear", 2, "1_naive_baseline_prelu_upsample_conv_U2", "8,32,64,32,8", "upsample_conv"),
    # U4 point is Stage 2's "+PReLU" probe (identical config), see nas_stages.py
    ("bilinear", 4, "2_special_ops_prelu", "4,16,32,16,4", "upsample_conv"),
    ("bilinear", 8, "1_naive_baseline_prelu_upsample_conv_U8", "4,8,16,8,4", "upsample_conv"),
    ("bilinear", 16, "1_naive_baseline_prelu_upsample_conv_U16", "4,4,8,4,4", "upsample_conv"),
]
FIELDS = ["split", "curve", "divisor", "run_name", "exp", "n_cases", "dice", "dice_binary",
          "dice_LAD", "dice_RCA", "dice_LCX", "dice_LM"]


def build_predictor(run_name: str, channels: str, decoder: str):
    os.environ.update({
        "ENET_CHANNELS": channels, "ENET_BOTTLENECKS": "4,8,8,2,1", "ENET_DECODER_TYPE": decoder,
        "ENET_CONTEXT_PATTERN": "default", "ENET_USE_DILATED": "1", "ENET_USE_ASYMMETRIC": "1",
        "ENET_USE_STRIDED": "1", "ENET_USE_DSC": "0", "ENET_USE_PRELU": "1",
    })
    folder = ua.NNUNET_RESULTS / ua.DATASET / f"nnUNetTrainerENet_{run_name}__nnUNetPlans__2d"
    predictor = ua.nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                   perform_everything_on_device=True, device=torch.device("cuda"),
                                   verbose=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(folder), use_folds=(0,),
                                                   checkpoint_name="checkpoint_best.pth")
    predictor.network = predictor.network.to(predictor.device).eval()
    return predictor


class SkipOff:
    """Forward hook on up4/up5 main_proj: zeros the skip branch while `active`."""

    def __init__(self, net) -> None:
        self.active = False
        for blk in (net.up4, net.up5):
            blk.main_proj.register_forward_hook(
                lambda _m, _i, out: torch.zeros_like(out) if self.active else out)


RANDOM_SEEDS = (0, 1, 2)
# exp -> (SkipPatch mode, seed, zero the skip). Only the first two apply to bilinear nets.
ARMS = {
    "control": ("control", 0, False),
    "noskip": ("control", 0, True),
    **{f"random_pc_s{s}": ("random", s, False) for s in RANDOM_SEEDS},
    "nearest_quarter": ("nearest_quarter", 0, False),
    "nearest_half": ("nearest_half", 0, False),
}
BILINEAR_ARMS = ("control", "noskip")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--csv", type=Path, default=HERE / "skip_ablation_results.csv",
                    help="results file (use a scratch path for smoke tests)")
    ap.add_argument("--divisors", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--curves", nargs="+", default=["maxunpool", "bilinear"])
    args = ap.parse_args()

    out_csv = args.csv
    rows = ua.load_rows(out_csv)
    done = {(r["split"], r["curve"], r["divisor"], r["exp"]) for r in rows}
    ids, images_dir, labels_dir = ua.case_ids(args.split, args.limit)
    print(f"split={args.split} cases={len(ids)}", flush=True)

    for curve, div, run_name, channels, decoder in NETS:
        if div not in args.divisors or curve not in args.curves:
            continue
        arms = list(ARMS) if decoder == "max_unpool" else list(BILINEAR_ARMS)
        todo = [e for e in arms if args.force or (args.split, curve, str(div), e) not in done]
        if not todo:
            print(f"[{curve} U{div}] nothing to do", flush=True)
            continue
        predictor = build_predictor(run_name, channels, decoder)
        net = predictor.network
        hook = SkipOff(net)
        patch = None
        if decoder == "max_unpool":
            # Control-equivalence check: the patched forward must reproduce the original.
            torch.manual_seed(0)
            probe = torch.randn(1, 1, 512, 512, device="cuda")
            with torch.no_grad():
                reference = net(probe).float()
            patch = ua.SkipPatch(net, "per_channel")
            with torch.no_grad():
                patched = net(probe).float()
            max_diff = (reference - patched).abs().max().item()
            assert max_diff < 1e-5, f"patched control forward differs from original: {max_diff}"
            print(f"[{curve} U{div}] control forward matches the original (max |diff| {max_diff:.2e})", flush=True)
        for exp in todo:
            mode, seed, zero_skip = ARMS[exp]
            hook.active = zero_skip
            if patch is not None:
                patch.mode, patch.seed = mode, seed
                for s in patch.stats.values():
                    s.reset()
            pred_dir = Path("/tmp/stage1_skip_ablation") / f"{args.split}_{curve}_{div}_{exp}"
            shutil.rmtree(pred_dir, ignore_errors=True)
            pred_dir.mkdir(parents=True)
            t0 = time.time()
            predictor.predict_from_files(
                [[str(images_dir / f"{cid}_0000.png")] for cid in ids],
                [str(pred_dir / cid) for cid in ids],
                save_probabilities=False, overwrite=True,
                num_processes_preprocessing=args.workers, num_processes_segmentation_export=args.workers,
                folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
            row = {"split": args.split, "curve": curve, "divisor": div, "run_name": run_name, "exp": exp,
                   **ua.score_dice(labels_dir, pred_dir)}
            shutil.rmtree(pred_dir, ignore_errors=True)
            rows = [r for r in rows if (r["split"], r["curve"], r["divisor"], r["exp"])
                    != (args.split, curve, str(div), exp)] + [row]
            ua.write_rows(out_csv, FIELDS, rows)
            print(f"[{curve} U{div}] {exp:<16} dice={row['dice']:.4f} binary={row['dice_binary']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        del predictor, net, hook, patch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

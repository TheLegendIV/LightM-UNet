"""Upscale pareto sweep -- push the Original baseline's capacity UP (opposite
direction from the compression/ pruning study) to see if a bigger ENet beats
Original on real Dice before the 37-run compression pipeline starts pruning
from Original. Cherry-picked methodology from wip_pc's dirty pareto/
graduation study: short (15-epoch) sweep over an experiment table -> plot
the pareto front (params vs. final Dice) -> manually pick winners ->
upscale/graduate.py re-runs winners at full epoch count.

Every experiment is expressed through this repo's own parametric
enet/nnunetv2/nets/ENet.py knobs (channels, bottlenecks_per_stage) -- no new
architecture classes. Three tracks, combinatorial over "which stage(s) get
bumped" (each stage independently, then 2 together, then 3 together, ...):

  Track A "MU" (channel WIDTH, decoder=max_unpool): 4 slots -- stem
  (f_i=f5, moved together), flank (f1=f4, moved together), f2, f3. Forced
  pairing is MaxUnpool2d's requirement: it needs down1/down2's pooling
  INPUT widths (initial_channels, stage1_channels) to match up5/up4's
  projected OUTPUT widths (stage5_channels, stage4_channels), i.e.
  f_i==f5 and f1==f4. stage2/stage3 are unaffected by that constraint (no
  pooling of their own -- see ENet.py's proj2_to_3), so max_unpool still
  leaves 4 real DoF once stage2/stage3 are split via ENet's 6-tuple
  channels form. All non-empty subsets of size 1-4 = C(4,1..4) = 15 configs.
  bottlenecks_per_stage stays native (4,8,8,2,1).

  Track B "UC" (channel WIDTH, decoder=upsample_conv, Cfinal_ops's actual
  pick -- no symmetry constraint): 5 independent slots -- f_i, f1, f23
  (shared, un-split), f4, f5. Subsets of size 1-2 = C(5,1)+C(5,2) = 15
  configs (stops at pairs, not all the way to 4-combined, since "less is
  fine if sufficient" and 15 is the per-track budget). bottlenecks_per_stage
  stays native.

  Track C "BD" (bottleneck DEPTH, decoder=upsample_conv, channels fixed at
  Original): 5 independent slots -- stage1, stage2, stage3, regular4,
  regular5 block counts. Depth has no bearing on max_unpool's constraint at
  all (indices only depend on channel widths), so one track covers it.
  Subsets of size 1-2 = C(5,1)+C(5,2) = 15 configs. +4 blocks per selected
  stage.

45 configs total. Channel-track bump = 1.75x (rounded to a multiple of 4);
depth-track bump = +4 blocks/stage -- both calibrated so the worst case in
each track stays close to or under ~1.1M params, the relaxed cap for this
sweep (depth has much more headroom, worst case ~516K).

Usage:
    python upscale/run_upscale_pareto.py --dry-run
    python upscale/run_upscale_pareto.py --configs MU01 UC03 BD10
    python upscale/run_upscale_pareto.py                 # full sweep, 15 epochs
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import shutil
import sys
from pathlib import Path

from pareto_common import repo_root, run_subprocess_and_parse

REPO_ROOT = repo_root()
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "compression"))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from utils import count_params  # noqa: E402

ORIGINAL = {"fi": 16, "f1": 64, "f2": 128, "f3": 128, "f4": 64, "f5": 16}
BOTTLENECKS = (4, 8, 8, 2, 1)  # native -- depth is not part of this sweep, per instruction
# Op flags fixed at Cfinal_ops's choices for both tracks -- only channel
# width varies in this sweep.
USE_DILATED = True
USE_ASYMMETRIC = True
USE_STRIDED = True
USE_DSC = False
BUMP_MULT = 1.75
MAX_STAGES_TRACK_A = 4  # stem, flank, f2, f3 -- all non-empty subsets
MAX_COMBO_SIZE_TRACK_B = 2  # fi, f1, f23, f4, f5 -- subsets up to pairs
MAX_COMBO_SIZE_TRACK_C = 2  # stage1, stage2, stage3, regular4, regular5 -- subsets up to pairs
DEPTH_BUMP = 4  # extra blocks per selected stage in track C


def _bump(value: int, mult: float = BUMP_MULT) -> int:
    return max(4, int(round(value * mult / 4)) * 4)


def _build_track_a() -> list[dict]:
    """max_unpool track: stem(f_i=f5)/flank(f1=f4)/f2/f3, sizes 1..4."""
    slots = ["stem", "flank", "f2", "f3"]
    experiments = []
    idx = 0
    for r in range(1, MAX_STAGES_TRACK_A + 1):
        for combo in itertools.combinations(slots, r):
            idx += 1
            v = dict(ORIGINAL)
            if "stem" in combo:
                v["fi"] = _bump(ORIGINAL["fi"])
                v["f5"] = v["fi"]
            if "flank" in combo:
                v["f1"] = _bump(ORIGINAL["f1"])
                v["f4"] = v["f1"]
            if "f2" in combo:
                v["f2"] = _bump(ORIGINAL["f2"])
            if "f3" in combo:
                v["f3"] = _bump(ORIGINAL["f3"])
            experiments.append({
                "id": f"MU{idx:02d}",
                "name": "mu_" + "_".join(combo),
                "track": "max_unpool",
                "channels": (v["fi"], v["f1"], v["f2"], v["f3"], v["f4"], v["f5"]),
                "bottlenecks": BOTTLENECKS,
                "decoder_type": "max_unpool",
                "hypothesis": f"max_unpool track: bump {'+'.join(combo)} by {BUMP_MULT}x vs Original.",
            })
    return experiments


def _build_track_b() -> list[dict]:
    """upsample_conv track: f_i/f1/f23/f4/f5, sizes 1..2."""
    slots = ["fi", "f1", "f23", "f4", "f5"]
    experiments = []
    idx = 0
    for r in range(1, MAX_COMBO_SIZE_TRACK_B + 1):
        for combo in itertools.combinations(slots, r):
            idx += 1
            v = {"fi": ORIGINAL["fi"], "f1": ORIGINAL["f1"], "f23": ORIGINAL["f2"],
                 "f4": ORIGINAL["f4"], "f5": ORIGINAL["f5"]}
            for s in combo:
                v[s] = _bump(v[s])
            experiments.append({
                "id": f"UC{idx:02d}",
                "name": "uc_" + "_".join(combo),
                "track": "upsample_conv",
                "channels": (v["fi"], v["f1"], v["f23"], v["f4"], v["f5"]),
                "bottlenecks": BOTTLENECKS,
                "decoder_type": "upsample_conv",
                "hypothesis": f"upsample_conv track: bump {'+'.join(combo)} by {BUMP_MULT}x vs Original.",
            })
    return experiments


def _build_track_c() -> list[dict]:
    """bottleneck-depth track: stage1/stage2/stage3/regular4/regular5 block
    counts, sizes 1..2, at fixed Original channels. Depth has no bearing on
    max_unpool's f_i==f5/f1==f4 constraint (MaxUnpool2d indices only depend
    on channel widths, not block counts -- see ENet.py's 6-tuple-channels
    comment), so decoder_type is just fixed at upsample_conv (Cfinal_ops's
    pick) rather than needing a second symmetric-track split."""
    slots = ["stage1", "stage2", "stage3", "regular4", "regular5"]
    native = {"stage1": 4, "stage2": 8, "stage3": 8, "regular4": 2, "regular5": 1}
    experiments = []
    idx = 0
    for r in range(1, MAX_COMBO_SIZE_TRACK_C + 1):
        for combo in itertools.combinations(slots, r):
            idx += 1
            bn = dict(native)
            for s in combo:
                bn[s] += DEPTH_BUMP
            experiments.append({
                "id": f"BD{idx:02d}",
                "name": "bd_" + "_".join(combo),
                "track": "bottleneck_depth",
                "channels": (ORIGINAL["fi"], ORIGINAL["f1"], ORIGINAL["f2"], ORIGINAL["f4"], ORIGINAL["f5"]),
                "bottlenecks": tuple(bn[s] for s in slots),
                "decoder_type": "upsample_conv",
                "hypothesis": f"bottleneck-depth track: +{DEPTH_BUMP} blocks to {'+'.join(combo)} vs Original's native depth.",
            })
    return experiments


EXPERIMENTS = _build_track_a() + _build_track_b() + _build_track_c()


def experiment_label(exp: dict) -> str:
    return f"{exp['id']}_{exp['name']}"


def attach_param_budgets(experiments: list[dict]) -> list[dict]:
    baseline_params = None
    result = []
    for exp in experiments:
        model = ENet(in_channels=1, out_channels=2, channels=exp["channels"],
                      bottlenecks_per_stage=exp["bottlenecks"], decoder_type=exp["decoder_type"])
        params, _ = count_params(model)
        row = dict(exp)
        row["params"] = params
        row["params_m"] = params / 1e6
        result.append(row)
        if exp["track"] == "upsample_conv" and baseline_params is None:
            pass  # baseline (Original itself) isn't in-sweep -- see docstring; no in-sweep %-delta reference
    return result


def run_experiment(exp: dict, args: argparse.Namespace, package_root: Path, output_root: Path) -> dict:
    exp_results_root = output_root / experiment_label(exp)
    if args.overwrite and exp_results_root.exists():
        shutil.rmtree(exp_results_root)
    exp_results_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(args.nnunet_raw)
    env["nnUNet_preprocessed"] = str(args.nnunet_preprocessed)
    env["nnUNet_results"] = str(exp_results_root)
    env["ENET_OUTPUT_FOLDER"] = str(exp_results_root / "fold_0")
    env["ENET_CHANNELS"] = ",".join(str(c) for c in exp["channels"])
    env["ENET_BOTTLENECKS"] = ",".join(str(c) for c in exp["bottlenecks"])
    env["ENET_DECODER_TYPE"] = exp["decoder_type"]
    env["ENET_USE_DILATED"] = "1" if USE_DILATED else "0"
    env["ENET_USE_ASYMMETRIC"] = "1" if USE_ASYMMETRIC else "0"
    env["ENET_USE_STRIDED"] = "1" if USE_STRIDED else "0"
    env["ENET_USE_DSC"] = "1" if USE_DSC else "0"
    env["ENET_EPOCHS"] = str(args.epochs)
    env["ENET_BATCH_SIZE"] = str(args.batch_size)
    env["ENET_DISABLE_CHECKPOINTING"] = "0" if args.keep_checkpoints else "1"
    env["ENET_SKIP_FINAL_VALIDATION"] = "0" if args.run_final_validation else "1"
    env["ENET_SKIP_ARCH_PLOT"] = "0" if args.plot_architecture else "1"
    env["ENET_SEED"] = str(args.seed)
    env["PYTHONHASHSEED"] = str(args.seed)
    if args.iters is not None:
        env["ENET_ITERATIONS_PER_EPOCH"] = str(args.iters)
    if args.val_iters is not None:
        env["ENET_VAL_ITERATIONS_PER_EPOCH"] = str(args.val_iters)

    command = ["nnUNetv2_train", str(args.dataset), args.configuration, str(args.fold), "-tr", "nnUNetTrainerENet"]

    print(f"\n=== {exp['id']} {exp['name']} ({exp['track']}) ===")
    print("channels:", exp["channels"], "bottlenecks:", exp["bottlenecks"])
    print(f"params:   {exp['params_m']:.3f}M")
    print("results:", exp_results_root)
    print("command:", " ".join(command))

    metrics, completed, duration_s = run_subprocess_and_parse(command, package_root, env, exp_results_root)

    mean_epoch_time = sum(metrics.epoch_times) / len(metrics.epoch_times) if metrics.epoch_times else None
    return {
        "id": exp["id"], "name": exp["name"], "track": exp["track"],
        "channels": ",".join(str(c) for c in exp["channels"]),
        "bottlenecks": ",".join(str(c) for c in exp["bottlenecks"]),
        "decoder_type": exp["decoder_type"],
        "params": exp["params"], "params_m": exp["params_m"],
        "hypothesis": exp["hypothesis"],
        "returncode": completed.returncode, "duration_s": duration_s,
        "mean_epoch_time_s": mean_epoch_time,
        "final_dice": metrics.final_dice, "best_dice": metrics.best_dice,
        "momentum": metrics.momentum, "acceleration": metrics.acceleration,
        "num_dice_epochs": len(metrics.pseudo_dice),
        "pseudo_dice_by_epoch": ";".join(f"{v:.6f}" for v in metrics.pseudo_dice),
        "results_root": str(exp_results_root),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "name", "track", "channels", "bottlenecks", "decoder_type", "params", "params_m",
        "final_dice", "best_dice", "momentum", "acceleration", "num_dice_epochs",
        "mean_epoch_time_s", "duration_s", "returncode", "hypothesis",
        "pseudo_dice_by_epoch", "results_root",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_results(rows: list[dict], output_root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots: {exc}")
        return

    plotted = [r for r in rows if r["final_dice"] is not None]
    if not plotted:
        return

    track_colors = {"max_unpool": "#2a78d6", "upsample_conv": "#eb6834", "bottleneck_depth": "#1baf7a"}
    track_markers = {"max_unpool": "o", "upsample_conv": "s", "bottleneck_depth": "^"}

    plt.figure(figsize=(9.5, 6))
    for track in ("max_unpool", "upsample_conv", "bottleneck_depth"):
        subset = [r for r in plotted if r["track"] == track]
        if not subset:
            continue
        plt.scatter([r["params_m"] for r in subset], [r["final_dice"] for r in subset],
                    c=track_colors[track], marker=track_markers[track], s=70, zorder=3, label=track)
        for r in subset:
            plt.annotate(r["id"], (r["params_m"], r["final_dice"]), textcoords="offset points",
                         xytext=(4, 4), fontsize=7)
    plt.xlabel("Parameters (M)")
    plt.ylabel("Final mean pseudo Dice (15 epochs)")
    plt.title("Upscale pareto sweep -- max_unpool vs upsample_conv tracks")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "pareto_final_dice.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Upscale pareto sweep over ENet.py's channels (default 15 epochs).")
    parser.add_argument("--configs", nargs="+", default=[e["id"] for e in EXPERIMENTS])
    parser.add_argument("--dataset", default="501")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--val-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--plot-architecture", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print configs and parameter counts, then exit.")
    parser.add_argument("--run-final-validation", action="store_true")
    parser.add_argument("--output-root", type=Path, default=root / "upscale" / "results" / "pareto_e15")
    parser.add_argument("--nnunet-raw", type=Path, default=root / "data" / "nnUNet_raw")
    parser.add_argument("--nnunet-preprocessed", type=Path, default=root / "data" / "nnUNet_preprocessed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = REPO_ROOT / "enet"
    experiments = attach_param_budgets(EXPERIMENTS)
    by_id = {e["id"]: e for e in experiments}

    unknown = [c for c in args.configs if c not in by_id]
    if unknown:
        raise ValueError(f"Unknown config IDs: {unknown}. Valid: {list(by_id)}")

    print(f"=== Upscale pareto configs ({len(args.configs)} selected of {len(EXPERIMENTS)} total) ===")
    for cid in args.configs:
        e = by_id[cid]
        print(f"  {e['id']:>5} [{e['track']:>13}] params={e['params_m']:.3f}M  "
              f"channels={e['channels']}")
    if args.dry_run:
        max_params = max(e["params"] for e in experiments)
        print(f"\nWorst-case params across all {len(EXPERIMENTS)} configs: {max_params:,}")
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid in args.configs:
        row = run_experiment(by_id[cid], args, package_root, args.output_root)
        rows.append(row)
        write_csv(rows, args.output_root / "summary.csv")
        plot_results(rows, args.output_root)
        if row["returncode"] != 0:
            print(f"Experiment {cid} failed (rc={row['returncode']}). Stopping.")
            return row["returncode"]

    print("\n=== Summary ===")
    for row in rows:
        print(f"  {row['id']:>5}  params={row['params_m']:.3f}M  final_dice={row['final_dice']}  duration_s={row['duration_s']:.0f}")
    print("CSV:", args.output_root / "summary.csv")
    print("Plot:", args.output_root / "pareto_final_dice.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

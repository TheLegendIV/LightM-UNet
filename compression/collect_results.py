"""Run inference if needed, compute per-class dice/clDice/n_components +
params/FLOPs for one trained ENet checkpoint, and write/update its row in
results.csv.

The general-purpose successor to analysis/501_ARCADE/record_architecture_stats.py
(architecture stats only, two-baseline scope) -- this covers every stage's
sweep configs, reusing the same counting (compression/utils.py) and the same
topology/dice primitives (analysis/501_ARCADE/segmentation_topology.py), not
reimplementing either.

Defaults to the current 4-class objective (Dataset509_ARCADE_1x1_4c:
background/LAD/RCA/LCX/LM) -- pass --dataset-name/--dataset-id/--out-channels
to point at a different dataset (e.g. the retired binary Dataset501_ARCADE,
see compression/slurm/archive/README.md).

Usage:
    python compression/collect_results.py \
        --net-name nnUNetTrainerENet_1_naive_baseline_Baseline --stage 1_naive_baseline \
        --channels 16,64,128,64,16 --decoder-type upsample_conv --use-prelu 0

    python compression/collect_results.py \
        --net-name nnUNetTrainerENet_E1 --stage stage1 \
        --dataset-name Dataset501_ARCADE --dataset-id 501 --out-channels 2 \
        --channels 20,72,144,72,20
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import fcntl  # POSIX only (HPC/Linux) -- guarded for local Windows dev
except ImportError:
    fcntl = None

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
ANALYSIS_ROOT = REPO_ROOT / "analysis" / "501_ARCADE"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(ANALYSIS_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.ENet import ENet, apply_block_pruning  # noqa: E402
from nnunetv2.nets.QuantENet import QuantENet  # noqa: E402
import segmentation_topology as topo  # noqa: E402
from utils import count_bops, count_buffer_elements, count_flops, count_params  # noqa: E402

NNUNET_RAW = REPO_ROOT / "data" / "nnUNet_raw"
NNUNET_PREPROCESSED = REPO_ROOT / "data" / "nnUNet_preprocessed"
NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"
RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"

# Per-class dice/cldice columns match Dataset509_ARCADE_1x1_4c's dataset.json
# labels (background=0, LAD=1, RCA=2, LCX=3, LM=4) -- hardcoded here rather
# than derived, same pragmatic single-dataset-scope choice as the rest of
# this file; update if a differently-labeled dataset is ever swept.
RESULTS_COLUMNS = [
    "config_name", "stage", "f_i", "f1", "f2", "f3", "f4", "f5",
    "bottlenecks_per_stage", "decoder_type", "ops_flags", "quant_bits",
    "params", "flops", "bops", "mem_elements",
    "dice", "dice_binary", "dice_LAD", "dice_RCA", "dice_LCX", "dice_LM",
    "cldice", "cldice_LAD", "cldice_RCA", "cldice_LCX", "cldice_LM",
    "n_components",
    "epochs", "converged_flag", "seed",
]


def parse_tuple5(value: str, name: str) -> tuple[int, ...]:
    parts = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if len(parts) != 5:
        raise ValueError(f"--{name} must have exactly 5 comma-separated integers, got {value!r}.")
    return parts


def parse_channels(value: str) -> tuple[int, ...]:
    """5 values (initial, stage1, stage2/3, stage4, stage5) or 6 if
    stage2/stage3 are split (initial, stage1, stage2, stage3, stage4,
    stage5) -- see ENet.py's 6-tuple channels form (upscale/'s max_unpool
    track)."""
    parts = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if len(parts) not in (5, 6):
        raise ValueError(f"--channels must have 5 or 6 comma-separated integers, got {value!r}.")
    return parts


def read_dataset_labels(dataset_name: str) -> dict[str, int]:
    """name -> id, straight from dataset.json's 'labels' dict (nnU-Net's own
    schema, the same source LabelManager uses to derive
    num_segmentation_heads) -- so out-channels and per-class dice columns
    both stay in sync with whatever --dataset-name a job points at, instead
    of being hardcoded per dataset."""
    dataset_json_path = NNUNET_RAW / dataset_name / "dataset.json"
    with open(dataset_json_path) as f:
        return json.load(f)["labels"]


def foreground_class_ids_and_names(labels: dict[str, int]) -> list[tuple[int, str]]:
    """[(1, 'LAD'), (2, 'RCA'), (3, 'LCX'), (4, 'LM')] for Dataset509 -- id 0
    is always background by nnU-Net convention, excluded regardless of its
    literal name key."""
    return sorted((class_id, name) for name, class_id in labels.items() if class_id != 0)


def run_inference(
    dataset_name: str,
    net_name: str,
    channels: tuple[int, ...],
    bottlenecks_per_stage: tuple[int, ...],
    decoder_type: str,
    use_dilated: bool,
    use_asymmetric: bool,
    use_strided: bool,
    configuration: str,
    plans_name: str,
    fold: int,
    checkpoint_name: str,
    device: str,
    quant_bits: int = 32,
    use_dsc: bool = False,
    context_pattern: str = "default",
    use_prelu: bool = True,
    prelu_variant: str = "standard",
    leaky_slope: float | None = None,
    leaky_slope_map: str | None = None,
    shallow_dilation: bool = False,
    separable_dilated: bool = False,
    merge_dilated_pairs: bool = False,
    dsc_dilated_only: bool = False,
    double_projections: bool = False,
    two_block_skip: bool = False,
    dsc_no_projection: bool = False,
    shallow_dilation_wide: bool = False,
    shallow_dilation_dense: bool = False,
    dsc_no_projection_context_only: bool = False,
    reg_bookend_dsc: bool = False,
    merge_reg_boundary: bool = False,
    dsc_separable: bool = False,
    pruned_blocks: str | None = None,
) -> Path:
    """Uses `nnUNetv2_predict_from_modelfolder` (-m <exact folder>), NOT
    plain `nnUNetv2_predict` (-tr/-p/-c/-d): the latter's folder resolution
    (get_output_folder) depends only on trainer_class/plans/configuration,
    identical across every sweep config -- it can't distinguish
    nnUNetTrainerENet_E1 from nnUNetTrainerENet_stage1b_no_dilated, both
    trained via the same trainer_class with ENET_OUTPUT_FOLDER redirecting
    each to its own net_name-suffixed folder (matches
    train_enet_e1.job / record_architecture_stats.py's convention, which
    DOES use net_name for the folder path). -m sidesteps that ambiguity by
    pointing at the fold-containing folder directly."""
    images_ts_dir = NNUNET_RAW / dataset_name / "imagesTs"
    prediction_dir = NNUNET_RAW / dataset_name / f"labelsPr_{net_name}"
    model_folder = NNUNET_RESULTS / dataset_name / f"{net_name}__{plans_name}__{configuration}"
    checkpoint_path = model_folder / f"fold_{fold}" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not images_ts_dir.exists():
        raise FileNotFoundError(f"imagesTs not found: {images_ts_dir}")

    prediction_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    env["nnUNet_results"] = str(NNUNET_RESULTS)
    env["ENET_CHANNELS"] = ",".join(str(c) for c in channels)
    env["ENET_BOTTLENECKS"] = ",".join(str(n) for n in bottlenecks_per_stage)
    env["ENET_DECODER_TYPE"] = decoder_type
    env["ENET_USE_DILATED"] = "1" if use_dilated else "0"
    env["ENET_USE_ASYMMETRIC"] = "1" if use_asymmetric else "0"
    env["ENET_USE_STRIDED"] = "1" if use_strided else "0"
    env["ENET_USE_DSC"] = "1" if use_dsc else "0"
    env["ENET_CONTEXT_PATTERN"] = context_pattern
    env["ENET_USE_PRELU"] = "1" if use_prelu else "0"
    env["ENET_PRELU_VARIANT"] = prelu_variant
    if leaky_slope is not None:
        env["ENET_LEAKY_SLOPE"] = str(leaky_slope)
    if leaky_slope_map is not None:
        env["ENET_LEAKY_SLOPE_MAP"] = leaky_slope_map
    if pruned_blocks is not None:
        env["ENET_PRUNED_BLOCKS"] = pruned_blocks
    env["ENET_SHALLOW_DILATION"] = "1" if shallow_dilation else "0"
    env["ENET_SEPARABLE_DILATED"] = "1" if separable_dilated else "0"
    env["ENET_MERGE_DILATED_PAIRS"] = "1" if merge_dilated_pairs else "0"
    env["ENET_DSC_DILATED_ONLY"] = "1" if dsc_dilated_only else "0"
    env["ENET_DOUBLE_PROJECTIONS"] = "1" if double_projections else "0"
    env["ENET_TWO_BLOCK_SKIP"] = "1" if two_block_skip else "0"
    env["ENET_DSC_NO_PROJECTION"] = "1" if dsc_no_projection else "0"
    env["ENET_SHALLOW_DILATION_WIDE"] = "1" if shallow_dilation_wide else "0"
    env["ENET_SHALLOW_DILATION_DENSE"] = "1" if shallow_dilation_dense else "0"
    env["ENET_DSC_NO_PROJECTION_CONTEXT_ONLY"] = "1" if dsc_no_projection_context_only else "0"
    env["ENET_REG_BOOKEND_DSC"] = "1" if reg_bookend_dsc else "0"
    env["ENET_MERGE_REG_BOUNDARY"] = "1" if merge_reg_boundary else "0"
    env["ENET_DSC_SEPARABLE"] = "1" if dsc_separable else "0"
    if quant_bits != 32:
        # Picked up by nnUNetTrainerENetQuant.build_network_architecture,
        # dynamically imported via the checkpoint's own stored trainer_name
        # (see nnUNetPredictor.initialize_from_trained_model_folder) -- not
        # passed as a CLI flag, same as every other ENET_* knob.
        env["ENET_QUANT_BITS"] = str(quant_bits)

    command = [
        shutil.which("nnUNetv2_predict_from_modelfolder") or "nnUNetv2_predict_from_modelfolder",
        "-i", str(images_ts_dir),
        "-o", str(prediction_dir),
        "-m", str(model_folder),
        "-f", str(fold),
        "-chk", checkpoint_name,
        "-device", device,
        "--disable_progress_bar",
        "--disable_tta",
    ]
    print("Running:", " ".join(command))
    completed = subprocess.run(command, cwd=PACKAGE_ROOT, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"nnUNetv2_predict failed with return code {completed.returncode}")
    return prediction_dir


def compute_eval_metrics(labels_ts_dir: Path, prediction_dir: Path, dataset_name: str) -> dict:
    """Per-class dice/clDice (mean over each class's per-case scores, one
    class label = 1..K from dataset.json) + a single mean-across-classes
    'dice'/'cldice' gate column (equally weighted across the 4 foreground
    classes, NOT case-weighted or pixel-weighted), plus a separate
    'dice_binary' -- every foreground class collapsed to one "vessel" label
    before scoring (gt > 0 vs. pred > 0), the same scoring convention the
    retired binary Dataset501 objective used throughout. 'dice' and
    'dice_binary' answer different questions and can diverge: 'dice_binary'
    only cares whether a pixel is vessel-or-not, so it's blind to class-
    confusion errors (predicting LAD where GT says LCX scores as correct);
    'dice' requires the right class too, so it's blind to nothing but also
    lets one badly-scoring class (e.g. the sparse LM) drag the mean down
    even if the other three are strong. Plus pooled (all-foreground-classes-
    together) n_components. Reuses segmentation_topology.dice_score/
    cldice_score/fragmentation_stats unchanged -- this only supplies
    per-class boolean masks (gt == class_id) via the binarize=False
    iter_matched_cases, instead of a single gt > BACKGROUND call.
    n_components stays pooled rather than per-class: fragmentation counts on
    4 sparse, often-empty-per-image anatomical branches (esp. LM) would be
    dominated by small-sample noise -- open call, revisit if per-class
    fragmentation turns out to matter."""
    labels = read_dataset_labels(dataset_name)
    class_ids_names = foreground_class_ids_and_names(labels)

    per_class_dice = {name: [] for _, name in class_ids_names}
    per_class_cldice = {name: [] for _, name in class_ids_names}
    binary_dice_list = []
    n_components_list = []
    n_cases = 0
    for _, gt, pred in topo.iter_matched_cases(labels_ts_dir, prediction_dir, binarize=False):
        n_cases += 1
        for class_id, name in class_ids_names:
            gt_mask = gt == class_id
            pred_mask = pred == class_id
            per_class_dice[name].append(topo.dice_score(gt_mask, pred_mask))
            per_class_cldice[name].append(topo.cldice_score(gt_mask, pred_mask))
        binary_dice_list.append(topo.dice_score(gt > 0, pred > 0))
        n_components_list.append(topo.fragmentation_stats(gt, pred)["pred_components"])
    if n_cases == 0:
        raise FileNotFoundError(
            f"No matched labelsTs/{prediction_dir.name} case pairs -- check predictions exist."
        )

    result = {}
    class_dice_means, class_cldice_means = [], []
    for _, name in class_ids_names:
        dice_mean = sum(per_class_dice[name]) / len(per_class_dice[name])
        cldice_mean = sum(per_class_cldice[name]) / len(per_class_cldice[name])
        result[f"dice_{name}"] = dice_mean
        result[f"cldice_{name}"] = cldice_mean
        class_dice_means.append(dice_mean)
        class_cldice_means.append(cldice_mean)
    result["dice"] = sum(class_dice_means) / len(class_dice_means)
    result["dice_binary"] = sum(binary_dice_list) / len(binary_dice_list)
    result["cldice"] = sum(class_cldice_means) / len(class_cldice_means)
    result["n_components"] = sum(n_components_list) / len(n_components_list)
    return result


def read_training_info(fold_dir: Path, checkpoint_name: str, trailing_window: int = 15) -> dict:
    """epochs/converged_flag read from the checkpoint's OWN stored metadata
    -- NOT debug.json, which turned out (checked against a real checkpoint,
    not assumed) to be a static snapshot written once at trainer
    initialization: it always shows current_epoch=0 regardless of how much
    training actually happened, silently wrong for every real run.

    converged_flag follows agent_instructions_1.yaml's actual definition
    ("false if val Dice still rising at the end"), not "did it reach
    num_epochs" -- compares the mean EMA foreground Dice over the training
    run's last two `trailing_window`-epoch halves (needs
    checkpoint_final.pth specifically for the full curve, since
    checkpoint_best.pth's own log only runs up to whichever epoch was
    best)."""
    checkpoint_path = fold_dir / checkpoint_name
    if not checkpoint_path.exists():
        print(f"No {checkpoint_name} at {fold_dir} -- epochs/converged_flag left blank.")
        return {"epochs": None, "converged_flag": None}
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    epochs = checkpoint.get("current_epoch")

    final_checkpoint_path = fold_dir / "checkpoint_final.pth"
    converged_flag = None
    if final_checkpoint_path.exists():
        final_checkpoint = torch.load(final_checkpoint_path, map_location="cpu")
        ema_dice = [float(x) for x in final_checkpoint.get("logging", {}).get("ema_fg_dice", [])]
        if len(ema_dice) >= trailing_window * 2:
            early_mean = sum(ema_dice[-trailing_window * 2:-trailing_window]) / trailing_window
            late_mean = sum(ema_dice[-trailing_window:]) / trailing_window
            converged_flag = late_mean <= early_mean  # still net rising -> NOT converged
        else:
            print(f"Only {len(ema_dice)} logged epochs in checkpoint_final.pth -- "
                  f"too few for a reliable trend (need >= {trailing_window * 2}), converged_flag left blank.")
    else:
        print(f"No checkpoint_final.pth at {fold_dir} -- converged_flag left blank (need the full loss curve).")

    return {"epochs": epochs, "converged_flag": converged_flag}


def upsert_row(row: dict) -> None:
    """Read-modify-write on the shared results.csv -- multiple Slurm array
    tasks (e.g. upscale/graduate.py's graduation array, or any of
    compression/slurm/*_array.job's cells) can call this within seconds of
    each other. Without a lock, two tasks reading the same pre-update state
    both add their own new row and the later write clobbers the earlier
    one's -- silently losing a whole run's result. flock serializes the
    whole read+merge+write critical section so every writer's read reflects
    every prior writer's completed write.

    The final to_csv is retried a few times on OSError: observed in
    practice (Windows dev machine, Docker bind mount) as a transient
    `OSError: [Errno 22] Invalid argument` specifically after a long-running
    process (real inference + thop profiling) reaches this point -- NOT
    reproducible calling upsert_row standalone/in isolation right after
    import, and NOT a lock-contention issue (single-process runs hit it
    too). Whatever underlying bind-mount flakiness causes it, it has always
    cleared within a couple of short retries in practice; this is cheap
    insurance against losing an otherwise-fully-computed row (real
    inference + metrics already ran by this point) to what is, empirically,
    a few-hundred-millisecond hiccup."""
    row = {col: row.get(col) for col in RESULTS_COLUMNS}
    lock_path = RESULTS_CSV.parent / ".results.csv.lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if RESULTS_CSV.exists():
                existing = pd.read_csv(RESULTS_CSV)
                existing = existing[existing["config_name"] != row["config_name"]]
                combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            else:
                combined = pd.DataFrame([row], columns=RESULTS_COLUMNS)
            last_error = None
            for attempt in range(5):
                try:
                    combined.to_csv(RESULTS_CSV, index=False)
                    last_error = None
                    break
                except OSError as error:
                    last_error = error
                    print(f"  [retry {attempt + 1}/5] results.csv write failed ({error}), retrying...")
                    time.sleep(1.0)
            if last_error is not None:
                raise last_error
            print(f"Wrote {RESULTS_CSV} ({len(combined)} rows).")
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net-name", required=True, help="config_name, and the labelsPr_<net-name> prediction folder.")
    parser.add_argument("--dataset-name", default="Dataset509_ARCADE_1x1_4c",
                         help="nnUNet_raw/<name> and nnUNet_results/<name> subfolder. Any future "
                              "Dataset509-derived sweep stage passes this (or relies on the default) -- "
                              "not narrowly bolted onto just one job.")
    parser.add_argument("--dataset-id", default="509",
                         help="Informational only (not consumed by this script -- kept for parity with "
                              "the training job's own DATASET_ID, same convention as --trainer-class).")
    parser.add_argument("--trainer-class", default="nnUNetTrainerENet", help="Informational only (not used for inference -- see run_inference's docstring). Kept for parity with the training job's -tr flag.")
    parser.add_argument("--stage", required=True, help="e.g. stage1, stage1b, stage2, early_probe.")
    parser.add_argument("--channels", required=True, type=parse_channels)
    parser.add_argument("--bottlenecks", default="4,8,8,2,1", type=lambda v: parse_tuple5(v, "bottlenecks"))
    parser.add_argument("--decoder-type", default="max_unpool", choices=["max_unpool", "upsample_conv"])
    parser.add_argument("--use-dilated", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-asymmetric", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-strided", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-dsc", type=int, default=0, choices=[0, 1], help="Depthwise-separable inner conv (rejects combination with --use-asymmetric 1).")
    parser.add_argument("--context-pattern", default="default",
                         choices=["default", "sparse", "dense_dilation", "dense_dilation_a", "dense_dilation_lead1",
                                  "dense_dilation_reg_interleaved", "dense_dilation_reg_trailing",
                                  "d16_reg_interleaved", "dense_dilation_reg_interleaved_double_mid",
                                  "dense_dilation_d2_projected", "dense_dilation_d8_d16_projected"],
                         help="'sparse' = regular/dilated4/regular/dilated16 (section 2a's div2/div4 bottleneck axis), no 2/8 rungs, never asymmetric. 'dense_dilation' = every context-stage slot dilated (2/4/8/16 repeated twice over 8 slots), no plain/asymmetric slots at all. 'dense_dilation_a' = same but with the gridding-investigation's coprime (1,5,7,17) schedule instead of (2,4,8,16). 'dense_dilation_reg_interleaved' = (2,4,8,16) with a full RegularBottleneck bookending each cycle (needs dsc_no_projection=1 and stage2/3 bottleneck depth 11, see ENet.py's DENSE_DILATION_REG_INTERLEAVED_PATTERN). 'dense_dilation_reg_trailing' = (2,4,8,16,reg) repeating, reg TRAILS each dilation cycle instead of leading it, meant for use_dsc=1 + dsc_no_projection=0 (DSC WITH projection kept) at stage2/3 bottleneck depth 10, see ENet.py's DENSE_DILATION_REG_TRAILING_PATTERN. 'd16_reg_interleaved' = S15's own pattern: dilation=16 ONLY repeated 4 times, reg-bookended on both sides of EVERY repeat (reg,16,reg,16,reg,16,reg,16,reg, 9 slots) -- needs dsc_no_projection=1 and stage2/3 bottleneck depth 9, see ENet.py's D16_REG_INTERLEAVED_PATTERN. 'dense_dilation_reg_interleaved_double_mid' = S19's own pattern: same as dense_dilation_reg_interleaved but the one truly-interior reg (between a stage's own two dilation cycles) is doubled into two consecutive RegularBottlenecks (reg,2,4,8,16,reg,reg,2,4,8,16,reg, 12 slots) -- leading/trailing bookends stay single, so the stage2/3 seam is unchanged; meant for dsc_no_projection=0 (S10's own recipe), stage2/3 bottleneck depth 12, see ENet.py's DENSE_DILATION_REG_INTERLEAVED_DOUBLE_MID_PATTERN. 'dense_dilation_d2_projected'/'dense_dilation_d8_d16_projected' = plain dense_dilation (8 slots, no reg bookends, native depth 8) but under dsc_no_projection=1, the named dilation rate(s) (d=2 only, or d=8+d=16) become a real DILATED projected RegularBottleneck (full-rank inner conv, real reduce/expand, restoring cross-channel mixing at that rate) instead of DSCNoProjectionBottleneck -- the other rates stay DSCNoProjectionBottleneck (no proj + DSC), see ENet.py's DENSE_DILATION_D2_PROJECTED_PATTERN/DENSE_DILATION_D8_D16_PROJECTED_PATTERN.")
    parser.add_argument("--use-prelu", type=int, default=1, choices=[0, 1], help="0 = collapse the encoder's PReLU to plain ReLU too (section 1d's ablation) -- decoder is always ReLU regardless, see ENet.py.")
    parser.add_argument("--prelu-variant", default="standard", choices=["standard", "leaky", "nonneg", "nonneg_block"],
                         help="Only meaningful with --use-prelu 1. 'standard' = real learnable nn.PReLU (default). "
                              "'leaky' = fixed nn.LeakyReLU(0.01), no learnable params. 'nonneg' = learnable PReLU "
                              "clamped to a>=0 every forward pass, ruling out the sign-flipping negative-slope "
                              "behavior real PReLU sometimes learns (see compression/plot_prelu_activations.py). "
                              "'nonneg_block' = ONE learnable non-negative scalar SHARED across every activation "
                              "site within a bottleneck block (not per-channel) -- the FINN-deployable target: "
                              "converts losslessly to a fixed per-block LeakyReLU at inference via "
                              "ENet.py's apply_leaky_slope_overrides, since there's only one real trained number "
                              "per block to begin with. See ENet.py's PReluVariant/_activation.")
    parser.add_argument("--leaky-slope", type=float, default=None,
                         help="Only meaningful with --prelu-variant leaky. Overrides that variant's fixed 0.01 "
                              "negative_slope network-wide with this value (e.g. a mean slope collected from an "
                              "already-trained PReLU checkpoint via ENet.py's collect_prelu_global_mean). See "
                              "ENet.py's apply_leaky_slope_overrides.")
    parser.add_argument("--leaky-slope-map", default=None,
                         help="Only meaningful with --prelu-variant leaky. JSON string mapping bottleneck-container "
                              "dotted module names (see ENet.py's collect_prelu_block_means) to a per-block "
                              "negative_slope override, e.g. '{\"stage2.0\": 0.13, \"stage2.1\": 0.09}'. Any "
                              "LeakyReLU site not covered by a key here falls back to --leaky-slope. See ENet.py's "
                              "apply_leaky_slope_overrides.")
    parser.add_argument("--pruned-blocks", default=None,
                         help="Comma-separated dotted block names to post-training structural-ablate (replace "
                              "with nn.Identity(), no retraining), e.g. 'stage3.0' or 'stage3.0,stage2.5'. Only "
                              "sound for channel-preserving residual blocks -- see ENet.py's apply_block_pruning.")
    parser.add_argument("--shallow-dilation", type=int, default=0, choices=[0, 1], help="Stage 4.4.1: alternating regular/dilated(16) in stage1 and regular4 (regular5 unchanged). See ENet.py.")
    parser.add_argument("--shallow-dilation-wide", type=int, default=0, choices=[0, 1], help="Like --shallow-dilation but at dilation=32 instead of 16 -- stage1/regular4 run at 1/4 resolution, coarser than stage2/3, so there's headroom for a wider rate there. See ENet.py's SHALLOW_DILATION_WIDE_PATTERN.")
    parser.add_argument("--shallow-dilation-dense", type=int, default=0, choices=[0, 1], help="Stage 5.2: swaps --shallow-dilation/--shallow-dilation-wide's alternating pattern for an every-slot-dilated one (needs one of those two set). See ENet.py's SHALLOW_DILATION_DENSE_PATTERN / SHALLOW_DILATION_WIDE_DENSE_PATTERN.")
    parser.add_argument("--dsc-no-projection-context-only", type=int, default=0, choices=[0, 1], help="Stage 8.1: narrows --dsc-no-projection's scope to stage2/stage3 only -- regular1/regular4/regular5 revert to normal projected bottlenecks (needs --dsc-no-projection 1 also set).")
    parser.add_argument("--reg-bookend-dsc", type=int, default=0, choices=[0, 1], help="Stage 8.3: applies DSC (with projection kept) to context_pattern=dense_dilation_reg_interleaved's reg-bookend slots, instead of leaving them full-rank. No-op without that context_pattern and --dsc-no-projection 1.")
    parser.add_argument("--merge-reg-boundary", type=int, default=0, choices=[0, 1], help="S15-derived probe: skips stage3's own leading reg-bookend (shifts its pattern-index origin by +1) so it doesn't sit directly next to stage2's trailing reg with nothing dilated between them -- relies on stage2's own trailing reg to cover the seam alone. Shrink stage3's own bottleneck depth by 1 to compensate (e.g. 11 -> 10 for dense_dilation_reg_interleaved). Needs --dsc-no-projection 1. See ENet.py's merge_reg_boundary.")
    parser.add_argument("--dsc-separable", type=int, default=0, choices=[0, 1], help="Stage 18: factors DSCNoProjectionBottleneck's own depthwise KxK into a (K,1)+(1,K) depthwise pair, applied at every dilation rate. Needs --dsc-no-projection 1. See ENet.py's dsc_separable.")
    parser.add_argument("--separable-dilated", type=int, default=0, choices=[0, 1], help="Stage 4.4.2: factor every dilated 3x3 in the context pattern into a (3,1)+(1,3) pair, same dilation on both passes.")
    parser.add_argument("--merge-dilated-pairs", type=int, default=0, choices=[0, 1], help="Stage 4.4.3: fuse each (regular-or-asymmetric, dilated) context-pattern PAIR into one block with a shared reduce/expand -- halves stage2/3's block count.")
    parser.add_argument("--dsc-dilated-only", type=int, default=0, choices=[0, 1], help="Stage 4.4.4: depthwise-separable ONLY on the context pattern's dilated slots, independent of --use-dsc.")
    parser.add_argument("--double-projections", type=int, default=0, choices=[0, 1], help="Stage 4.5: stack an extra 1x1 conv+BN+act in every bottleneck's reduce AND expand projection, network-wide.")
    parser.add_argument("--two-block-skip", type=int, default=0, choices=[0, 1], help="Stage 4.6: extra short residual spanning every 2 consecutive context-stage blocks, on top of each block's own internal residual, in stage2 and stage3.")
    parser.add_argument("--dsc-no-projection", type=int, default=0, choices=[0, 1], help="DSC everywhere (stage1/regular1, stage2, stage3, regular4, regular5), with NO reduce/expand 1x1 projection pair -- depthwise+pointwise applied directly at full channel width. Requires --use-asymmetric 0.")
    parser.add_argument("--quant-bits", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=None,
                         help="Defaults to len(dataset.json['labels']) for --dataset-name (5 for "
                              "Dataset509_ARCADE_1x1_4c: background+LAD+RCA+LCX+LM) -- matches "
                              "nnU-Net's own LabelManager.num_segmentation_heads derivation.")
    parser.add_argument("--input-hw", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    parser.add_argument("--plans-name", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-inference", action="store_true", help="Assume labelsPr_<net-name> already exists.")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    if args.out_channels is None:
        args.out_channels = len(read_dataset_labels(dataset_name))

    # FLOPs/MACs always come from the plain FP32 ENet: thop silently
    # undercounts a QuantENet by ~40x (doesn't recognize Brevitas's quant
    # layers as countable ops -- measured, see utils.count_bops's
    # docstring), and MAC count is topology-determined, not quantization-
    # determined (verified topology-identical between ENet/QuantENet).
    fp32_model = ENet(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        channels=args.channels,
        bottlenecks_per_stage=args.bottlenecks,
        decoder_type=args.decoder_type,
        use_dilated=bool(args.use_dilated),
        use_asymmetric=bool(args.use_asymmetric),
        use_strided=bool(args.use_strided),
        use_dsc=bool(args.use_dsc),
        context_pattern=args.context_pattern,
        use_prelu=bool(args.use_prelu),
        prelu_variant=args.prelu_variant,
        shallow_dilation=bool(args.shallow_dilation),
        separable_dilated=bool(args.separable_dilated),
        merge_dilated_pairs=bool(args.merge_dilated_pairs),
        dsc_dilated_only=bool(args.dsc_dilated_only),
        double_projections=bool(args.double_projections),
        two_block_skip=bool(args.two_block_skip),
        dsc_no_projection=bool(args.dsc_no_projection),
        shallow_dilation_wide=bool(args.shallow_dilation_wide),
        shallow_dilation_dense=bool(args.shallow_dilation_dense),
        dsc_no_projection_context_only=bool(args.dsc_no_projection_context_only),
        reg_bookend_dsc=bool(args.reg_bookend_dsc),
        merge_reg_boundary=bool(args.merge_reg_boundary),
        dsc_separable=bool(args.dsc_separable),
    )
    if args.pruned_blocks:
        apply_block_pruning(fp32_model, [name.strip() for name in args.pruned_blocks.split(",") if name.strip()])
    # FINN sliding-window-buffer memory estimate (activation elements, not
    # bits -- see utils.count_buffer_elements's own docstring). Always from
    # the FP32 reference model, same rationale as MACs below: buffer size is
    # topology/channel-width-determined, not quantization-determined. MUST
    # run before count_flops: thop's profile() leaves its own forward hooks
    # registered on every Conv2d after it returns (a thop quirk -- it cleans
    # up the total_ops/total_params buffers it adds but not the hooks that
    # reference them), so a second forward pass on the same model afterward
    # crashes those stale hooks with AttributeError. Running the buffer pass
    # first keeps it on a hook-free model.
    mem_elements = count_buffer_elements(fp32_model, args.in_channels, tuple(args.input_hw))["total"]
    macs, flops = count_flops(fp32_model, args.in_channels, tuple(args.input_hw))

    if args.quant_bits == 32:
        total_params, _ = count_params(fp32_model)
        bops = None
    else:
        # Real deployed model's param count (close to but not identical to
        # the FP32 count -- Brevitas quant layers carry a little extra
        # bookkeeping, e.g. scale factors).
        quant_model = QuantENet(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=args.channels,
            bottlenecks_per_stage=args.bottlenecks,
            decoder_type=args.decoder_type,
            use_dilated=bool(args.use_dilated),
            use_asymmetric=bool(args.use_asymmetric),
            use_strided=bool(args.use_strided),
            use_dsc=bool(args.use_dsc),
            weight_bit_width=args.quant_bits,
            act_bit_width=args.quant_bits,
            context_pattern=args.context_pattern,
            dsc_no_projection=bool(args.dsc_no_projection),
            dsc_no_projection_context_only=bool(args.dsc_no_projection_context_only),
            separable_dilated=bool(args.separable_dilated),
            leaky_slope_map=json.loads(args.leaky_slope_map) if args.leaky_slope_map else None,
        )
        total_params, _ = count_params(quant_model)
        bops = count_bops(macs, args.quant_bits)

    prediction_dir = NNUNET_RAW / dataset_name / f"labelsPr_{args.net_name}"
    if not args.skip_inference and not prediction_dir.exists():
        prediction_dir = run_inference(
            dataset_name=dataset_name,
            net_name=args.net_name,
            channels=args.channels,
            bottlenecks_per_stage=args.bottlenecks,
            decoder_type=args.decoder_type,
            use_dilated=bool(args.use_dilated),
            use_asymmetric=bool(args.use_asymmetric),
            use_strided=bool(args.use_strided),
            configuration=args.configuration,
            plans_name=args.plans_name,
            fold=args.fold,
            checkpoint_name=args.checkpoint_name,
            device=args.device,
            quant_bits=args.quant_bits,
            use_dsc=bool(args.use_dsc),
            context_pattern=args.context_pattern,
            use_prelu=bool(args.use_prelu),
            prelu_variant=args.prelu_variant,
            leaky_slope=args.leaky_slope,
            leaky_slope_map=args.leaky_slope_map,
            shallow_dilation=bool(args.shallow_dilation),
            separable_dilated=bool(args.separable_dilated),
            merge_dilated_pairs=bool(args.merge_dilated_pairs),
            dsc_dilated_only=bool(args.dsc_dilated_only),
            double_projections=bool(args.double_projections),
            two_block_skip=bool(args.two_block_skip),
            dsc_no_projection=bool(args.dsc_no_projection),
            shallow_dilation_wide=bool(args.shallow_dilation_wide),
            shallow_dilation_dense=bool(args.shallow_dilation_dense),
            dsc_no_projection_context_only=bool(args.dsc_no_projection_context_only),
            reg_bookend_dsc=bool(args.reg_bookend_dsc),
            merge_reg_boundary=bool(args.merge_reg_boundary),
            dsc_separable=bool(args.dsc_separable),
            pruned_blocks=args.pruned_blocks,
        )
    labels_ts_dir = NNUNET_RAW / dataset_name / "labelsTs"
    eval_metrics = compute_eval_metrics(labels_ts_dir, prediction_dir, dataset_name)

    fold_dir = (
        NNUNET_RESULTS / dataset_name / f"{args.net_name}__{args.plans_name}__{args.configuration}"
        / f"fold_{args.fold}"
    )
    training_info = read_training_info(fold_dir, args.checkpoint_name)

    # args.channels is ENet.py's 5-value tuple (initial, stage1, stage23,
    # stage4, stage5) -- stage2/stage3 share one value architecturally --
    # or the 6-value split form (initial, stage1, stage2, stage3, stage4,
    # stage5) used by upscale/'s max_unpool track. results.csv's schema
    # always has separate f2/f3 columns (equal in the 5-value case).
    if len(args.channels) == 5:
        f_i, f1, f23, f4, f5 = args.channels
        f2, f3 = f23, f23
    else:
        f_i, f1, f2, f3, f4, f5 = args.channels
    row = {
        "config_name": args.net_name,
        "stage": args.stage,
        "f_i": f_i, "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5,
        "bottlenecks_per_stage": ",".join(str(n) for n in args.bottlenecks),
        "decoder_type": args.decoder_type,
        # prelu is n/a for quant rows: QuantENet (unlike ENet) takes no
        # use_prelu param at all -- it hardcodes QuantReLU everywhere since
        # Brevitas/FINN has no quantized PReLU op (see QuantENet.py's module
        # docstring). --use-prelu's default only ever describes the FP32
        # reference model built above for FLOPs counting, not the actual
        # trained network, so stamping it here for quant rows would silently
        # misrepresent the real architecture.
        "ops_flags": (
            f"dilated={args.use_dilated},asymmetric={args.use_asymmetric},strided={args.use_strided},dsc={args.use_dsc},context_pattern={args.context_pattern},prelu="
            + ("n/a(quant-forces-relu)" if args.quant_bits != 32 else str(args.use_prelu))
            + ",prelu_variant="
            + ("n/a(quant-forces-relu)" if args.quant_bits != 32 else args.prelu_variant)
            + f",leaky_slope={args.leaky_slope},leaky_slope_map={args.leaky_slope_map}"
            + f",shallow_dilation={args.shallow_dilation},separable_dilated={args.separable_dilated}"
            + f",merge_dilated_pairs={args.merge_dilated_pairs},dsc_dilated_only={args.dsc_dilated_only}"
            + f",double_projections={args.double_projections},two_block_skip={args.two_block_skip}"
            + f",dsc_no_projection={args.dsc_no_projection},shallow_dilation_wide={args.shallow_dilation_wide}"
            + f",shallow_dilation_dense={args.shallow_dilation_dense}"
            + f",dsc_no_projection_context_only={args.dsc_no_projection_context_only},reg_bookend_dsc={args.reg_bookend_dsc}"
            + f",merge_reg_boundary={args.merge_reg_boundary}"
            + f",dsc_separable={args.dsc_separable}"
            + f",pruned_blocks={args.pruned_blocks}"
        ),
        "quant_bits": args.quant_bits,
        "params": total_params,
        "flops": flops,
        "bops": bops,
        "mem_elements": mem_elements,
        "seed": args.seed,
        **eval_metrics,
        **training_info,
    }
    upsert_row(row)
    for key, value in row.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

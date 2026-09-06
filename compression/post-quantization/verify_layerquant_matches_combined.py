"""Decisive smoke test: does LayerQuantENet EXACTLY match CombinedQuantENet
when built from the SAME block_bits_*.json (broadcast to every layer inside
each block via LayerQuantENet.expand_block_bits_to_layer_bits), the SAME
FP32 source checkpoint, and calibrated on the LITERAL SAME list of
calibration image tensors (not just the same --calibration-seed -- the same
in-memory objects, in the same order, fed to both models' calibration loops
in this one process)?

WHY this design, not two separate calibrate_*.py invocations: an earlier
session found a ~0.041 dice gap between independently-produced
"..._perblock_VERIFY_alpha0.5_calibrated" and
"..._perlayer_VERIFY_alpha0.5_calibrated" checkpoints, versus a ~0.017
seed-to-seed noise floor measured from two independently-calibrated
CombinedQuantENet instances -- ambiguous with only one noise sample. This
script removes every remaining confound between the two architectures'
calibration runs (same checkpoint, same calibration data AND order, same
process) so any remaining gap is attributable to the architectures/
calibration mechanism, not to which random images each one happened to see.

A single forward pass at uniform (pre-calibration) bits was already
confirmed bit-identical (max_abs_diff=0.0) between the two architectures in
an earlier session -- this script extends that check through calibration and
through nnU-Net's own dice metric on real held-out validation cases.

Usage:
    python compression/post-quantization/verify_layerquant_matches_combined.py \\
        --block-bits-file compression/hawq/artifacts/block_bits_folding_12_separable_dense_relu_joint_alpha0.5_maxlat1000ms.json \\
        --n-calibration-images 32 --n-val-cases 30
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from brevitas.graph.calibrate import calibration_mode

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.CombinedQuantENet import CombinedQuantENet  # noqa: E402
from nnunetv2.nets.LayerQuantENet import LayerQuantENet, expand_block_bits_to_layer_bits, layer_names_for  # noqa: E402

NNUNET_PREPROCESSED = REPO_ROOT / "data" / "nnUNet_preprocessed"
NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"
DATASET_NAME = "Dataset509_ARCADE_1x1_4c"

CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation"
COMMON_KWARGS = dict(
    out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
    context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
    use_dsc=False, dsc_no_projection=False, separable_dilated=True, trainable_slope=False,
)


def load_calibration_batches(n_images: int, seed: int) -> list[torch.Tensor]:
    """Byte-identical to calibrate_12_separable_dense_relu_per{block,layer}.py's
    own helper -- reused here only so the SAME list of tensors (not just the
    same seed) is fed to both models below."""
    preprocessed_dir = NNUNET_PREPROCESSED / DATASET_NAME / "nnUNetPlans_2d"
    image_files = sorted(p for p in preprocessed_dir.glob("*.npy") if not p.name.endswith("_seg.npy"))
    rng = random.Random(seed)
    sampled = rng.sample(image_files, k=min(n_images, len(image_files)))
    return [torch.from_numpy(np.load(p)).float() for p in sampled]


def calibrate(model: torch.nn.Module, batches: list[torch.Tensor], device: str) -> int:
    model.to(device)
    model.train()
    n_used = 0
    with torch.no_grad(), calibration_mode(model):
        for batch in batches:
            try:
                model(batch.to(device))
                n_used += 1
            except RuntimeError as error:
                print(f"  [skip] calibration image with shape {tuple(batch.shape)} failed: {error}")
    model.eval()
    return n_used


def load_case(case_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    d = NNUNET_PREPROCESSED / DATASET_NAME / "nnUNetPlans_2d"
    img = torch.from_numpy(np.load(d / f"{case_id}.npy")).float()
    seg = torch.from_numpy(np.load(d / f"{case_id}_seg.npy")).long()
    seg[seg == -1] = 0
    return img, seg


def hard_dice_per_class(pred: torch.Tensor, gt: torch.Tensor, n_classes: int) -> np.ndarray:
    dice = np.full(n_classes, np.nan)
    for c in range(n_classes):
        p, g = (pred == c), (gt == c)
        union = p.sum().item() + g.sum().item()
        if union == 0:
            continue
        dice[c] = 2 * (p & g).sum().item() / union
    return dice


def evaluate(model: torch.nn.Module, case_ids: list[str], device: str, n_classes: int) -> dict:
    """Skips a case on RuntimeError (the known odd-H/W-not-divisible-by-
    stride-8 quirk both architectures share, see calibrate()'s own identical
    skip) -- callers comparing two models' per-case logits must intersect on
    "used_case_ids" rather than assuming case_ids itself, since a case that
    fails for one architecture is guaranteed (same quirk, same shape check)
    to fail for the other too, but this isn't re-verified here."""
    model.eval()
    all_dice, all_logits, used_case_ids = [], {}, []
    with torch.no_grad():
        for case_id in case_ids:
            img, seg = load_case(case_id)
            try:
                out = model(img.to(device))
            except RuntimeError as error:
                print(f"  [skip] val case {case_id} shape {tuple(img.shape)} failed forward pass: {error}")
                continue
            out_t = out.value if hasattr(out, "value") else out
            all_logits[case_id] = out_t.cpu()
            used_case_ids.append(case_id)
            pred = out_t.argmax(dim=1).squeeze(0).cpu()
            gt = seg.squeeze(0).squeeze(0)
            all_dice.append(hard_dice_per_class(pred, gt, n_classes))
    stacked = np.stack(all_dice)
    per_class_mean = np.nanmean(stacked, axis=0)
    return {
        "per_class_dice": per_class_mean.tolist(),
        "mean_foreground_dice": float(np.nanmean(per_class_mean[1:])),
        "logits": all_logits,
        "used_case_ids": used_case_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block-bits-file", type=Path, required=True)
    parser.add_argument("--source-net-name", default="nnUNetTrainerENet_12_separable_dense_relu")
    parser.add_argument("--source-checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--n-calibration-images", type=int, default=32)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--n-val-cases", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.block_bits_file) as f:
        block_bits = json.load(f)
    block_weight_bits, block_act_bits = block_bits["stage_weight_bits"], block_bits["stage_act_bits"]

    source_checkpoint = (
        NNUNET_RESULTS / DATASET_NAME / f"{args.source_net_name}__nnUNetPlans__2d"
        / "fold_0" / args.source_checkpoint_name
    )
    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)
    print(f"Source FP32 checkpoint: {source_checkpoint}")
    print(f"Block bits file: {args.block_bits_file}")

    print("\nBuilding CombinedQuantENet from block bits...")
    combined = CombinedQuantENet.from_pretrained(source_checkpoint, block_weight_bits, block_act_bits, **COMMON_KWARGS)

    print("Building LayerQuantENet from the SAME block bits, broadcast per-layer...")
    layer_weight_names, layer_act_names = layer_names_for(
        out_channels=COMMON_KWARGS["out_channels"], channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, separable_dilated=True,
    )
    layer_weight_bits, layer_act_bits = expand_block_bits_to_layer_bits(
        block_weight_bits, block_act_bits, layer_weight_names, layer_act_names,
    )
    layer_quant = LayerQuantENet.from_pretrained(source_checkpoint, layer_weight_bits, layer_act_bits, **COMMON_KWARGS)

    print(f"\nLoading {args.n_calibration_images} calibration images (seed={args.calibration_seed}) -- "
          f"the SAME tensor list will be fed to BOTH models below.")
    calibration_batches = load_calibration_batches(args.n_calibration_images, args.calibration_seed)

    print("Calibrating CombinedQuantENet...")
    torch.manual_seed(args.calibration_seed)
    n_used_combined = calibrate(combined, calibration_batches, args.device)
    print(f"  used {n_used_combined}/{len(calibration_batches)} images.")
    combined.to("cpu")

    print("Calibrating LayerQuantENet (same images, same order, RNG RE-SEEDED to the same state -- "
          "DownsamplingBottleneck/UpsamplingBottleneck's nn.Dropout2d is ACTIVE during calibration (model.train() "
          "is required for Brevitas's observer hooks to fire), so without this reseed the two models' dropout "
          "masks diverge from here on, corrupting residual_add's calibrated activation-quantizer scale "
          "non-reproducibly even given identical calibration images).")
    torch.manual_seed(args.calibration_seed)
    n_used_layer = calibrate(layer_quant, calibration_batches, args.device)
    print(f"  used {n_used_layer}/{len(calibration_batches)} images.")
    layer_quant.to("cpu")

    print("\n=== Calibrated weight-quantizer scale spot-check (same layer name, both models) ===")
    combined_modules = dict(combined.named_modules())
    layer_modules = dict(layer_quant.named_modules())
    spot_check_names = [n for n in layer_weight_names if n in combined_modules and n in layer_modules][:6] + \
        [n for n in layer_weight_names if n in combined_modules and n in layer_modules][-3:]
    for name in dict.fromkeys(spot_check_names):
        cm, lm = combined_modules[name], layer_modules[name]
        if not (hasattr(cm, "quant_weight") and hasattr(lm, "quant_weight")):
            continue
        cs = cm.quant_weight().scale
        ls = lm.quant_weight().scale
        cs_val = cs.item() if cs.numel() == 1 else cs.flatten()[:3].tolist()
        ls_val = ls.item() if ls.numel() == 1 else ls.flatten()[:3].tolist()
        match = "MATCH" if torch.allclose(cs, ls, rtol=1e-5, atol=1e-8) else "DIFFERS"
        print(f"  {name:30s} combined_scale={cs_val}  layer_scale={ls_val}  [{match}]")

    assert n_used_combined == n_used_layer, (
        f"Calibration used a DIFFERENT number of images for each model ({n_used_combined} vs {n_used_layer}) -- "
        f"one architecture is failing forward on an image shape the other tolerates. Investigate before trusting "
        f"any dice comparison below."
    )

    print("\n=== Layer-by-layer output trace on ONE common image (finds first divergence point) ===")
    with open(REPO_ROOT / "data" / "nnUNet_preprocessed" / DATASET_NAME / "splits_final.json") as f:
        splits_probe = json.load(f)
    probe_img = None
    for case_id in splits_probe[0]["val"]:
        img, _ = load_case(case_id)
        h, w = img.shape[-2], img.shape[-1]
        if h % 8 == 0 and w % 8 == 0:
            probe_img = (case_id, img)
            break
    if probe_img is None:
        print("  No clean (H,W divisible by 8) val case found -- skipping trace.")
    else:
        case_id, img = probe_img
        print(f"  Using case {case_id}, shape {tuple(img.shape)}")
        trace_names = ["initial", "down1.pool", "down1.reduce", "down1.conv", "down1.expand",
                       "down1.residual_add", "down1.out_act", "down1", "down2", "up4", "up5", "final"]
        trace_names = [n for n in trace_names if n and n in combined_modules and n in layer_modules]
        captured = {"combined": {}, "layer": {}}

        def make_hook(store, name):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                out_t = out.value if hasattr(out, "value") else out
                store[name] = out_t.detach().clone()
            return hook

        handles = []
        for n in trace_names:
            handles.append(combined_modules[n].register_forward_hook(make_hook(captured["combined"], n)))
            handles.append(layer_modules[n].register_forward_hook(make_hook(captured["layer"], n)))
        combined.eval()
        layer_quant.eval()
        with torch.no_grad():
            combined(img)
            layer_quant(img)
        for h in handles:
            h.remove()
        for n in trace_names:
            a, b = captured["combined"][n], captured["layer"][n]
            diff = (a - b).abs().max().item()
            print(f"  after {n:15s} max_abs_diff={diff:.6e}  {'[DIVERGES HERE]' if diff > 1e-4 else '[match]'}")

    with open(REPO_ROOT / "data" / "nnUNet_preprocessed" / DATASET_NAME / "splits_final.json") as f:
        splits = json.load(f)
    val_cases = splits[0]["val"][: args.n_val_cases]
    print(f"\nEvaluating both on {len(val_cases)} real held-out fold_0 validation cases...")

    combined_result = evaluate(combined, val_cases, args.device, COMMON_KWARGS["out_channels"])
    layer_result = evaluate(layer_quant, val_cases, args.device, COMMON_KWARGS["out_channels"])

    common_cases = sorted(set(combined_result["used_case_ids"]) & set(layer_result["used_case_ids"]))
    if len(common_cases) < len(val_cases):
        print(f"\nNote: {len(val_cases) - len(common_cases)}/{len(val_cases)} val case(s) skipped (odd-shape "
              f"forward-pass failure) -- comparing logits on the {len(common_cases)} cases both models produced.")

    max_abs_diff = 0.0
    n_argmax_mismatch = 0
    for case_id in common_cases:
        a, b = combined_result["logits"][case_id], layer_result["logits"][case_id]
        max_abs_diff = max(max_abs_diff, (a - b).abs().max().item())
        if not torch.equal(a.argmax(dim=1), b.argmax(dim=1)):
            n_argmax_mismatch += 1

    print("\n=== Results ===")
    print(f"CombinedQuantENet  mean foreground Dice ({len(combined_result['used_case_ids'])} cases): {combined_result['mean_foreground_dice']:.4f}")
    print(f"LayerQuantENet     mean foreground Dice ({len(layer_result['used_case_ids'])} cases): {layer_result['mean_foreground_dice']:.4f}")
    print(f"Delta:                                   {layer_result['mean_foreground_dice'] - combined_result['mean_foreground_dice']:+.4f}")
    print(f"\nMax abs logit diff across {len(common_cases)} common val cases: {max_abs_diff:.6e}")
    print(f"Cases with a DIFFERENT argmax prediction anywhere: {n_argmax_mismatch}/{len(common_cases)}")
    print(f"\nVERDICT: {'EXACT MATCH' if max_abs_diff < 1e-4 and n_argmax_mismatch == 0 else 'MISMATCH -- see above'}")


if __name__ == "__main__":
    main()

"""Unified S12-dense QONNX export (pytorch/host side) + conv-order dump.

Presets (override any field with --channels/--bottlenecks/...):
  --resolution 512 : decoder nearest_upsample,      context dense_dilation,      CHANNELS (4,16,32,16,4), Dataset509_ARCADE_1x1_4c
  --resolution 256 : decoder nearest_conv_upsample, context dense_dilation_half, CHANNELS (4,16,32,16,4), Dataset510_ARCADE_256_4c
BOTTLENECKS (4,8,8,2,1) for both.

Weights: --checkpoint <nnU-Net checkpoint_best.pth[.txt]> (LayerQuantEnetFINN.from_pretrained) is required --
there is no dummy-weight fallback.
Bits: --bits-file <MILP layer_bits_SITES_*.json> (sites missing from it fall back to 6 bits).

Writes <out-dir>/<name>.onnx and <out-dir>/<name>_conv_order.json (the
latter is what finn_s12_build.py --conv-order needs; it is bit-independent).

Example (reproduces the S12_dense_dsr_ablation_v1 dsr_off export):
    python hardware/s12_export.py --resolution 512 --tag dsr_off \\
        --bits-file MILP/artifacts/S12_dense_dsr_ablation_v1/dsr_off/layer_bits_SITES_dsr_off.json \\
        --checkpoint data/nnUNet_results/Dataset509_ARCADE_1x1_4c/nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_upsample_perlayer_12_dense_relu_nearest_upsample_wm_joint_alpha1.0_perlayer_candidatebits468_forcedsp_lut50_bram50_dsp90_ft15ep__nnUNetPlans__2d/fold_0/checkpoint_best.pth.txt
Then: docker cp <onnx> <container>:/home/thelegendiv/finn/notebooks/enet/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "hardware"))

import brevitas.nn as qnn  # noqa: E402
import finn_enet_prod_export  # noqa: E402
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402

FALLBACK_BITS = 6
PRESETS = {
    512: dict(decoder="nearest_upsample", context="dense_dilation", channels=(4, 16, 32, 16, 4),
              dataset="Dataset509_ARCADE_1x1_4c", name_stem="quantEnet_12_dense_relu_nearest_upsample"),
    256: dict(decoder="nearest_conv_upsample", context="dense_dilation_half", channels=(4, 16, 32, 16, 4),
              dataset="Dataset510_ARCADE_256_4c", name_stem="quantEnet_12_dense_relu_nearest_conv_upsample"),
}
BOTTLENECKS = (4, 8, 8, 2, 1)
CONV_ORDER_MODULE_TYPES = (qnn.QuantConv2d, qnn.QuantConvTranspose2d, nn.MaxPool2d)


def load_layer_bits(bits_file: Path, weight_names, act_names, fallback: int = FALLBACK_BITS):
    with open(bits_file) as f:
        raw = json.load(f)
    raw_w, raw_a = raw["layer_weight_bits"], raw["layer_act_bits"]
    missing = [n for n in weight_names if n not in raw_w] + [n for n in act_names if n not in raw_a]
    if missing:
        print(f"  WARNING: {len(missing)} site(s) missing from {bits_file.name}, using fallback={fallback}: {missing}")
    return ({n: raw_w.get(n, fallback) for n in weight_names},
            {n: raw_a.get(n, fallback) for n in act_names})


def dump_conv_order(model: nn.Module, in_channels: int, out_path: Path) -> None:
    """Forward-execution order of every weight-bearing/pool module (64x64 pass)."""
    name_by_id, to_hook = {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, CONV_ORDER_MODULE_TYPES) and id(mod) not in name_by_id:
            name_by_id[id(mod)] = name
            to_hook.append(mod)
    ordered = []

    def _record(mod, _inp, _out):
        shape = list(mod.weight.shape) if getattr(mod, "weight", None) is not None else None
        ordered.append({"logical_name": name_by_id[id(mod)], "module_type": type(mod).__name__, "weight_shape": shape})

    handles = [m.register_forward_hook(_record) for m in to_hook]
    with torch.no_grad():
        model(torch.randn(1, in_channels, 64, 64))
    for h in handles:
        h.remove()
    with open(out_path, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"  conv order: {len(ordered)} module calls -> {out_path}")


# ------------------------------------------------------- optional calibration / cross-test
def _pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    h, w = x.shape[-2:]
    pad_h, pad_w = (-h) % multiple, (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def calibrate(model, preprocessed_dir: Path, n: int | None) -> None:
    from brevitas.graph.calibrate import calibration_mode
    paths = sorted(preprocessed_dir.glob("train_*_p0000.npy"))
    if not paths:
        raise FileNotFoundError(f"no train_*_p0000.npy patches under {preprocessed_dir}")
    paths = paths if n is None else paths[:n]
    with torch.no_grad(), calibration_mode(model):
        for p in paths:
            model(_pad_to_multiple(torch.from_numpy(np.load(p)).float()))
    print(f"  calibrated runtime-stats scales over {len(paths)} training patches")


def _zscore(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    return (arr - arr.mean()) / max(arr.std(), 1e-8)


def cross_test(model, dataset: str, real_pred_dir: Path | None) -> None:
    """FINN-mirror Dice on imagesTs/labelsTs (+ agreement vs. the real model's predictions)."""
    from PIL import Image
    sys.path.insert(0, str(REPO_ROOT / "analysis" / "501_ARCADE"))
    import segmentation_topology as topo

    raw = REPO_ROOT / "data" / "nnUNet_raw" / dataset
    with open(raw / "dataset.json") as f:
        classes = sorted((cid, name) for name, cid in json.load(f)["labels"].items() if cid != 0)
    preds = {}
    with torch.no_grad():
        for path in sorted((raw / "imagesTs").glob("*_0000.png")):
            x = torch.from_numpy(_zscore(np.asarray(Image.open(path)))).float()[None, None]
            preds[path.stem.rsplit("_", 1)[0]] = model(x).argmax(dim=1)[0].to(torch.uint8).numpy()

    def _dice_vs(ref_dir: Path, label: str):
        per_class, binary, agree, total = {n: [] for _, n in classes}, [], 0, 0
        for path in sorted(ref_dir.glob("*.png")):
            if path.stem not in preds:
                continue
            ref, pred = np.asarray(Image.open(path)).astype(np.uint8), preds[path.stem]
            for cid, n in classes:
                per_class[n].append(topo.dice_score(ref == cid, pred == cid))
            binary.append(topo.dice_score(ref > 0, pred > 0))
            agree, total = agree + int((ref == pred).sum()), total + ref.size
        means = {n: sum(v) / len(v) for n, v in per_class.items()}
        print(f"\n{label} ({len(binary)} cases): mean dice={sum(means.values()) / len(means):.4f} "
              f"binary={sum(binary) / len(binary):.4f} pixel_agree={agree / total:.4f}")
        for n, v in means.items():
            print(f"  dice_{n}: {v:.4f}")

    _dice_vs(raw / "labelsTs", "FINN mirror vs. ground truth")
    if real_pred_dir is not None and real_pred_dir.exists():
        _dice_vs(real_pred_dir, "FINN mirror vs. real model predictions")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolution", type=int, choices=sorted(PRESETS), required=True)
    p.add_argument("--tag", required=True, help="goes into the onnx name")
    p.add_argument("--bits-file", required=True, help="MILP layer_bits_SITES_*.json")
    p.add_argument("--checkpoint", required=True, help="nnU-Net checkpoint_best.pth[.txt]")
    p.add_argument("--name", help="override onnx basename (default <stem>_{trained|dummy}_<tag>_<R>x<R>)")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "hardware" / "outputs"))
    p.add_argument("--decoder")
    p.add_argument("--context-pattern")
    p.add_argument("--channels", type=int, nargs=5)
    p.add_argument("--bottlenecks", type=int, nargs=5, default=BOTTLENECKS)
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--out-channels", type=int, default=5)
    p.add_argument("--calibrate", type=int, metavar="N", default=None,
                   help="runtime-stats calibration over N preprocessed train patches (-1 = all)")
    p.add_argument("--cross-test", action="store_true")
    p.add_argument("--real-pred-dir", help="labelsPr_* dir of the real model, for --cross-test agreement")
    args = p.parse_args()

    preset = PRESETS[args.resolution]
    decoder = args.decoder or preset["decoder"]
    context = args.context_pattern or preset["context"]
    channels = tuple(args.channels or preset["channels"])
    bottlenecks = tuple(args.bottlenecks)
    res = args.resolution
    name = args.name or f"{preset['name_stem']}_trained_{args.tag}_{res}x{res}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    finn_enet_prod_export.OUT_DIR = out_dir

    weight_names, act_names = layer_names_for(
        out_channels=args.out_channels, channels=channels, bottlenecks_per_stage=bottlenecks,
        context_pattern=context, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
        decoder_type=decoder,
    )
    w_bits, a_bits = load_layer_bits(Path(args.bits_file), weight_names, act_names)
    arch = dict(in_channels=args.in_channels, out_channels=args.out_channels, channels=channels,
                bottlenecks_per_stage=bottlenecks, context_pattern=context, decoder_type=decoder)
    print(f"=== {name}: decoder={decoder} context={context} channels={channels} "
          f"{len(weight_names)} weight / {len(act_names)} act sites ===")
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        p.error(f"checkpoint not found: {ckpt}")
    model = LayerQuantEnetFINN.from_pretrained(ckpt, w_bits, a_bits, **arch).eval()

    if args.calibrate is not None:
        pre = REPO_ROOT / "data" / "nnUNet_preprocessed" / preset["dataset"] / "nnUNetPlans_2d"
        calibrate(model, pre, None if args.calibrate < 0 else args.calibrate)

    dump_conv_order(model, args.in_channels, out_dir / f"{name}_conv_order.json")

    dummy = torch.rand(1, args.in_channels, res, res) * 2 - 1
    with torch.no_grad():
        out = model(dummy)
    assert tuple(out.shape[1:]) == (args.out_channels, res, res), f"unexpected output shape {tuple(out.shape)}"
    finn_enet_prod_export.export_model(model, name, dummy)
    print(f"Exported {out_dir / (name + '.onnx')}")

    if args.cross_test:
        cross_test(model, preset["dataset"], Path(args.real_pred_dir) if args.real_pred_dir else None)


if __name__ == "__main__":
    main()

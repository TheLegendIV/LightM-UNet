"""Post-export verification gate for LayerQuantEnetFINN hardware builds.

Run after EVERY export (the export scripts call build_reference()/check_onnx() and abort on failure).

Hardware input contract (LayerQuantEnetFINN(uint8_input=True)):
    z = per-image z-score of the raw 256x256 image
    u = clip(round(z / s_in) + 128, 0, 255)            # exact INT6 level + 128, UINT8
    class_id = argmax over the 5 logits                 # UINT8 output

Stages, each gated (non-zero exit / AssertionError on failure):
  G0  PyTorch (uint8 contract) is not degenerate and reaches the minimum Dice vs ground truth.
      Ground truth = preprocessed val split (z-scored val_*_p0000.npy + gt_segmentations/val_*.png);
      the held-out test labels (imagesTs/labelsTs) are not required.
  G1  PyTorch(uint8 contract, u) == PyTorch(float z) pixel-for-pixel (the encoding is lossless).
  G2  every exported ONNX stage (QONNX export, step_enet_streamline, ...) executed with qonnx on the
      SAME u inputs agrees with PyTorch and keeps the Dice.
  G3  first MultiThreshold of a streamlined ONNX has integer thresholds 97..159 (u domain).

build_reference() writes <ref>.npz (u inputs, PyTorch preds, ground truth) so the later stages can run
without torch/brevitas, e.g. inside the FINN container against the preamble's intermediate models:

    python hardware/verify_export.py --ref <ref>.npz \\
        --onnx qonnx=<name>.onnx streamline=<preamble>/intermediate_models/step_enet_streamline.onnx \\
        --first-thresholds <preamble>/intermediate_models/step_enet_streamline.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

NUM_CLASSES = 5


# ------------------------------------------------------------------ metrics
def dice_report(preds: np.ndarray, gts: np.ndarray, num_classes: int = NUM_CLASSES) -> dict:
    """preds/gts: (N,H,W) class ids. Per-class mean of per-case Dice over cases where the class is in
    the ground truth or the prediction; fg_dice = mean over foreground classes; binary = fg-vs-bg."""
    def dice(a: np.ndarray, b: np.ndarray) -> float | None:
        s = int(a.sum()) + int(b.sum())
        return None if s == 0 else 2.0 * int((a & b).sum()) / s

    out: dict[str, float] = {}
    for c in range(1, num_classes):
        vals = [d for p, g in zip(preds, gts) if (d := dice(p == c, g == c)) is not None]
        out[f"dice_c{c}"] = float(np.mean(vals)) if vals else float("nan")
    out["fg_dice"] = float(np.nanmean([out[f"dice_c{c}"] for c in range(1, num_classes)]))
    vals = [d for p, g in zip(preds, gts) if (d := dice(p > 0, g > 0)) is not None]
    out["binary_dice"] = float(np.mean(vals)) if vals else float("nan")
    out["max_class_fraction"] = float(max(np.bincount(preds.ravel(), minlength=num_classes)) / preds.size)
    return out


def _fmt(report: dict) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in report.items())


# ------------------------------------------------------------------ data
def load_val_cases(pre_dir: Path, n: int | None, hw: int = 256) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(names, z (N,1,H,W) float32 per-image z-scored, gt (N,H,W) uint8) from a nnU-Net preprocessed dir
    (<pre_dir>/nnUNetPlans_2d/val_*_p0000.npy and <pre_dir>/gt_segmentations/val_*_p0000.png)."""
    from PIL import Image

    if pre_dir.name == "nnUNetPlans_2d":
        pre_dir = pre_dir.parent
    paths = sorted((pre_dir / "nnUNetPlans_2d").glob("val_*_p0000.npy"), key=lambda p: int(p.stem.split("_")[1]))
    paths = [p for p in paths if not p.stem.endswith("_seg")]
    # nnU-Net's nonzero-crop shrinks some cases (e.g. 256x254); the hardware always sees the full
    # 256x256 frame, so only uncropped cases are valid hardware inputs / match the 256x256 GT PNGs.
    paths = [p for p in paths if np.load(p, mmap_mode="r").shape[-2:] == (hw, hw)]
    if not paths:
        raise FileNotFoundError(f"no val_*_p0000.npy under {pre_dir / 'nnUNetPlans_2d'}")
    paths = paths if n is None else paths[:n]
    names, zs, gts = [], [], []
    for p in paths:
        arr = np.load(p).astype(np.float32)
        z = arr.reshape(1, *arr.shape[-2:])
        names.append(p.stem)
        zs.append(z)
        gts.append(np.asarray(Image.open(pre_dir / "gt_segmentations" / f"{p.stem}.png")).astype(np.uint8))
    return names, np.stack(zs), np.stack(gts)


# ------------------------------------------------------------------ G0 / G1: PyTorch reference
def build_reference(
    model, pre_dir: Path, out_npz: Path, *, n_cases: int | None = 200, n_onnx_cases: int = 10,
    min_fg_dice: float = 0.0, min_agreement: float = 0.9999, hw: int = 256,
) -> dict:
    """G0+G1 on a LayerQuantEnetFINN(uint8_input=True) with trained scales bound. Writes out_npz."""
    import torch

    assert getattr(model, "uint8_input", False), "model must be built with uint8_input=True"
    model.eval()
    s_in = model.bind_input_scale()
    assert abs(s_in - 1.0) > 1e-6, "input_quant scale is the 1.0 default -- trained act scales were not loaded"
    names, z, gt = load_val_cases(pre_dir, n_cases, hw)
    zt = torch.from_numpy(z)
    u = model.encode_uint8(zt, s_in)

    def run(x, uint8: bool) -> np.ndarray:
        model.uint8_input = uint8
        preds = []
        with torch.no_grad():
            for i in range(len(x)):
                preds.append(model(x[i:i + 1]).argmax(dim=1)[0].to(torch.uint8).numpy())
        model.uint8_input = True
        return np.stack(preds)

    pred_u8 = run(u, True)
    pred_float = run(zt, False)

    rep_u8, rep_float = dice_report(pred_u8, gt), dice_report(pred_float, gt)
    agree = float((pred_u8 == pred_float).mean())
    print(f"[G0] PyTorch uint8-contract vs GT ({len(names)} val cases): {_fmt(rep_u8)}")
    print(f"[G0] PyTorch float-z       vs GT                     : {_fmt(rep_float)}")
    print(f"[G1] uint8 vs float pixel agreement: {agree:.6f}")

    np.savez_compressed(
        out_npz, names=np.array(names), u=u.numpy().astype(np.uint8)[:, 0], z=z[:, 0],
        pred=pred_u8, gt=gt, input_scale=np.float64(s_in), n_onnx=np.int64(n_onnx_cases),
    )
    print(f"  reference written: {out_npz}")

    assert rep_u8["max_class_fraction"] < 0.999, (
        f"degenerate network: one class covers {rep_u8['max_class_fraction']:.4f} of all pixels")
    assert rep_u8["fg_dice"] >= min_fg_dice, f"fg Dice {rep_u8['fg_dice']:.4f} < required {min_fg_dice}"
    assert agree >= min_agreement, f"uint8 encoding changes predictions: agreement {agree:.6f} < {min_agreement}"
    return rep_u8


# ------------------------------------------------------------------ G2: ONNX stages
def _to_class_map(out: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.squeeze(out, axis=0) if out.ndim == 4 else out
    if out.ndim == 3 and out.shape[0] == num_classes:
        return out.argmax(axis=0).astype(np.uint8)
    if out.ndim == 3 and out.shape[-1] == num_classes:
        return out.argmax(axis=-1).astype(np.uint8)
    return np.squeeze(out).astype(np.uint8)  # LabelSelect output: class ids already


def check_onnx(
    ref_npz: Path, onnx_path: Path, label: str, *, n_cases: int | None = None,
    min_agreement: float = 0.995, max_dice_drop: float = 0.01,
) -> dict:
    """G2: execute onnx_path with qonnx on the reference's u inputs; compare with the PyTorch preds."""
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    from qonnx.transformation.infer_shapes import InferShapes

    ref = np.load(ref_npz)
    n = int(ref["n_onnx"]) if n_cases is None else n_cases
    u, pred, gt = ref["u"][:n], ref["pred"][:n], ref["gt"][:n]
    model = ModelWrapper(str(onnx_path)).transform(InferShapes())
    for node in model.graph.node:  # optional Resize inputs (roi) are '' -> qonnx rejects them as unshaped
        for i, t in enumerate(node.input):
            if t == "":
                node.input[i] = f"{node.name}_empty{i}"
                model.set_initializer(node.input[i], np.zeros((0,), dtype=np.float32))
    model.check_all_tensor_shapes_specified(fix_missing_init_shape=True)
    in_name, out_name = model.graph.input[0].name, model.graph.output[0].name
    in_shape = model.get_tensor_shape(in_name)
    # the QONNX export forces INT8 on the (float) logit output; qonnx's integer-rounding sanity check would reject it
    from qonnx.core.datatype import DataType
    if model.get_tensor_datatype(out_name) not in (DataType["FLOAT32"],) and not model.get_nodes_by_op_type("LabelSelect"):
        model.set_tensor_datatype(out_name, DataType["FLOAT32"])
    got = []
    for i in range(len(u)):
        x = u[i].astype(np.float32).reshape(in_shape)
        got.append(_to_class_map(execute_onnx(model, {in_name: x})[out_name], NUM_CLASSES))
    got = np.stack(got)
    agree = float((got == pred).mean())
    rep_hw, rep_pt = dice_report(got, gt), dice_report(pred, gt)
    print(f"[G2:{label}] {len(u)} cases  agreement vs PyTorch={agree:.6f}")
    print(f"[G2:{label}]   onnx : {_fmt(rep_hw)}")
    print(f"[G2:{label}]   torch: {_fmt(rep_pt)}")
    assert agree >= min_agreement, f"[{label}] pixel agreement {agree:.6f} < {min_agreement}"
    drop = rep_pt["fg_dice"] - rep_hw["fg_dice"]
    assert drop <= max_dice_drop, f"[{label}] fg Dice drop {drop:.4f} > {max_dice_drop}"
    return rep_hw


# ------------------------------------------------------------------ G3: first-threshold structure
def check_first_thresholds(onnx_path: Path) -> None:
    """The first MultiThreshold of a streamlined model must hold integer u-domain thresholds 97..159."""
    from qonnx.core.modelwrapper import ModelWrapper

    model = ModelWrapper(str(onnx_path))
    mt = model.get_nodes_by_op_type("MultiThreshold")[0]
    t = model.get_initializer(mt.input[1])
    expected = np.arange(97, 160, dtype=np.float64)
    assert t.shape[1] == 63, f"expected 63 thresholds per channel, got {t.shape}"
    assert np.allclose(t, expected[None, :], atol=1e-3), (
        f"[G3] first MultiThreshold thresholds are not 97..159: min={t.min()} max={t.max()} "
        f"(input contract not absorbed into the thresholds)")
    print(f"[G3] {Path(onnx_path).name}: first MultiThreshold thresholds == 97..159")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="reference .npz written by build_reference()")
    p.add_argument("--onnx", nargs="*", default=[], metavar="LABEL=PATH")
    p.add_argument("--first-thresholds", nargs="*", default=[], metavar="PATH")
    p.add_argument("--n", type=int, default=None, help="cases per ONNX (default: reference's n_onnx)")
    p.add_argument("--min-agreement", type=float, default=0.995)
    p.add_argument("--max-dice-drop", type=float, default=0.01)
    args = p.parse_args()
    for spec in args.onnx:
        label, _, path = spec.partition("=")
        check_onnx(Path(args.ref), Path(path), label, n_cases=args.n,
                   min_agreement=args.min_agreement, max_dice_drop=args.max_dice_drop)
    for path in args.first_thresholds:
        check_first_thresholds(Path(path))
    print("VERIFY OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"VERIFY FAILED: {e}", file=sys.stderr)
        sys.exit(1)

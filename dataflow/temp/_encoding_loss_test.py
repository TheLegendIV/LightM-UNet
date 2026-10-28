"""How much does each input encoding change the net's prediction vs. the exact per-image z-score reference?"""
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "hardware" / "builds" / "S12_dense_256_u4_analytical_v1"))
sys.path.insert(0, str(REPO_ROOT / "hardware" / "builds" / "12_dense_relu_nearest_conv_upsample_256"))
import finn_export_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep as ft  # noqa: E402
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402

kw = dict(out_channels=5, channels=ft.CHANNELS, bottlenecks_per_stage=ft.BOTTLENECKS_PER_STAGE,
          context_pattern=ft.CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
          use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
          decoder_type=ft.DECODER_TYPE)
wn, an = layer_names_for(**kw)
wb, ab = ft.load_layer_bits(Path(ft.DEFAULT_BITS_FILE), wn, an)
model = LayerQuantEnetFINN.from_pretrained(
    ft.DEFAULT_CHECKPOINT, wb, ab, in_channels=1, out_channels=5, channels=ft.CHANNELS,
    bottlenecks_per_stage=ft.BOTTLENECKS_PER_STAGE, context_pattern=ft.CONTEXT_PATTERN, decoder_type=ft.DECODER_TYPE).eval()
s = float(model.initial.input_quant(torch.zeros(1, 1, 8, 8)).scale)
T = (np.arange(63) - 31.5) * s  # level boundaries in z units


def levels_from_thresholds(x, thr):
    return np.searchsorted(thr, x, side="right") - 32  # -32 + #{thr <= x}


def predict(levels):
    with torch.no_grad():
        return model(torch.from_numpy((levels * s).astype(np.float32))[None, None]).argmax(1)[0].numpy()


imgs = {Path(p).stem: np.asarray(Image.open(p)).astype(np.float32) for p in sorted(glob.glob(str(REPO_ROOT / "deployment" / "256_test_images" / "*.png")))}
mus = np.array([a.mean() for a in imgs.values()])
sds = np.array([a.std() for a in imgs.values()])
print("input scale", round(s, 4), "| per-image mean range", mus.min().round(1), mus.max().round(1), "| std range", sds.min().round(1), sds.max().round(1))
print("one INT6 step in gray levels (per-image std * scale):", (sds * s).round(1).tolist())
gmu, gsd = float(mus.mean()), float(sds.mean())
print("global mean/std", round(gmu, 1), round(gsd, 1))

def fg_dice(a, b):
    a, b = a > 0, b > 0
    return 2 * float((a & b).sum()) / max(float(a.sum() + b.sum()), 1.0)


tot = {k: [] for k in ("fine32", "fine16", "global")}
for name, a in imgs.items():
    z = (a - a.mean()) / a.std()
    ref_lv = levels_from_thresholds(z, T)
    ref = predict(ref_lv)
    row = [name, f"ref_fg_px={int((ref > 0).sum())}"]
    for tag, mult in (("fine32", 32), ("fine16", 16)):
        u = np.clip(np.round(z * mult) + 128, 0, 255)
        thr = np.ceil(128 + mult * T)
        lv = levels_from_thresholds(u, thr)
        d = fg_dice(predict(lv), ref)
        tot[tag].append(d)
        row.append(f"{tag}: lvl_diff={float((lv != ref_lv).mean()):.3f} fgDice={d:.3f}")
    thr_g = np.ceil(gmu + gsd * T)  # raw uint8 pixel thresholds, one global mean/std
    lv_g = levels_from_thresholds(a, thr_g)
    d = fg_dice(predict(lv_g), ref)
    tot["global"].append(d)
    row.append(f"global: lvl_diff={float((lv_g != ref_lv).mean()):.3f} fgDice={d:.3f}")
    print(" | ".join(row))
print("mean fg Dice vs exact reference:", {k: round(float(np.mean(v)), 3) for k, v in tot.items()})

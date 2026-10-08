"""PyTorch reference (LayerQuantEnetFINN ft15ep checkpoint) on deployment/256_test_images.

Writes per image: input u8 encodings for the board and the reference class map.
Run: .venv\\Scripts\\python.exe hardware/temp/_ref_infer_256_test_images.py
"""
import glob
import os
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

IN_DIR = REPO_ROOT / "deployment" / "256_test_images"
OUT_DIR = REPO_ROOT / "deployment" / "image_transfer_interface" / "reference_256"
OUT_DIR.mkdir(parents=True, exist_ok=True)

shape_kwargs = dict(
    out_channels=5, channels=ft.CHANNELS, bottlenecks_per_stage=ft.BOTTLENECKS_PER_STAGE,
    context_pattern=ft.CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
    use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
    decoder_type=ft.DECODER_TYPE,
)
wn, an = layer_names_for(**shape_kwargs)
wb, ab = ft.load_layer_bits(Path(ft.DEFAULT_BITS_FILE), wn, an)
model = LayerQuantEnetFINN.from_pretrained(
    ft.DEFAULT_CHECKPOINT, wb, ab, in_channels=1, out_channels=5, channels=ft.CHANNELS,
    bottlenecks_per_stage=ft.BOTTLENECKS_PER_STAGE, context_pattern=ft.CONTEXT_PATTERN, decoder_type=ft.DECODER_TYPE,
).eval()
_q = model.initial.input_quant(torch.zeros(1, 1, 8, 8))
print("input_quant scale", float(_q.scale), "zero_point", float(_q.zero_point), "bits", float(_q.bit_width))

for p in sorted(glob.glob(str(IN_DIR / "*.png"))):
    stem = Path(p).stem
    a = np.asarray(Image.open(p)).astype(np.float32)
    z = (a - a.mean()) / max(a.std(), 1e-8)
    with torch.no_grad():
        pred = model(torch.from_numpy(z)[None, None]).argmax(dim=1)[0].numpy().astype(np.uint8)
    # board encoding: INT6 level of the model's own input quantizer, offset by 128 (UINT8 zero-point 128)
    with torch.no_grad():
        q = model.initial.input_quant(torch.from_numpy(z)[None, None]).int()[0, 0].numpy()
    u = (q.astype(np.int32) + 128).astype(np.uint8)
    u.tofile(OUT_DIR / f"{stem}_input_u8.raw")
    pred.tofile(OUT_DIR / f"{stem}_ref_pred.raw")
    cnt = {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))}
    print(stem, "z min/max", round(float(z.min()), 2), round(float(z.max()), 2), "ref classes", cnt)

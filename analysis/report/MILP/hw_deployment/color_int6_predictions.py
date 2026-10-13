"""Colour the INT6 QAT net's predictions (LayerQuantEnetFINN, wm uniform INT6, 15 ep) for the first 10 imagesTs cases
(test_1 .. test_10, numeric order; the set has no test_0). Prediction only: one RGB PNG per case at native resolution, written next to this script.
  python analysis/report/MILP/hw_deployment/color_int6_predictions.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[4]
PRED = REPO / "data" / "nnUNet_raw" / "Dataset509_ARCADE_1x1_4c" / (
    "labelsPr_nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_upsample_perlayer_"
    "12_dense_relu_nearest_upsample_wm_uniform_int6_samefolding_perlayer_lut50_bram50_dsp90_ft15ep")
OUT = Path(__file__).resolve().parent / "int6_pred_color"

# background, LAD, RCA, LCX, LM
PALETTE = np.array([(0, 0, 0), (230, 25, 75), (60, 180, 75), (67, 99, 216), (255, 225, 25)], dtype=np.uint8)

OUT.mkdir(exist_ok=True)
for i in range(1, 11):
    pred = np.array(Image.open(PRED / f"test_{i}_p0000.png"))
    Image.fromarray(PALETTE[pred]).save(OUT / f"test_{i:03d}.png")
print(f"10 cases -> {OUT}")

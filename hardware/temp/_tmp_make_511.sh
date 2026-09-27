#!/bin/bash
set -e
export nnUNet_raw=/workspace/LightM-UNet/data/nnUNet_raw
export nnUNet_preprocessed=/workspace/LightM-UNet/data/nnUNet_preprocessed
export nnUNet_results=/workspace/LightM-UNet/data/nnUNet_results
cd /workspace/LightM-UNet/enet
python3 nnunetv2/dataset_conversion/Dataset511_ARCADE_1x1_3c.py
D=$nnUNet_raw/Dataset511_ARCADE_1x1_3c
for s in imagesTr imagesTs labelsTr labelsTs; do echo "$s $(ls $D/$s | wc -l)"; done
python3 - <<'EOF'
import numpy as np, glob
from PIL import Image
u = set()
for p in glob.glob("/workspace/LightM-UNet/data/nnUNet_raw/Dataset511_ARCADE_1x1_3c/labelsTr/*.png"):
    u |= set(np.unique(np.array(Image.open(p))).tolist())
print("labelsTr unique:", sorted(u))
EOF

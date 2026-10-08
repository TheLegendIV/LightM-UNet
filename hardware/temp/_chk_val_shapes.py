from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

d = Path(r"C:\DEV\repos\LightM-UNet\data\nnUNet_preprocessed\Dataset510_ARCADE_256_4c")
zs, gs = Counter(), Counter()
for p in sorted((d / "nnUNetPlans_2d").glob("val_*_p0000.npy")):
    zs[np.load(p, mmap_mode="r").shape] += 1
    gs[Image.open(d / "gt_segmentations" / f"{p.stem}.png").size] += 1
print(zs, gs)

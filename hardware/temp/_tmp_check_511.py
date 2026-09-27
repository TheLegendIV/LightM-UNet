import glob
import numpy as np
from PIL import Image

root = "/workspace/LightM-UNet/data/nnUNet_raw/Dataset511_ARCADE_1x1_3c"
vals = set()
for f in glob.glob(f"{root}/labelsT*/*.png"):
    vals |= set(np.unique(np.array(Image.open(f))).tolist())
print("label values:", sorted(vals))

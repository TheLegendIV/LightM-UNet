from pathlib import Path
p = Path("/workspace/LightM-UNet/hardware/builds/12_dense_relu_nearest_upsample_512/finn_export_12_dense_relu_nearest_upsample_dummy.py").resolve()
print("x1", p.parent)
print("x2", p.parent.parent)
print("x3", p.parent.parent.parent)

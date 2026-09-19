import os
from qonnx.core.modelwrapper import ModelWrapper

OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_20260918_144349"
part_dir = os.path.join(OUTPUT_DIR, "intermediate_models", "supported_op_partitions")

for i in range(8):
    p = os.path.join(part_dir, f"partition_{i}.onnx")
    if not os.path.exists(p):
        print(f"partition_{i}: MISSING onnx at {p}")
        continue
    m = ModelWrapper(p)
    stitch_proj = m.get_metadata_prop("vivado_stitch_proj")
    exists = os.path.exists(stitch_proj) if stitch_proj else None
    print(f"partition_{i}: vivado_stitch_proj={stitch_proj} exists_on_disk={exists}")

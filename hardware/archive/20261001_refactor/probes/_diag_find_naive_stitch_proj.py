import glob
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")
from qonnx.core.modelwrapper import ModelWrapper

for t in ["p0", "p1"]:
    print("===", t, "===")
    files = glob.glob(
        f"/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/tiny_2part_collision_test_{t}/intermediate_models/*.onnx"
    )
    print(files)
    # pick the last-stage model (post stitched-ip) if present
    cand = [f for f in files if "step_create_stitched_ip" in f or "stitched" in f.lower()]
    if not cand:
        cand = sorted(files)
    if cand:
        m = ModelWrapper(cand[-1])
        proj = m.get_metadata_prop("vivado_stitch_proj")
        print("  model:", cand[-1])
        print("  vivado_stitch_proj:", proj)
        if proj and os.path.isdir(proj):
            xpr = glob.glob(os.path.join(proj, "*.xpr"))
            print("  xpr:", xpr)
            bd_root = os.path.join(proj, "finn_vivado_stitch_proj.srcs", "sources_1", "bd")
            if os.path.isdir(bd_root):
                print("  bd dirs:", os.listdir(bd_root))

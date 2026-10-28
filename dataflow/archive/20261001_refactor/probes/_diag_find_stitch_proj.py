import sys
sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")
from qonnx.core.modelwrapper import ModelWrapper
import os

for tag in ["nofix_p0", "fix_p0"]:
    fn = f"finn_deployment_outputs/tiny_2part_prefix_test_{tag}/final_partition_model.onnx"
    m = ModelWrapper(fn)
    proj = m.get_metadata_prop("vivado_stitch_proj")
    print(tag, "->", proj)
    if proj and os.path.isdir(proj):
        ip_dir = os.path.join(proj, "ip")
        if os.path.isdir(ip_dir):
            print("  top ip/ contents:", sorted(os.listdir(ip_dir)))
        repo_paths_file = os.path.join(proj, "all_verilog_srcs.txt")
        # look for child hls ip catalog dirs anywhere under proj
        for root, dirs, files in os.walk(proj):
            for d in dirs:
                if "component.xml" in os.listdir(os.path.join(root, d)) if os.path.isdir(os.path.join(root, d)) else False:
                    pass
        # simpler: find all component.xml files
        matches = []
        for root, dirs, files in os.walk(proj):
            if "component.xml" in files:
                matches.append(root)
        print("  component.xml dirs:")
        for mm in sorted(matches):
            print("   ", mm)

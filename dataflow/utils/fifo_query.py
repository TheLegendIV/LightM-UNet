import sys, os, glob
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v2_20260918"
parent_model = ModelWrapper(os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent_built.onnx"))
node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[2]
model_path = getCustomOp(node).get_nodeattr("model")
part_model = ModelWrapper(model_path)
proj = part_model.get_metadata_prop("vivado_stitch_proj")
print("model_path:", model_path)
print("vivado_stitch_proj:", proj)
print("wrapper_filename:", part_model.get_metadata_prop("wrapper_filename"))
print("deep_fifos:")
for n in part_model.get_nodes_by_op_type("StreamingFIFO_rtl"):
    inst = getCustomOp(n)
    depth = inst.get_nodeattr("depth")
    if depth > 64:
        print(n.name, "depth=", depth, "ram_style=", inst.get_nodeattr("ram_style"))
files = sorted(glob.glob(os.path.join(proj, "**", "*StreamingFIFO_rtl*"), recursive=True))
print("---FIFO_FILES---")
for f in files[:50]: print(f)
vfiles = [f for f in files if f.lower().endswith(".v")]
style = []
for f in vfiles:
    try:
        text = open(f, encoding="utf-8", errors="replace").read()
    except OSError: continue
    if "ram_style" in text: style.append(f)
print("---RAM_STYLE_FILES---")
for f in style: print(f)
if style:
    f=style[0]
    print("---SNIPPET_FILE---")
    print(f)
    lines=open(f, encoding="utf-8", errors="replace").read().splitlines()
    for i,line in enumerate(lines):
        if "ram_style" in line:
            print("---SNIPPET---")
            for x in lines[max(0,i-3):i+4]: print(x)
            break

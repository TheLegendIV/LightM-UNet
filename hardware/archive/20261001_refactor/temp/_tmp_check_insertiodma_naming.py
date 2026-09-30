import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA

OUT = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_nearest_conv_upsample_256_w8_16_v4_trained_rtl_mvau_8way_full_256x256_20260928_043156"
fn0 = OUT + "/intermediate_models/supported_op_partitions/partition_0.onnx"
m = ModelWrapper(fn0)
orig_names = [n.name for n in m.graph.node]
m = m.transform(InsertIODMA(max_intfwidth=32, insert_input=True, insert_output=False, insert_extmemw=False))
new_names = [n.name for n in m.graph.node]
dupes = {n for n in new_names if new_names.count(n) > 1}
added = [n for n in new_names if n not in orig_names]
missing = [n for n in orig_names if n not in new_names]
print("dupes (should be empty):", dupes)
print("added (should be exactly 1 new IODMA node):", added)
print("missing pre-existing names (should be empty -- confirms prefixes preserved WITHOUT GiveUniqueNodeNames):", missing)

import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper

m = ModelWrapper(
    "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
    "12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_20260918_144349/"
    "intermediate_models/supported_op_partitions/partition_0.onnx"
)
print("vivado_stitch_proj=", m.get_metadata_prop("vivado_stitch_proj"))
print("wrapper_filename=", m.get_metadata_prop("wrapper_filename"))

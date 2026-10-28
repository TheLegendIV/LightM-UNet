import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from finn.util.basic import make_build_dir
from finn.util.fpgadataflow import is_fpgadataflow_node, is_hls_node
from qonnx.custom_op.registry import getCustomOp
print("imports OK")
print("is_hls_node callable:", callable(is_hls_node))
print("make_build_dir callable:", callable(make_build_dir))

from qonnx.core.modelwrapper import ModelWrapper

model = ModelWrapper('/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/partition6_autosize_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_autosize1/partition6_autosize_postfifo.onnx')
for i, node in enumerate(model.graph.node):
    marker = ""
    if node.op_type in ("DuplicateStreams_hls", "AddStreams_hls"):
        marker += "  **FORK/JOIN**"
    if "MVAU_rtl_8" == node.name:
        marker += "  <<<< MVAU_rtl_8"
    print(f"{i:3d} {node.op_type:35s} {node.name}{marker}")

from qonnx.core.modelwrapper import ModelWrapper

a = ModelWrapper('/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/partition6_refix_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_refix1/partition6_refix_stitched.onnx')
b = ModelWrapper('/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/partition6_autosize_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_autosize1/partition6_autosize_postfifo.onnx')

for label, m in (("analytical", a), ("autosize", b)):
    n_fifo = sum(1 for n in m.graph.node if "StreamingFIFO" in n.op_type)
    n_total = len(m.graph.node)
    print(f"{label}: total_nodes={n_total} fifo_nodes={n_fifo}")

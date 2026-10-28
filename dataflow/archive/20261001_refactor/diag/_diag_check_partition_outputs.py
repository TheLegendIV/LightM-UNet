from qonnx.core.modelwrapper import ModelWrapper

BASE = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_8way_full_20260913_095107/intermediate_models/supported_op_partitions"

for i in range(8):
    fn = f"{BASE}/partition_{i}.onnx"
    m = ModelWrapper(fn)
    outs = [o.name for o in m.graph.output]
    n_nodes = len(m.graph.node)
    producers = [m.find_producer(o) for o in outs]
    print(i, "nodes=", n_nodes, "outputs=", outs, "producer_found=", [p is not None for p in producers])

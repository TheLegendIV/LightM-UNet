from qonnx.core.modelwrapper import ModelWrapper

m = ModelWrapper(
    "finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_preamble_20260913_091715/"
    "intermediate_models/step_enet_convert_to_hw.onnx"
)
ds = m.get_nodes_by_op_type("DuplicateStreams")
ad = m.get_nodes_by_op_type("AddStreams")
print("DuplicateStreams:", len(ds), "AddStreams:", len(ad))
nodes = list(m.graph.node)


def idx(n):
    return nodes.index(n)


for n in sorted(ds, key=idx):
    print("DS", idx(n), n.name)
for n in sorted(ad, key=idx):
    print("AD", idx(n), n.name)

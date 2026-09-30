from qonnx.core.modelwrapper import ModelWrapper

OUTDIR = (
    "finn_deployment_outputs/"
    "12_dense_relu_warmstart150ep_alpha025_trained_8way_full_20260913_095107"
)

for i in range(8):
    fn = f"{OUTDIR}/intermediate_models/supported_op_partitions/partition_{i}.onnx"
    m = ModelWrapper(fn)
    nodes = list(m.graph.node)
    last = nodes[-1]
    outs = [(o.name, m.find_producer(o.name).name if m.find_producer(o.name) else None) for o in m.graph.output]
    print(f"partition {i}: n_nodes={len(nodes)} last_node={last.name}({last.op_type}) outputs={outs}")

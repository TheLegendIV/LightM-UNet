import onnx
import sys

PARTITION1_PREFIFO = (
    r"C:\DEV\repos\LightM-UNet\hardware\builds\S12_dense_256_u4_analytical_v1\outputs\int6_fps250_lat200_milpfold_8way_20261005_235400"
    r"\intermediate_models_listing_only\supported_op_partitions\partition_1_prefifo_autosize.onnx"
)
ANALYTICAL_FINAL = (
    r"C:\DEV\repos\LightM-UNet\MILP\artifacts\S12_dense_256_u4_analytical_v1\int6_fps250_lat200\enet_dataflow_final.onnx"
)


def dump_addstreams(path, label):
    print("=" * 20, label, "=" * 20)
    m = onnx.load(path)
    g = m.graph
    # build producer map: tensor -> node
    producer = {}
    for n in g.node:
        for o in n.output:
            producer[o] = n
    adds = [n for n in g.node if n.op_type.startswith("Add")]
    print(f"total nodes: {len(g.node)}, Add*-nodes: {len(adds)}")
    for i, n in enumerate(adds):
        ins_info = []
        for inp in n.input:
            p = producer.get(inp)
            ins_info.append(f"{inp}<-{p.op_type if p else '???'}:{p.name if p else ''}")
        print(f"  [{i}] name={n.name!r} op_type={n.op_type} inputs={ins_info}")


dump_addstreams(PARTITION1_PREFIFO, "partition_1_prefifo_autosize.onnx")
print()
dump_addstreams(ANALYTICAL_FINAL, "enet_dataflow_final.onnx")

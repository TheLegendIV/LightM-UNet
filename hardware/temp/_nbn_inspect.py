import sys, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
import node_by_node_check as n

NODE = sys.argv[1] if len(sys.argv) > 1 else "MVAU_rtl_1"
P = "finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions/partition_1_postfifo_autosize.onnx"
m = ModelWrapper(P)
node = [x for x in m.graph.node if x.name == NODE][0]
op = getCustomOp(node)
for a in ["MW", "MH", "SIMD", "PE", "inputDataType", "weightDataType", "outputDataType", "accDataType", "ActVal", "noActivation", "mem_mode", "binaryXnorMode", "resType", "code_gen_dir_ipgen"]:
    try:
        print(a, "=", op.get_nodeattr(a))
    except Exception as e:
        print(a, "ERR", e)
print("in", node.input, "out", node.output)
print("in dt", m.get_tensor_datatype(node.input[0]), m.get_tensor_shape(node.input[0]), "out dt", m.get_tensor_datatype(node.output[0]), m.get_tensor_shape(node.output[0]))
W = m.get_initializer(node.input[1]) if len(node.input) > 1 else None
if W is not None:
    print("W", W.shape, W.min(), W.max(), "unique", np.unique(W)[:12])
    print("W col0 (first 16):", W[:16, 0])
# python tensors
gdir = Path("/tmp/golden_pp/case0")
import golden_per_partition as gp
in_name = m.graph.input[0].name
x = gp.from_stream(np.fromfile(gdir / "p1_in.raw", np.uint8), m.get_tensor_datatype(in_name)).astype(np.float32).reshape(m.get_tensor_shape(in_name))
ctx = {t.name: np.asarray(m.get_initializer(t.name), dtype=np.float32) for t in m.graph.initializer}
ctx[in_name] = x
for nd in m.graph.node:
    o = getCustomOp(nd)
    for t in nd.output:
        ctx[t] = np.zeros(m.get_tensor_shape(t), np.float32)
    n.py_exec_fn(o)(o, ctx, m.graph)
    if nd.name == NODE:
        break
xin = ctx[node.input[0]]
yo = ctx[node.output[0]]
print("input first pixel (first 40):", xin.reshape(-1, xin.shape[-1])[0][:40].astype(int))
print("py out first pixel (first 16):", yo.reshape(-1, yo.shape[-1])[0][:16].astype(int))
if W is not None:
    mw = W.shape[0]
    ref = xin.reshape(-1, xin.shape[-1])[0][:mw] @ W
    print("x@W first pixel (first 16):", ref[:16].astype(int))
# rtl
tm = {t["stream"]: t for t in json.loads(Path("/tmp/nbn_p1/taps/taps.json").read_text())}
t = tm[f"GenericPartition_1_{NODE}_out_V"]
raw = np.fromfile(f"/tmp/nbn_p1/taps/tap_{t['idx']}.bin", np.uint8)
dt = op.get_output_datatype(0)
pe = op.get_nodeattr("PE")
print("tap nb", t["nb"], "beats", raw.size // t["nb"], "pe", pe, "bits", dt.bitwidth(), dt)
r = n.decode_stream(raw, t["nb"], pe, dt.bitwidth(), dt.signed())
print("rtl first pixel (first 16):", r.reshape(-1, yo.shape[-1])[0][:16])
print("raw first 16 bytes:", raw[:16])
# which output channel pattern: agreement per channel with sign/scale test
e = yo.reshape(-1, yo.shape[-1]).astype(np.int64)
rr = r[: e.size].reshape(e.shape)
bad = (e != rr)
print("bad frac per channel:", np.round(bad.mean(0), 3))

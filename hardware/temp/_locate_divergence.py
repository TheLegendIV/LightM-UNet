import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx

P = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/u8in_signbias2_preamble_20261008_174934/intermediate_models/"
MODEL = sys.argv[1] if len(sys.argv) > 1 else P + "step_enet_streamline.onnx"
G = "/tmp/golden_out/test_1_p0000_0000_input_u8/"
IN = "/tmp/golden_in/test_1_p0000_0000_input_u8.raw"

from qonnx.transformation.infer_shapes import InferShapes
m = ModelWrapper(MODEL).transform(InferShapes())
iname = m.graph.input[0].name
for node in m.graph.node:
    for i, t_ in enumerate(node.input):
        if t_ == '':
            node.input[i] = f'{node.name}_empty{i}'
            m.set_initializer(node.input[i], np.zeros((0,), dtype=np.float32))
m.check_all_tensor_shapes_specified(fix_missing_init_shape=True)
ishape = m.get_tensor_shape(iname)
print("input", iname, ishape, m.get_tensor_datatype(iname))
u = np.fromfile(IN, np.uint8).astype(np.float32).reshape(ishape)
ctx = execute_onnx(m, {iname: u}, return_full_exec_context=True)
out = m.graph.output[0].name
print('final', out, ctx[out].shape)

# boundary streams: (partition whose output it is, NHWC shape)
bounds = [(0, (128, 128, 4)), (1, (32, 32, 32)), (2, (32, 32, 32)), (3, (32, 32, 32)),
          (4, (32, 32, 32)), (5, (64, 64, 16)), (6, (128, 128, 4))]
for k, shp in bounds:
    raw = np.fromfile(G + f"GenericPartition_{k}_out.raw", np.uint8).astype(np.int64)
    signed = np.where(raw >= 64, raw - 128, raw) if k > 0 else raw
    n = raw.size
    best = []
    for name, t in ctx.items():
        t = np.asarray(t)
        if t.size != n or t.dtype.kind not in "fiu":
            continue
        for lay, tt in (("as_is", t), ("nchw2nhwc", t.transpose(0, 2, 3, 1) if t.ndim == 4 else None)):
            if tt is None:
                continue
            f = np.rint(tt.ravel()).astype(np.int64)
            for sn, s in (("raw", raw), ("signed", signed)):
                best.append(((f == s).mean(), name, lay, sn))
    best.sort(reverse=True)
    print(f"B{k} (p{k} out {shp}): top matches ->", [(round(a, 4), b, c, d) for a, b, c, d in best[:4]])

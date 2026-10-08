import sys, numpy as np
sys.argv = sys.argv[:1] + ["x"] + sys.argv[1:]
import finn_enet_build as fb
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds as R
orig_apply = R.apply
def apply(self, model):
    before = {n.name: model.get_initializer(n.input[1]).copy() for n in model.graph.node if n.op_type == "MultiThreshold"}
    out, mod = orig_apply(self, model)
    for n in out.graph.node:
        if n.op_type == "MultiThreshold":
            T = out.get_initializer(n.input[1]); T0 = before.get(n.name)
            if T0 is not None and T0.shape == T.shape and not np.array_equal(T0, T):
                p = out.find_producer(n.input[0])
                print("CHANGED", n.name, "in dtype", out.get_tensor_datatype(n.input[0]), "producer", p.op_type if p else None, p.name if p else None, "thr0", T0[0][:3], "->", T[0][:3], flush=True)
    return out, mod
R.apply = apply
m = ModelWrapper(sys.argv[2])
fb._streamline_linear(m, None)

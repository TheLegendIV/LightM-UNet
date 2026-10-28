import time, sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.general import GiveUniqueNodeNames
from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights
import collections

name = sys.argv[1]
t = time.time()
m = ModelWrapper("/home/thelegendiv/finn/notebooks/enet/%s.onnx" % name)
print("load %.1fs nodes=%d vi=%d init=%d" % (time.time() - t, len(m.graph.node), len(m.graph.value_info), len(m.graph.initializer)))
t = time.time()
m = m.transform(InferShapes())
print("InferShapes %.1fs" % (time.time() - t))
nq = sum(1 for n in m.graph.node if n.op_type == "Quant")
print("Quant nodes", nq)
tr = FoldQuantWeights()
for i in range(6):
    t = time.time()
    m, mod = tr.apply(m)
    print("fold iter %d: %.2fs modified=%s nodes=%d" % (i, time.time() - t, mod, len(m.graph.node)))
    if not mod:
        break

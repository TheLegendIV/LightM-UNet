"""G3 check on the BUILT hardware: first thresholding of partition 0 must hold u-domain thresholds 97..159."""
import sys
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper

OUT = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
BUILD = OUT + "S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions/"
PREAMBLES = ["u8in_signbias2_preamble_20261008_174934", "S12_dense_256_u4_u8in_biasfix_preamble_20261008_155048"]
EXPECT = np.arange(97, 160, dtype=np.float64)


def first_thr(path, op_substr):
    m = ModelWrapper(path)
    for n in m.graph.node:
        if op_substr in n.op_type:
            return m, n
    return m, None


def report(label, path, op_substr):
    m, n = first_thr(path, op_substr)
    if n is None:
        print(f"[G3] {label}: no {op_substr} node"); return False
    t = m.get_initializer(n.input[1])
    print(f"[G3] {label}: first {n.op_type} name={n.name} thr shape={t.shape} min={t.min()} max={t.max()} "
          f"in_dtype={m.get_tensor_datatype(n.input[0])} out_dtype={m.get_tensor_datatype(n.output[0])}")
    ok = t.shape[-1] == 63 and np.allclose(t, EXPECT[None, :], atol=1e-3) if t.ndim == 2 else False
    print(f"[G3] {label}: {'PASS' if ok else 'FAIL'}  (row0[:3]={t[0][:3]} row0[-3:]={t[0][-3:]})")
    return ok


ok = True
for p in PREAMBLES:
    ok &= report(f"preamble {p} streamline", OUT + p + "/intermediate_models/step_enet_streamline.onnx", "MultiThreshold")
for i in (0, 1):
    pass
ok &= report("build partition_0.onnx", BUILD + "partition_0.onnx", "Thresholding")
ok &= report("build partition_0_postfifo_autosize.onnx", BUILD + "partition_0_postfifo_autosize.onnx", "Thresholding")
print("G3 OVERALL", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

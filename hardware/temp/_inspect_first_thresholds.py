import numpy as np
from qonnx.core.modelwrapper import ModelWrapper

PRE = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_analytical_v2_ft15ep_preamble_20261008_032400/intermediate_models/"
for f in ["step_enet_tidy", "step_qonnx_to_finn", "step_enet_streamline", "step_enet_convert_to_hw_rtl_mvau"]:
    m = ModelWrapper(PRE + f + ".onnx")
    cur = m.graph.input[0].name
    print("=====", f, "input dt", m.get_tensor_datatype(cur))
    for _ in range(4):
        cons = m.find_consumer(cur)
        if cons is None:
            break
        print("node:", cons.op_type, cons.name)
        for i in cons.input[1:]:
            t = m.get_initializer(i)
            if t is not None:
                fl = np.asarray(t).reshape(-1)
                print("  init", i, t.shape, "vals", np.round(fl[:6], 4).tolist(), "..", np.round(fl[-3:], 4).tolist(), "uniq", np.unique(np.round(fl, 4)).size)
        for a in cons.attribute:
            if a.name in ("out_scale", "out_bias", "out_dtype"):
                print("  attr", a.name, a.f if a.type == 1 else a.s.decode())
        cur = cons.output[0]

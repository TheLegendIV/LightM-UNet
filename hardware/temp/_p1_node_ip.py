import warnings
warnings.filterwarnings("ignore")
import os, glob
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

D = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions/"
for f in sorted(glob.glob(D + "partition_1*.onnx")):
    m = ModelWrapper(f)
    print("==", os.path.basename(f), len(m.graph.node), "nodes")
    seen = 0
    for n in m.graph.node:
        if n.op_type not in ("StreamingMaxPool_hls", "ConvolutionInputGenerator_rtl", "Thresholding_rtl", "MVAU_rtl"):
            continue
        o = getCustomOp(n)
        r = {a: o.get_nodeattr(a) for a in ("code_gen_dir_ipgen", "ipgen_path", "ip_path", "exec_mode", "rtlsim_so") if a in [x.name for x in n.attribute]}
        ok = {k: (os.path.exists(v) if isinstance(v, str) and v else None) for k, v in r.items()}
        if seen < 6:
            print(n.op_type, n.name.split("_", 1)[1], {k: (v[-60:] if isinstance(v, str) else v) for k, v in r.items()}, ok)
            seen += 1

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import golden_per_partition as gp

PRE = "finn_deployment_outputs/u8in_signbias2_preamble_20261008_174934/intermediate_models/assign_stage_partition_ids_8way.onnx"
d = np.load("quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep_u8in_verify_ref.npz")
cases = [int(x) for x in sys.argv[1].split(",")]
for i in cases:
    u8 = d["u"][i].astype(np.uint8)
    res, _, _ = gp.software_boundaries(Path(PRE), u8, None)
    outs = [(t, a) for t, a, dt in res[7] if a.size == 65536 and str(dt) == "UINT8"]
    onnx = np.rint(outs[-1][1]).astype(np.uint8).reshape(256, 256)
    torch_p = d["pred"][i]
    rtl = np.fromfile(f"/tmp/chain_mixed_all/case{i}/GenericPartition_7_out.raw", np.uint8).reshape(256, 256)
    corner = np.zeros((256, 256), bool)
    corner[254:, 254:] = True
    def ex(a, b):
        return int(((a != b) & ~corner).sum())
    print(f"case{i}: onnx-vs-torch bad {int((onnx != torch_p).sum())} (non-corner {ex(onnx, torch_p)}) | "
          f"rtl-vs-onnx bad {int((rtl != onnx).sum())} (non-corner {ex(rtl, onnx)}) | "
          f"rtl-vs-torch bad {int((rtl != torch_p).sum())} (non-corner {ex(rtl, torch_p)}) | "
          f"onnx corner {onnx[254:, 254:].ravel().tolist()} torch corner {torch_p[254:, 254:].ravel().tolist()} rtl corner {rtl[254:, 254:].ravel().tolist()}",
          flush=True)

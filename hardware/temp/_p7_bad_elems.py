import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import golden_per_partition as gp
import tap_compare as tc
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

SW = "finn_deployment_outputs/u8in_signbias2_preamble_20261008_174934/intermediate_models/assign_stage_partition_ids_8way.onnx"
PART = "finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions/partition_7.onnx"
taps = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/taps_p7")
u8 = np.fromfile("/tmp/chain_mixed_all/case0/in.raw", np.uint8)
ctx = {}
gp.software_boundaries(Path(SW), u8, None, full_ctx=ctx)
pm = ModelWrapper(PART)
nodes = {n.name: n for n in pm.graph.node}
tj = {t["stream"]: t for t in json.loads((taps / "taps.json").read_text())}
pairs = [("MVAU_rtl_1", "MVAU_84_out0"), ("Thresholding_rtl_4", "Thresholding_139_out0"),
         ("FMPadding_Pixel_hls_0", "FMPadding_Pixel_2_out0"), ("MVAU_rtl_2", "MVAU_85_out0")]
for nm, swn in pairs:
    nd = nodes[f"GenericPartition_7_{nm}"]
    inst = getCustomOp(nd)
    dt = inst.get_output_datatype(0)
    pe = inst.get_folded_output_shape(0)[-1]
    shape = inst.get_normal_output_shape(0)
    t = tj[f"GenericPartition_7_{nm}_out_V"] if f"GenericPartition_7_{nm}_out_V" in tj else tj[f"GenericPartition_7_{nm}_out0_V"]
    raw = np.fromfile(taps / f"tap_{t['idx']}.bin", np.uint8)
    v = tc.decode(raw, t["nb"], pe, dt.bitwidth(), dt.signed()).reshape(shape)
    r = np.rint(np.asarray(ctx[swn])).astype(np.int64).reshape(shape)
    bad = np.argwhere(v != r)
    print(nm, "shape", shape, "n_bad", len(bad), flush=True)
    for b in bad[:12]:
        print("   idx", b.tolist(), "rtl", int(v[tuple(b)]), "sw", int(r[tuple(b)]), flush=True)

#!/bin/bash
export HOME=/tmp/home_dir
cd /home/thelegendiv/finn/notebooks/enet
D=finn_deployment_outputs/$(ls finn_deployment_outputs | grep S12_dense_256_u4_bilinear_analytical_v1_ft15ep_preamble_ | tail -1)
M=$D/intermediate_models
echo "== verify (streamline)"
python3 verify_export.py --ref quantEnet_S12_dense_256_u4_bilinear_analytical_v1_ft15ep_u8in_verify_ref.npz \
  --onnx streamline=$M/step_enet_streamline.onnx --first-thresholds $M/step_enet_streamline.onnx 2>&1 | grep -vE "Warning|warn" | tail -12
echo "== final model op histogram + partitions"
python3 - <<EOF 2>&1 | grep -vE "Warning|warn"
import collections
from qonnx.core.modelwrapper import ModelWrapper
m = ModelWrapper("$M/assign_stage_partition_ids_8way.onnx")
c = collections.Counter(n.op_type for n in m.graph.node)
print(dict(c))
p = collections.Counter()
for n in m.graph.node:
    a = [x for x in n.attribute if x.name == "partition_id"]
    p[a[0].i if a else None] += 1
print("partition_id counts:", dict(sorted(p.items(), key=lambda kv: (kv[0] is None, kv[0]))))
for n in m.graph.node:
    if n.op_type in ("UpsampleNearestNeighbour", "UpsampleNearestNeighbour_Batch", "FMPadding_rtl", "FMPadding", "VVAU", "VVAU_hls", "VVAU_rtl", "ConvolutionInputGenerator_rtl", "ConvolutionInputGenerator"):
        a = {x.name: (x.i if x.type == 2 else x.ints and list(x.ints) or x.s.decode() if x.s else None) for x in n.attribute if x.name in ("partition_id", "depthwise", "SIMD", "PE", "IFMDim", "KernelDim", "IFMChannels", "ImgDim", "NumChannels", "OFMDim")}
        print(n.name, n.op_type, a)
EOF

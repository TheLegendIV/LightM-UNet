#!/bin/bash
export HOME=/tmp/home_dir
cd /home/thelegendiv/finn/notebooks/enet
D=finn_deployment_outputs/$(ls finn_deployment_outputs | grep S12_dense_512_u4_analytical_v1_ft15ep_preamble_ | tail -1)
M=$D/intermediate_models
echo "preamble: $D"
echo "== verify (streamline)"
python3 verify_export.py --ref quantEnet_S12_dense_512_u4_analytical_v1_ft15ep_u8in_verify_ref.npz \
  --onnx streamline=$M/step_enet_streamline.onnx --first-thresholds $M/step_enet_streamline.onnx 2>&1 | grep -vE "Warning|warn" | tail -12
echo "== dangling report"
python3 -c "import json;r=json.load(open('$M/dangling_node_report.json'));print(r['n_dangling'],'/',r['total_nodes'])"
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
EOF

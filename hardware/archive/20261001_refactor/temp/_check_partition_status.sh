#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_512x512_20260922_011241
echo "---LS---"
ls -la --time-style=full-iso supported_op_partitions/ 2>/dev/null
echo "---OPTYPES---"
for f in supported_op_partitions/partition_*.onnx; do
  echo "$f:"
  python3 -c "
import onnx
m = onnx.load('$f')
ops = set(n.op_type for n in m.graph.node)
print(sorted(ops))
" 2>&1 | tail -3
done

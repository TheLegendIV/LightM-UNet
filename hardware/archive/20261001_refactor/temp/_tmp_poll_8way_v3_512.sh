#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
OUTDIR=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_512x512_20260920_181619
echo "--- top level ---"
ls -la "$OUTDIR" 2>/dev/null
echo "--- supported_op_partitions ---"
ls -la "$OUTDIR/supported_op_partitions" 2>/dev/null
echo "--- intermediate_models ---"
ls -la "$OUTDIR/intermediate_models" 2>/dev/null | tail -20
echo "--- report dir (if any) ---"
ls -la "$OUTDIR/report" 2>/dev/null
echo "--- process elapsed time ---"
ps -o pid,etime,cmd -p 563742,563743,563772,563773,563774,563775 2>/dev/null

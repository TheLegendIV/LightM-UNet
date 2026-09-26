#!/bin/bash
cd ~/finn/notebooks/enet
for f in finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram.py \
         finn_hawq_preamble_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_512x512.py \
         regen_stitched_ip_8way_full_v3.py \
         finn_enet_ip_resume_partitioned_8way_skip_rtlsim.py \
         finn_zynqbuild_12_dense_relu_alpha025_rtl_mvau_v3_partition0.py \
         finn_zynqbuild_12_dense_relu_alpha025_rtl_mvau_v3_partitionN.py \
         analyze_fifo_topology.py \
         append_result_row.py \
         finn_partition_build_steps_rtl_mvau.py \
         finn_partition_build_steps.py; do
  echo "=== $f ==="
  if [ -f "$f" ]; then
    grep -E '^(from|import) [a-zA-Z_]' "$f" | grep -v -E 'numpy|torch|onnx|finn\.|qonnx|brevitas|^import (os|sys|json|re|time|shutil|copy|glob|subprocess|argparse|logging|dataclasses|pickle|math|traceback|warnings|random|itertools|functools|collections|typing|datetime|concurrent)'
  else
    echo "  (FILE NOT FOUND)"
  fi
done

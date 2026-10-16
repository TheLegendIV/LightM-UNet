#!/bin/bash
# Per-partition golden check (RTL via Verilator vs software boundary tensors) of the ASSEMBLED INT8 design:
#   partition 0 from the InferPool initial-pool build, 1/5/7 from the name-stable-FIFO rebuild, 2/3/4/6 from the full 8-way build.
# Software reference = the original preamble's assign_stage_partition_ids_8way.onnx (StreamingMaxPool python exec = true max-pool semantics).
# Usage: docker exec finn_persistent bash _launch_golden_int8.sh [extra golden_per_partition.py args, e.g. --valid-pct 60 --ready-pct 60 --out golden_pp_int8_bp]
export HOME=/tmp/home_dir
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
T=finn_build_tmp
A=$T/S12_dense_512_bilinear_int8_maxfps_v1_assembled
mkdir -p $A
ln -sfn $PWD/$T/S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_ipool_p0/GenericPartition_0 $A/GenericPartition_0
for i in 1 5 7; do ln -sfn $PWD/$T/S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_p157fix/GenericPartition_$i $A/GenericPartition_$i; done
for i in 2 3 4 6; do ln -sfn $PWD/$T/S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_pairs/GenericPartition_$i $A/GenericPartition_$i; done
OUT=golden_pp_int8
nohup python3 -u golden_per_partition.py \
  --preamble finn_deployment_outputs/S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_pairs_preamble_20261015_182343 \
  --build-tmp $A \
  --ref quantEnet_S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_u8in_verify_ref.npz \
  --cases 0,1 --out $OUT "$@" > $OUT.log 2>&1 &
echo "launched $!"
disown -a

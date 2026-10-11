#!/bin/bash
# Bridge-only dry run for partitions 5 6 into a scratch output dir; prints the dw SWU lines and the VVAU/SWG/FMPad entries.
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
PDIR=finn_deployment_outputs/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_preamble_20261009_220505
OUT=/tmp/bil_dwfix_bridge
rm -rf $OUT; mkdir -p $OUT
python3 -u finn_s12_build.py $PDIR \
  --tag S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200 \
  --conv-order quantEnet_S12_dense_256_u4_bilinear_analytical_v1_conv_order.json \
  --folding-json layer_bits_folding_S12_dense_256_u4_bilinear_analytical_v1_int6_fps250_lat200.json \
  --partitions 5 6 --bridge-only --output-dir $OUT > /tmp/bil_dwfix_bridge.log 2>&1
echo "exit=$?"
grep -E "dw SWU|VVAU|Error|Traceback" /tmp/bil_dwfix_bridge.log | cut -c1-220
python3 - <<'EOF'
import json
for i in (5, 6):
    d = json.load(open(f"/tmp/bil_dwfix_bridge/hawq_folding_config_partition{i}.json"))
    print(i, {k: v for k, v in d.items() if "VVAU" in k or k.startswith("GenericPartition_%d_Conv" % i) and False})
    print(i, "n entries", len(d))
EOF

#!/bin/bash
# Generic launcher for one S12 512 bilinear build in finn_persistent: preamble (+ its checks), then the full 8-way build.
# Usage (inside the FINN container):  bash _launch_job.sh <name-stem> <tag> <folding-json-file-name> [preamble|build|all]
#   e.g. bash _launch_job.sh quantEnet_S12_dense_512_bilinear_int4_cap40_v1 S12_dense_512_bilinear_int4_cap40_v1 layer_bits_folding_S12_512_int4_cap40_simfifo.json all
# Needs in /home/thelegendiv/finn/notebooks/enet: <stem>_ft15ep_u8in.onnx, <stem>_conv_order.json, <folding json>. Logs: preamble_<tag>.log, build_<tag>.log, launch_<tag>.log.
# The build only starts when the preamble (streamline accuracy gate, threshold check, bridge / FIFO check) exited 0 AND printed BRIDGE EXIT=0. The preamble takes INITIAL_POOL_ROUTE from the json's pool_impl.
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
export NET=$1 TAG=$2 FOLD=$3 STAGE=${4:-all}
[ -n "$NET" ] && [ -n "$TAG" ] && [ -n "$FOLD" ] || { echo "usage: bash _launch_job.sh <stem> <tag> <folding json> [preamble|build|all]"; exit 2; }
cat > _run_$TAG.sh << 'EOF'
#!/bin/bash
if [ "$STAGE" = preamble ] || [ "$STAGE" = all ]; then
  python3 -u finn_s12_preamble.py ${NET}_ft15ep_u8in --tag $TAG --conv-order ${NET}_conv_order.json --folding-json $FOLD > preamble_$TAG.log 2>&1 || { echo "preamble failed"; exit 1; }
  grep -q "BRIDGE EXIT=0" preamble_$TAG.log || { echo "preamble bridge / FIFO check did not exit 0"; exit 1; }
fi
if [ "$STAGE" = build ] || [ "$STAGE" = all ]; then
  PDIR=$(ls -dt finn_deployment_outputs/${TAG}_preamble_* | head -1)
  echo "preamble dir: $PDIR" > build_$TAG.log
  python3 -u finn_s12_build.py $PDIR --tag $TAG --conv-order ${NET}_conv_order.json --folding-json $FOLD --partitions all >> build_$TAG.log 2>&1
fi
EOF
nohup bash _run_$TAG.sh > launch_$TAG.log 2>&1 &
echo "launched $!"
disown -a

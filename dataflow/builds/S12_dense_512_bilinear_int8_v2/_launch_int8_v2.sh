#!/bin/bash
# Uniform-INT8 bilinear 512 network, MILP artifact S12_512_bilin_int8_v2 (max-FPS solve with the corrected LabelSelect / pool cost model; InferPool initial pool, LabelSelect PE 5).
# Same exported network as S12_dense_512_bilinear_int8_maxfps_v1 (u8in onnx + conv order); only the folding json changes:
#   layer_bits_folding_S12_512_bilin_int8_v2_simfifo.json = MILP/artifacts/S12_512_bilin_int8_v2/simfifo/layer_bits_folding_maxfps_simfifo.json (v2 folding + net_explicit.py FIFO lists).
# The preamble takes INITIAL_POOL_ROUTE (infer_pool) from the json's pool_impl; the build checks the lowered partitions against it.
# Run inside the FINN container:   docker exec finn_persistent bash /home/thelegendiv/finn/notebooks/enet/_launch_int8_v2.sh [preamble|build|all]   (default all: preamble, then the 8-way build)
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
export STAGE=${1:-all}
export NET=quantEnet_S12_dense_512_bilinear_int8_maxfps_v1
export TAG=S12_dense_512_bilinear_int8_v2
export FOLD=layer_bits_folding_S12_512_bilin_int8_v2_simfifo.json
cat > _run_int8_v2.sh << 'EOF'
#!/bin/bash
if [ "$STAGE" = preamble ] || [ "$STAGE" = all ]; then
  python3 -u finn_s12_preamble.py ${NET}_ft15ep_u8in --tag $TAG --conv-order ${NET}_conv_order.json --folding-json $FOLD > preamble_int8_v2.log 2>&1 || { echo "preamble failed"; exit 1; }
fi
if [ "$STAGE" = build ] || [ "$STAGE" = all ]; then
  PDIR=$(ls -dt finn_deployment_outputs/${TAG}_preamble_* | head -1)
  echo "preamble dir: $PDIR" > build_int8_v2.log
  python3 -u finn_s12_build.py $PDIR --tag $TAG --conv-order ${NET}_conv_order.json --folding-json $FOLD --partitions all >> build_int8_v2.log 2>&1
fi
EOF
nohup bash _run_int8_v2.sh > launch_int8_v2.log 2>&1 &
echo "launched $!"
disown -a

#!/bin/bash
# Uniform-INT6 bilinear 512 network (fine-tuned uniform INT6 checkpoint), MILP artifact S12_512_ablation_v2/cap40_fps150/uniform_int6 (0.4 caps, 150 FPS at 100 MHz, min-resources, recalibrated cost model).
# Folding json = ILP/artifacts/S12_512_ablation_v2/cap40_fps150/uniform_int6/simfifo/layer_bits_folding_int6_cap40_simfifo.json (the MILP folding + net_explicit.py FIFO lists).
# The preamble takes INITIAL_POOL_ROUTE (infer_pool) from the json's pool_impl; the build checks the lowered partitions against it.
# Run inside the FINN container:   docker exec finn_persistent bash /home/thelegendiv/finn/notebooks/enet/_launch_int6_cap40_v1.sh [preamble|build|all]   (default all: preamble + its checks, then the 8-way build;
# the build only starts when the preamble (streamline accuracy gate, threshold check, bridge / FIFO check) exited 0)
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
export STAGE=${1:-all}
export NET=quantEnet_S12_dense_512_bilinear_int6_cap40_v1
export TAG=S12_dense_512_bilinear_int6_cap40_v1
export FOLD=layer_bits_folding_S12_512_int6_cap40_simfifo.json
cat > _run_int6_cap40.sh << 'EOF'
#!/bin/bash
if [ "$STAGE" = preamble ] || [ "$STAGE" = all ]; then
  python3 -u finn_s12_preamble.py ${NET}_ft15ep_u8in --tag $TAG --conv-order ${NET}_conv_order.json --folding-json $FOLD > preamble_int6_cap40.log 2>&1 || { echo "preamble failed"; exit 1; }
  grep -q "BRIDGE EXIT=0" preamble_int6_cap40.log || { echo "preamble bridge / FIFO check did not exit 0"; exit 1; }
fi
if [ "$STAGE" = build ] || [ "$STAGE" = all ]; then
  PDIR=$(ls -dt finn_deployment_outputs/${TAG}_preamble_* | head -1)
  echo "preamble dir: $PDIR" > build_int6_cap40.log
  python3 -u finn_s12_build.py $PDIR --tag $TAG --conv-order ${NET}_conv_order.json --folding-json $FOLD --partitions all >> build_int6_cap40.log 2>&1
fi
EOF
nohup bash _run_int6_cap40.sh > launch_int6_cap40.log 2>&1 &
echo "launched $!"
disown -a

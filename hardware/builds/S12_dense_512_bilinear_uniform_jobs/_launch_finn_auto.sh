#!/bin/bash
# FINN-native build of one S12 512 bilinear network: FINN auto-folding (step_target_fps_parallelization, no MILP folding) + FINN's rtlsim-based FIFO autosizing.
# Usage (inside the FINN container):  bash _launch_finn_auto.sh <name-stem> <tag> <target-fps> [preamble|build|all]
#   e.g. bash _launch_finn_auto.sh quantEnet_S12_dense_512_bilinear_int6_cap40_v1 S12_dense_512_bilinear_int6_cap40_finnauto_v1 150 all
# Needs <stem>_ft15ep_u8in.onnx in /home/thelegendiv/finn/notebooks/enet. Logs: preamble_<tag>.log, build_<tag>.log, launch_<tag>.log.
# No --folding-json: no pool_impl from the MILP, so INITIAL_POOL_ROUTE stays unset (FINN's stock MaxPool lowering); no bridge/FIFO check.
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT INITIAL_POOL_ROUTE
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
export NET=$1 TAG=$2 FPS=$3 STAGE=${4:-all}
[ -n "$NET" ] && [ -n "$TAG" ] && [ -n "$FPS" ] || { echo "usage: bash _launch_finn_auto.sh <stem> <tag> <target-fps> [preamble|build|all]"; exit 2; }
cat > _run_$TAG.sh << 'EOF'
#!/bin/bash
if [ "$STAGE" = preamble ] || [ "$STAGE" = all ]; then
  python3 -u finn_s12_preamble.py ${NET}_ft15ep_u8in --tag $TAG > preamble_$TAG.log 2>&1 || { echo "preamble failed"; exit 1; }
fi
if [ "$STAGE" = build ] || [ "$STAGE" = all ]; then
  PDIR=$(ls -dt finn_deployment_outputs/${TAG}_preamble_* | head -1)
  echo "preamble dir: $PDIR" > build_$TAG.log
  python3 -u finn_s12_build.py $PDIR --tag $TAG --target-fps $FPS --fifo-autosize rtlsim --partitions all >> build_$TAG.log 2>&1
fi
EOF
nohup bash _run_$TAG.sh > launch_$TAG.log 2>&1 &
echo "launched $!"
disown -a

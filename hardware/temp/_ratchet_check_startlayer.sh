cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
PDIR=finn_deployment_outputs/ratchet256_preamble_20261009_003521
CO=quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_conv_order.json
for arm in ratchet_1pct_simfifo ratchet_25pct_simfifo ratchet_100pct_simfifo ratchet_200pct_simfifo ratchet_off_simfifo; do
  echo "--- $arm (start stage2.0.reduce.0)"
  python3 check_milp_vs_landed_folding.py layer_bits_folding_$arm.json $PDIR/landed_partition2_$arm.onnx --conv-order $CO --start-layer stage2.0.reduce.0 2>&1 | grep -v Warning | cut -c1-300
done

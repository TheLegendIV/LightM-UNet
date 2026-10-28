cd /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs || exit 1
B=S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350
echo "-- build log mentions of preamble/model path"
grep -a -m6 -i -E 'preamble|streamline|\.onnx' $B/build_dataflow.log | cut -c1-220
echo "-- preamble dirs"
for d in u8in_signbias2_preamble_20261008_174934 S12_dense_256_u4_u8in_biasfix_preamble_20261008_155048; do
  echo "== $d"; ls -l --time-style=+%F_%T $d/intermediate_models 2>/dev/null | cut -c30-140 | head -30
done
echo "-- partition-0 workdir in finn_build_tmp"
ls /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200/ | head
ls /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200/GenericPartition_0 | grep -v code_gen_ipgen | head -20
echo "-- readme ablation mentions"
grep -n -i 'preamble' /home/thelegendiv/finn/notebooks/enet/run_arms.sh 2>/dev/null | head -8
ls /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/$B/intermediate_models/supported_op_partitions/ | head -3

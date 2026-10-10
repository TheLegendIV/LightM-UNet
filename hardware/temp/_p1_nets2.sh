cd /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200/GenericPartition_1
F=$(ls -d vivado_stitch_proj_*)/GenericPartition_1_wrapper.v
sed -n 17340,17420p $F
echo ==== maxpool nets
grep -n -E "^\s*wire .*MaxPool_hls_0" $F | head -20
echo ==== total tdata wires in the top IPI module
awk 'NR>17340 && NR<30000 && /^  wire .*tdata/' $F | wc -l
awk 'NR>17340 && NR<30000 && /^  wire .*tdata/' $F | head -12

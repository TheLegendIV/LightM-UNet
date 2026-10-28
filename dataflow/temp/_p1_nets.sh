cd /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200/GenericPartition_1
P=$(ls -d vivado_stitch_proj_*)
ls -la $P | head -20
ls $P/*.v 2>/dev/null
F=$P/GenericPartition_1_wrapper.v
wc -l $F
grep -n -E "^\s*wire .*_tdata|^\s*wire .*tdata" $F | head -20
grep -c "" $F
echo == compile.sh
cat rtlsim_single/compile.sh | cut -c1-600
echo == candidates of block design file
ls $P/*.srcs/sources_1/bd/*/hdl/ 2>/dev/null | head
grep -n -E "StreamingMaxPool_hls_0" $P/$P.srcs/sources_1/bd/*/hdl/*.v 2>/dev/null | head -10

cd /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_p7mvu/GenericPartition_7
W=$(ls -d vivado_stitch_proj_*)/GenericPartition_7_wrapper.v
ls -la $W
grep -n "MVAU_rtl_1" $W | head -30
echo ---- weight/memstream refs
grep -n -iE "memstream" $W | head -10
sed -n 1,80p /home/thelegendiv/finn/finn-rtllib/memstream/hdl/memstream.sv | head -5
ls code_gen_ipgen_*MVAU_rtl_1_*/

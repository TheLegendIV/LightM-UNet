cd /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200/GenericPartition_1
tr ' ' '\n' < rtlsim_single/compile.sh | grep -v '^$' | tail -22
awk 'NR>=33805 && /^endmodule/{print "endmodule at", NR; exit}' vivado_stitch_proj_uacq2nrg/GenericPartition_1_wrapper.v
ls rtlsim_single | head -20

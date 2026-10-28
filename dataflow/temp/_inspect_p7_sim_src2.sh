D=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v2_ft15ep_int6_fps250_lat200/GenericPartition_7
grep -n "module mvu_8sx8u_dsp48\|module mvu_vvu_axi\b" $D/vivado_stitch_proj_*/GenericPartition_7_wrapper.v | head
grep -c "if(zero)     B1" $D/vivado_stitch_proj_*/GenericPartition_7_wrapper.v
grep -n "if(zero)     B1" $D/vivado_stitch_proj_*/GenericPartition_7_wrapper.v | head
ls $D/vivado_stitch_proj_*/ | head -20
diff $D/vivado_stitch_proj_*/ip/src/mvu_8sx8u_dsp48.sv /home/thelegendiv/finn/finn-rtllib/mvu/mvu_8sx8u_dsp48.sv && echo SAME_AS_RTLLIB
cd /home/thelegendiv/finn; git status --short finn-rtllib | head
grep -rln "mvu_8sx8u_dsp48\|mvu_4sx4u\|mvu_vvu_8sx9" /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/*/GenericPartition_*/vivado_stitch_proj_*/GenericPartition_*_wrapper.v 2>/dev/null | head -20

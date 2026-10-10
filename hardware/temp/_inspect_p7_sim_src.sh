cd /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v2_ft15ep_int6_fps250_lat200/GenericPartition_7
ls
ls rtlsim_single 2>/dev/null | head -30
cat rtlsim_single/compile.sh 2>/dev/null | cut -c1-1500
find /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v2_ft15ep_int6_fps250_lat200/GenericPartition_7 -name 'mvu_8sx8u_dsp48*' | head
grep -rl "mvu_8sx8u_dsp48" /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v2_ft15ep_int6_fps250_lat200/GenericPartition_7/rtlsim_single 2>/dev/null | head

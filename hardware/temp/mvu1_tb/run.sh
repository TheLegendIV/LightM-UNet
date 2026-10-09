unset VERILATOR_ROOT
export HOME=/tmp/home_dir
V5=/tmp/home_dir/.local/lib/python3.10/site-packages/verilator/bin/verilator
R=/home/thelegendiv/finn/finn-rtllib/mvu
D=$(ls -d /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_p7mvu/GenericPartition_7/code_gen_ipgen_GenericPartition_7_MVAU_rtl_1_*)
O=/tmp/mvu1_tb
rm -rf $O; mkdir -p $O
cp /tmp/mvu1_tb_src/tb.cpp $O/
cd $O
perl $V5 --cc --exe --build -j 8 -Wno-fatal -Wno-lint -Wno-style -Wno-STMTDLY --no-timing -CFLAGS -std=c++17 \
  --top-module GenericPartition_7_MVAU_rtl_1 -Mdir $O/obj \
  $R/mvu_pkg.sv $R/add_multi.sv $R/replay_buffer.sv $R/mvu.sv $R/mvu_vvu_8sx9_dsp58.sv $R/mvu_vvu_axi.sv \
  $D/GenericPartition_7_MVAU_rtl_1_wrapper.v $O/tb.cpp -o tb > $O/build.log 2>&1
echo build rc=$?
tail -5 $O/build.log
for cfg in "2000 100 100 100 1" "2000 100 100 50 2" "2000 50 100 100 3" "2000 100 50 100 4" "2000 50 50 50 5" "2000 90 90 90 6" "2000 100 100 100 7 48 63" "2000 70 100 30 8 48 63"; do
  $O/obj/tb $cfg
done

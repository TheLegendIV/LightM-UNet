unset VERILATOR_ROOT
export HOME=/tmp/home_dir
V5=/tmp/home_dir/.local/lib/python3.10/site-packages/verilator/bin/verilator
R0=/home/thelegendiv/finn/finn-rtllib/mvu
R=/tmp/mvu1_patched_rtl
D=$(ls -d /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_p7mvu/GenericPartition_7/code_gen_ipgen_GenericPartition_7_MVAU_rtl_1_*)
O=/tmp/mvu1_tb_p
EXTRA=${1:-2}
rm -rf $R $O; mkdir -p $R $O
cp $R0/*.sv $R/
sed -i "s/localparam int unsigned  MAX_IN_FLIGHT = CORE_PIPELINE_DEPTH;/localparam int unsigned  MAX_IN_FLIGHT = CORE_PIPELINE_DEPTH + $EXTRA;/" $R/mvu_vvu_axi.sv
grep -n "MAX_IN_FLIGHT =" $R/mvu_vvu_axi.sv
cp /tmp/mvu1_tb_src/tb.cpp $O/
cd $O
perl $V5 --cc --exe --build -j 8 -Wno-fatal -Wno-lint -Wno-style -Wno-STMTDLY --no-timing -CFLAGS -std=c++17 \
  --top-module GenericPartition_7_MVAU_rtl_1 -Mdir $O/obj \
  $R/mvu_pkg.sv $R/add_multi.sv $R/replay_buffer.sv $R/mvu.sv $R/mvu_vvu_8sx9_dsp58.sv $R/mvu_vvu_axi.sv \
  $D/GenericPartition_7_MVAU_rtl_1_wrapper.v $O/tb.cpp -o tb > $O/build.log 2>&1
echo build rc=$?
for cfg in "2000 100 100 50 2" "2000 70 100 30 8 48 63" "2000 100 100 10 9" "2000 100 100 90 10" "2000 100 100 20 11 0 63" "2000 100 100 5 12"; do
  $O/obj/tb $cfg
done

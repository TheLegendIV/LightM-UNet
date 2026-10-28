set -u
B=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200/GenericPartition_7
SRC=/tmp/alpha_rtllib/finn-rtllib/mvu
W=/tmp/alpha_try
rm -rf $W && mkdir -p $W
source /tools/Xilinx/Vivado/2022.2/settings64.sh >/dev/null 2>&1
FILES="$SRC/mvu_pkg.sv $SRC/add_multi.sv $SRC/replay_buffer.sv $SRC/mvu.sv $SRC/mvu_vvu_8sx9_dsp58.sv $SRC/mvu_vvu_axi.sv"
for n in 0 1 2; do
  d=$(ls -d $B/code_gen_ipgen_GenericPartition_7_MVAU_rtl_${n}_*)
  for kind in wrapper wrapper_sim; do
    sed -e 's/parameter COMPUTE_CORE = "mvu_8sx8u_dsp48"/parameter VERSION = 2/' \
        -e 's/\.COMPUTE_CORE(COMPUTE_CORE)/.VERSION(VERSION)/' \
        $d/GenericPartition_7_MVAU_rtl_${n}_$kind.v > $W/mvau${n}_$kind.v
  done
  grep -n "VERSION\|COMPUTE_CORE" $W/mvau${n}_wrapper.v | head -4
done
echo "=== xvlog"
cd $W
xvlog -sv $FILES $W/mvau0_wrapper.v $W/mvau1_wrapper.v $W/mvau2_wrapper.v 2>&1 | tail -15
echo "=== verilator lint"
for n in 0 1 2; do
  top=GenericPartition_7_MVAU_rtl_${n}_axi_wrapper
  grep -m1 -oE "^module [A-Za-z0-9_]+" $W/mvau${n}_wrapper.v
  verilator --lint-only -Wno-fatal -Wno-lint -Wno-style --top-module $(grep -m1 -oE "^module [A-Za-z0-9_]+" $W/mvau${n}_wrapper.v | awk '{print $2}') $FILES $W/mvau${n}_wrapper_sim.v 2>&1 | head -15
done

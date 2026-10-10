set -u
B=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200/GenericPartition_7
SRC=/tmp/alpha_rtllib/finn-rtllib/mvu
W=/tmp/alpha_try
rm -rf $W && mkdir -p $W/rtl
source /tools/Xilinx/Vivado/2022.2/settings64.sh >/dev/null 2>&1
cp $SRC/*.sv $W/rtl/
sed -n 275,292p $W/rtl/mvu.sv
grep -n " iff " $W/rtl/*.sv
FILES="$W/rtl/mvu_pkg.sv $W/rtl/add_multi.sv $W/rtl/replay_buffer.sv $W/rtl/mvu.sv $W/rtl/mvu_vvu_8sx9_dsp58.sv $W/rtl/mvu_vvu_axi.sv"
for n in 0 1 2; do
  d=$(ls -d $B/code_gen_ipgen_GenericPartition_7_MVAU_rtl_${n}_*)
  for kind in wrapper wrapper_sim; do
    sed -E -e 's/parameter\s+COMPUTE_CORE\s*=\s*"mvu_8sx8u_dsp48"/parameter VERSION = 2/' \
        -e 's/\.COMPUTE_CORE\(COMPUTE_CORE\)/.VERSION(VERSION)/' \
        $d/GenericPartition_7_MVAU_rtl_${n}_$kind.v > $W/mvau${n}_$kind.v
  done
  grep -n "VERSION\|COMPUTE_CORE" $W/mvau${n}_wrapper.v | head -4
done
echo "=== xvlog (errors only)"
cd $W
xvlog -sv $FILES $W/mvau0_wrapper.v $W/mvau1_wrapper.v $W/mvau2_wrapper.v 2>&1 | grep -E "ERROR|CRITICAL" | head -20
echo "xvlog done"
echo "=== verilator lint with iff rewritten"
cp $W/rtl/mvu.sv $W/rtl/mvu_orig.sv
sed -i -E 's/always_ff @\(posedge clk iff en && !rst\) begin/always_ff @(posedge clk) if(en \&\& !rst) begin/' $W/rtl/mvu.sv
sed -n 282p $W/rtl/mvu.sv
for n in 0 1 2; do
  top=$(grep -m1 -oE "^module [A-Za-z0-9_]+" $W/mvau${n}_wrapper.v | awk '{print $2}')
  echo "-- $top"
  verilator --lint-only -Wno-fatal -Wno-lint -Wno-style --top-module $top $FILES $W/mvau${n}_wrapper_sim.v 2>&1 | head -12
done

#!/bin/bash
# Build partition 0's OWN standalone stitched-IP top module
# (GenericPartition_0_wrapper) with Verilator 5.x (same working recipe as
# hardware/temp/compile_v5.sh for the combined design), using the
# backpressure-capable testbench driver instead of the stock
# always-ready one.
set -e

PROJ=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix/GenericPartition_0/vivado_stitch_proj__pkbn76f
OUT=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix/GenericPartition_0/verilator_fifosim_bp_v5
mkdir -p "$OUT"
cp /home/thelegendiv/finn/notebooks/enet/verilator_fifosim_bp_partition0.cpp "$OUT/verilator_fifosim_bp_partition0.cpp"

export VERILATOR_ROOT=/tmp/home_dir/.local/lib/python3.10/site-packages/verilator
VERILATOR_BIN=$VERILATOR_ROOT/bin/verilator

source /tools/Xilinx/Vivado/2022.2/settings64.sh

perl "$VERILATOR_BIN" -Wno-fatal -Mdir "$OUT" \
  -y "$PROJ" \
  -y "$PROJ/pyverilator_vh" \
  --CFLAGS --std=c++17 -O3 --x-assign fast --x-initial fast --noassert --cc \
  /home/thelegendiv/finn/finn-rtllib/swg/swg_pkg.sv \
  $(cat "$PROJ/all_verilog_srcs.txt") \
  /tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv \
  /tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_cdc/hdl/xpm_cdc.sv \
  /tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_fifo/hdl/xpm_fifo.sv \
  --top-module GenericPartition_0_wrapper --exe verilator_fifosim_bp_partition0.cpp \
  --no-timing \
  -DDISABLE_XPM_ASSERTIONS -DOBSOLETE -DONESPIN --bbox-unsup

cd "$OUT"
make -j4 -f VGenericPartition_0_wrapper.mk VGenericPartition_0_wrapper CFG_CXXFLAGS_PCH_I=-include

echo "BUILD DONE: $OUT/VGenericPartition_0_wrapper"

#!/bin/bash
export VERILATOR_ROOT=/tmp/home_dir/.local/lib/python3.10/site-packages/verilator
VERILATOR_BIN=$VERILATOR_ROOT/bin/verilator
export OPT_FAST='-O3 -march=native'
BUILD=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix
OUT=$BUILD/verilator_fifosim_v5
rm -rf "$OUT"
mkdir -p "$OUT"
cp "$BUILD/verilator_fifosim_nothreads/verilator_fifosim.cpp" "$OUT/"
perl "$VERILATOR_BIN" -Wno-fatal -Mdir "$OUT" \
  -y "$BUILD/combined_stitch_proj_drp47atx" \
  -y "$BUILD/combined_stitch_proj_drp47atx/pyverilator_vh" \
  --CFLAGS --std=c++17 -O3 --x-assign fast --x-initial fast --noassert --cc \
  /home/thelegendiv/finn/finn-rtllib/swg/swg_pkg.sv \
  "$BUILD/combined_stitch_proj_drp47atx/finn_design_wrapper.v" \
  /tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv \
  /tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_cdc/hdl/xpm_cdc.sv \
  /tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_fifo/hdl/xpm_fifo.sv \
  --top-module finn_design_wrapper --exe verilator_fifosim.cpp \
  --no-timing \
  -DDISABLE_XPM_ASSERTIONS -DOBSOLETE -DONESPIN --bbox-unsup \
  > /tmp/v5_verilate.log 2>&1
echo "VERILATE_EXIT=$?"
tail -n 30 /tmp/v5_verilate.log
cd "$OUT"
make -j4 -f Vfinn_design_wrapper.mk Vfinn_design_wrapper CFG_CXXFLAGS_PCH_I=-include > /tmp/v5_make.log 2>&1
echo "MAKE_EXIT=$?"
tail -n 30 /tmp/v5_make.log

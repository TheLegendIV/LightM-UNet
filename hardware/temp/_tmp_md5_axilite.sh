#!/bin/bash
BASE=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.gen/sources_1/bd/top/ip/top_GenericPartition_7_0_0/src
for f in GenericPartition_7_GenericPartition_7_MVAU_rtl_2_wstrm_0 GenericPartition_7_GenericPartition_7_MVAU_rtl_1_wstrm_0 GenericPartition_7_GenericPartition_7_VVAU_hls_0_wstrm_0 GenericPartition_7_GenericPartition_7_MVAU_rtl_0_wstrm_0; do
  p="$BASE/$f/hdl/axilite_if.v"
  if [ -f "$p" ]; then md5sum "$p"; else echo "MISSING: $p"; fi
done
IPSHARED=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.gen/sources_1/bd/top/ipshared/8544/verilog/rtl_ops/GenericPartition_7_Thresholding_rtl_5/axilite_if.v
if [ -f "$IPSHARED" ]; then md5sum "$IPSHARED"; else echo "MISSING: $IPSHARED"; fi

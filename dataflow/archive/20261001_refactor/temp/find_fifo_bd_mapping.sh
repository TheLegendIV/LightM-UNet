#!/bin/bash
BD=/tmp/finn_dev_thelegendiv/vivado_stitch_proj_d9ohtyc0/finn_vivado_stitch_proj.srcs/sources_1/bd/GenericPartition_2/GenericPartition_2.bd
echo "=== FIFO_MEMORY_TYPE occurrences in .bd ==="
grep -c 'FIFO_MEMORY_TYPE' "$BD"
echo "=== context around StreamingFIFO_rtl_3 cell in .bd ==="
grep -n 'StreamingFIFO_rtl_3"' "$BD" | head -20
echo "=== search for VLNV/instance ref lines near FIFO_MEMORY_TYPE (first 3) ==="
grep -n 'FIFO_MEMORY_TYPE' "$BD" | head -3

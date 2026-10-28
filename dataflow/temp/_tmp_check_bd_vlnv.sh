#!/bin/bash
BD=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.srcs/sources_1/bd/top/top.bd
echo "=== GenericPartition cell VLNVs/xci refs in top.bd ==="
grep -o '"vlnv":[^,]*' "$BD" | sort -u | head -20
echo "---"
grep -o '"xci_name":[^,}]*' "$BD" | sort -u | head -20
echo "=== actual xci file content for GenericPartition_7 (if present) ==="
find /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.srcs/sources_1/bd/top/ip -iname "*GenericPartition_7*" -iname "*.xci" 2>/dev/null

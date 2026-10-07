#!/bin/bash
BD=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.srcs/sources_1/bd/top/top.bd
echo "---VLNV counts---"
grep -oE 'VLNV="[^"]+"' "$BD" | sort | uniq -c | sort -rn
echo "---instance names---"
grep -oE 'INSTANCE="[^"]+"' "$BD" | sort -u
echo "---alt cell name attr---"
grep -oE '<spirit:cellName>[^<]+' "$BD" | sort -u

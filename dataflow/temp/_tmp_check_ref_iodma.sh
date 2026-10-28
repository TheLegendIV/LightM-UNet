#!/bin/bash
XPR=/home/thelegendiv/finn/notebooks/enet/backup/vivado_zynq_proj_ivzy_6_z/finn_zynq_link.xpr
echo "=== IODMA entries in reference xpr ==="
grep -o 'IPRepoPath" Val="[^"]*IODMA[^"]*"' "$XPR"
echo "=== also check BD .tcl / bd files referencing IODMA instance + VLNV ==="
find /home/thelegendiv/finn/notebooks/enet/backup/vivado_zynq_proj_ivzy_6_z -iname '*.bd' -o -iname '*.tcl' | xargs grep -l 'IODMA' 2>/dev/null

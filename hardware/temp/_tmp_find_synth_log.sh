#!/bin/bash
LOG=$(find /home/thelegendiv/finn/vivado_projects/S12_256_analytical -iname "*.log" -newer /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr 2>/dev/null | xargs ls -t 2>/dev/null | head -5)
echo "Candidate recent logs:"
echo "$LOG"
echo "---"
find /home/thelegendiv/finn/vivado_projects/S12_256_analytical -iname "runme.log" -o -iname "vivado*.log" | xargs ls -lt 2>/dev/null | head -10

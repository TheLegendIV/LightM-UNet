#!/bin/bash
XPR=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
echo "=== total IPRepoPath count ==="
grep -c 'IPRepoPath' "$XPR"
echo "=== all entries ==="
grep -o 'IPRepoPath" Val="[^"]*"' "$XPR"

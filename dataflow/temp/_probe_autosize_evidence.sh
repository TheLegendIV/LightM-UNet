#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
echo "--- fifosim dirs compiled with c++17 (v5 autosizer / rtlsim):"
for d in finn_build_tmp/verilator_fifosim_*; do
  if grep -q "c++17" $d/compile.sh 2>/dev/null; then echo "$d $(stat -c %y $d | cut -c1-19) results=$( [ -f $d/results.txt ] && echo yes || echo no) nmaxcount=$(grep -c maxcount $d/results.txt 2>/dev/null)"; fi
done | tail -12
echo "--- stitched onnx per build:"
ls -d finn_deployment_outputs/ratchet_*partition2_20261009_1[5-8]* | while read o; do echo "$o: $(ls $o/partition2_*_stitched.onnx 2>/dev/null | head -1)"; done
echo "--- control autosize evidence:"
grep -E "maxcount|autosiz|fifo" /tmp/ooc_p2_ratchet_ablation_finn_autofold.log | grep -vE "warn" | head -8 | cut -c1-200

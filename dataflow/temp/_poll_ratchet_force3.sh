#!/bin/bash
echo "--- run log (launch/verify lines):"; grep -E "LAUNCHED|BUILD CHECK|ooc=|refusing" /tmp/ratchet256_run_p2_force.log | cut -c1-200
echo "--- jobs:"; ps aux | grep -E "python3 finn_s12_build.py" | grep -v grep | awk '{for(i=11;i<=NF;i++) if($i=="--tag"){print $(i+1)}}'
echo "--- per-build last meaningful line:"
for f in /tmp/ooc_p2_*.log; do echo "## $(basename $f) ($(stat -c %y $f | cut -c12-19))"; grep -vE "warnings.warn|UserWarning" "$f" | tail -2 | cut -c1-220; done
echo "--- errors:"; grep -lE "verilate failed|verilator make failed|Traceback|FAILED" /tmp/ooc_p2_*.log 2>/dev/null
echo "--- verilator fifosim dirs:"; ls -dt /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/verilator_fifosim_* 2>/dev/null | head -3

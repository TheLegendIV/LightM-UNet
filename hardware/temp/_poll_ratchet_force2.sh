#!/bin/bash
echo "--- run log tail:"; tail -30 /tmp/ratchet256_run_p2_force.log | cut -c1-200
echo "--- jobs:"; ps aux | grep -E "python3 finn_s12_build.py" | grep -v grep | awk '{for(i=11;i<=NF;i++) if($i=="--tag"){print $(i+1)}}'
echo "--- build log tails:"
for f in /tmp/ooc_p2_*.log; do echo "## $f"; tail -3 "$f" | cut -c1-200; done 2>/dev/null
echo "--- autosizer/verilator errors:"; grep -lE "verilate failed|verilator make failed|Traceback" /tmp/ooc_p2_*.log 2>/dev/null

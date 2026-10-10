cd /home/thelegendiv/finn/notebooks/enet
D=finn_deployment_outputs/ratchet_25pct_simfifo_milpfold_partition2_20261009_004521
ls -la $D | head -30
echo ---- logs
ls -lat /tmp/ratchet256_* /tmp/ooc_p2_* 2>/dev/null | head -30
echo ---- LANDED / mismatch lines
grep -nE "LANDED CHECK|MISMATCH|mismatch|refusing|^---" /tmp/ratchet256_run.log /tmp/ratchet256_bridge.log 2>/dev/null | cut -c1-300 | head -60

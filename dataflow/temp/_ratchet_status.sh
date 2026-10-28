cd /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs
echo "=== ratchet partition2 dirs"
ls -d ratchet*partition2* 2>/dev/null
D=ratchet_off_simfifo_milpfold_partition2_20261009_004541
echo "=== $D"
ls -la --time-style=+%F_%T $D | head -40
echo "=== build_dataflow.log tail"
tail -n 25 $D/build_dataflow.log 2>/dev/null | cut -c1-260
echo "=== report dir"
ls -la --time-style=+%F_%T $D/report 2>/dev/null | head -20
echo "=== processes"
ps -eo pid,etime,cmd | grep -E "ratchet|run_arms|finn_s12_build|vivado|verilator" | grep -v grep | cut -c1-160
echo "=== /tmp ratchet logs"
ls -l --time-style=+%F_%T /tmp/ratchet256* 2>/dev/null | awk '{print $5, $6, $7}'

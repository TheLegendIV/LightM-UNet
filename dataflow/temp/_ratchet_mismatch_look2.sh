cd /tmp
for f in ratchet256_run_p2.log ratchet256_landed_p2_ratchet_25pct_simfifo.log ratchet256_landed_p2_ratchet_off_simfifo.log; do echo "=== $f"; cat $f | cut -c1-400; done
echo "=== bridge_p2_25pct_simfifo (tail)"; tail -25 ratchet256_bridge_p2_ratchet_25pct_simfifo.log | cut -c1-300
echo "=== bridge_p2.log LANDED lines"; grep -nE "LANDED|mismatch|MISMATCH|^---|refusing" ratchet256_bridge_p2.log | cut -c1-300 | tail -40

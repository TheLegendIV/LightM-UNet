cd /home/thelegendiv/finn/notebooks/enet
echo "=== run_p2.log"; cat /tmp/ratchet256_run_p2.log | cut -c1-250
echo "=== bridge_all.log"; cat /tmp/ratchet256_bridge_all.log | cut -c1-250
echo "=== bridge_p2_ratchet_off_simfifo.log (tail 15)"; tail -n 15 /tmp/ratchet256_bridge_p2_ratchet_off_simfifo.log | cut -c1-250
echo "=== landed_p2_ratchet_off_simfifo.log"; cat /tmp/ratchet256_landed_p2_ratchet_off_simfifo.log | cut -c1-250
echo "=== intermediate_models"; ls -la --time-style=+%F_%T finn_deployment_outputs/ratchet_off_simfifo_milpfold_partition2_20261009_004541/intermediate_models/*
echo "=== any rtlsim/ooc outputs from the simfifo arms"
ls -d finn_deployment_outputs/ratchet_*simfifo*20261009* | while read d; do echo "$d: $(find $d -type f | wc -l) files, report=$(ls $d/report 2>/dev/null | wc -l)"; done
echo "=== ooc/synth dirs"
ls -d finn_build_tmp/*ratchet*simfifo* 2>/dev/null | head; ls -lt --time-style=+%F_%T finn_build_tmp | head -5

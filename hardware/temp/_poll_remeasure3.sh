#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
O=finn_deployment_outputs
date +%H:%M:%S
echo "--- workers:"; ps aux | grep -E "_remeasure_rtlsim|finn_s12_preamble|finn_s12_build" | grep -v grep | cut -c1-200
for l in a b; do echo "--- remeasure_$l log (tail):"; grep -vE "warnings.warn|UserWarning" /tmp/remeasure_rtlsim_$l.log | tail -8 | cut -c1-220; done
echo "--- per-build rtlsim report:"
for d in $O/ratchet_*_partition2_*; do
  r=$d/report/rtlsim_performance.json
  if [ -f $r ]; then echo "## $(basename $d): $(tr -d '\n ' < $r | cut -c1-400)"; else echo "## $(basename $d): rtlsim MISSING"; fi
done
echo "--- bilinear preamble:"; tail -n 2 /tmp/preamble_bilinear.log | cut -c1-200

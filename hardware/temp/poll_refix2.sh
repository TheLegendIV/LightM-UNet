#!/bin/bash
for p in 5 6; do
  echo "== P$p"
  grep -E 'DEEP FIFO|RESULT|FAILED|Traceback|Error' /tmp/partition${p}_refix2.log | cut -c1-400
  tail -1 /tmp/partition${p}_refix2.log | cut -c1-200
  pgrep -af "rebuild_partition${p}_refix2" | head -1
  d=$(ls -d /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/*refix2 2>/dev/null | head -5)
done
ls /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/ | grep refix2
find /home/thelegendiv/finn/notebooks/enet/finn_build_tmp -maxdepth 4 -type d -name 'rtlsim_single_refix2' 2>/dev/null

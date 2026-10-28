#!/bin/bash
source /tools/Xilinx/Vivado/2022.2/settings64.sh
cd /tmp
vivado -mode batch -nojournal -log /tmp/query_synth_run_settings.log -source /tmp/_tmp_query_synth_run_settings.tcl > /dev/null 2>&1 || true
grep -E '^RUN |ERROR' /tmp/query_synth_run_settings.log | sed -E 's/STEPS.SYNTH_DESIGN.ARGS.//' | cut -c1-330 > /tmp/synth_run_settings.txt
wc -l /tmp/synth_run_settings.txt
grep -c 'RESOURCE_SHARING=off' /tmp/synth_run_settings.txt
grep -E 'GenericPartition' /tmp/synth_run_settings.txt | sed -E 's/^RUN [^ ]+ \| //' | sort | uniq -c
echo ---
grep -vE 'GenericPartition' /tmp/synth_run_settings.txt | cut -c1-330 | head -12

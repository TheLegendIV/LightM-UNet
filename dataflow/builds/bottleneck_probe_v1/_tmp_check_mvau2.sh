#!/bin/bash
for d in dn_cin16_cout32_in64_int4_mvau_merged_20261003_212937 \
         dn_cin16_cout32_in64_int6_mvau_merged_20261003_213757 \
         dn_cin16_cout32_in64_int8_mvau_merged_20261003_213830 \
         dn_cin16_cout32_in64_int4_mvau_pool_merged_20261003_202519; do
  echo "=== $d ==="
  python3 -c "
import json
d = json.load(open('/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/$d/probe_result.json'))
rt = d['stages']['rtlsim']
print('steady_cyc_per_pixel', rt.get('steady_cyc_per_pixel'), 'target', rt.get('target_cyc_per_pixel'))
ooc = d['stages'].get('ooc')
print('ooc:', ooc)
"
done

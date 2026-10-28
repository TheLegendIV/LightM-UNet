#!/bin/bash
for d in dn_cin16_cout32_in64_int4_mvau_merged_20261003_212937 \
         dn_cin16_cout32_in64_int6_mvau_merged_20261003_213757 \
         dn_cin16_cout32_in64_int8_mvau_merged_20261003_213830 \
         dn_cin16_cout32_in64_int4_mvau_pool_merged_20261003_184441 \
         dn_cin16_cout32_in64_int4_mvau_pool_merged_20261003_184703 \
         dn_cin16_cout32_in64_int4_mvau_pool_merged_20261003_185655 \
         dn_cin16_cout32_in64_int4_mvau_pool_merged_20261003_202519; do
  echo "=== $d ==="
  python3 -c "
import json
d = json.load(open('/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/$d/probe_result.json'))
print('stages:', sorted(d['stages'].keys()))
rt = d['stages'].get('rtlsim')
if rt:
    print('rtlsim keys:', list(rt.keys()) if isinstance(rt, dict) else rt)
"
done

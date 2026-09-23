#!/bin/bash
echo "--- StreamingConcat mentions in relaunch log ---"
grep -n "StreamingConcat" /tmp/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_relaunch.log
echo "--- component.xml check (latest StreamingConcat_hls_0 dir) ---"
d=$(ls -dt /tmp/finn_dev_thelegendiv/code_gen_ipgen_StreamingConcat_hls_0_* 2>/dev/null | head -1)
echo "latest dir: $d"
find "$d" -iname component.xml 2>/dev/null
echo "--- retry-helper messages (if any) ---"
grep -n "_validate_and_retry_hls_ipgen" /tmp/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_relaunch.log

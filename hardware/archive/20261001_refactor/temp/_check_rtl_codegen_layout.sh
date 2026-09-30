#!/bin/bash
d=$(ls -d /tmp/finn_dev_thelegendiv/code_gen_ipgen_Thresholding_rtl_0_* | head -1)
echo "DIR: $d"
find "$d" -maxdepth 2 -type f
echo "--- StreamingFIFO_rtl example ---"
d2=$(ls -d /tmp/finn_dev_thelegendiv/code_gen_ipgen_StreamingFIFO_rtl_9_* | head -1)
echo "DIR2: $d2"
find "$d2" -maxdepth 2 -type f

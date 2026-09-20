#!/bin/bash
echo "--- MVAU_rtl_0.v diff (p0 vs p1) ---"
diff /tmp/finn_dev_thelegendiv/code_gen_ipgen_MVAU_rtl_0_eo10d7_d/MVAU_rtl_0.v /tmp/finn_dev_thelegendiv/code_gen_ipgen_MVAU_rtl_0_njzdpdz9/MVAU_rtl_0.v
echo "diff exit: $?"
echo "--- StreamingFIFO_rtl_0.v diff (p0 vs p1) ---"
diff /tmp/finn_dev_thelegendiv/code_gen_ipgen_StreamingFIFO_rtl_0_9u6w_e6m/StreamingFIFO_rtl_0.v /tmp/finn_dev_thelegendiv/code_gen_ipgen_StreamingFIFO_rtl_0_0ppp80qt/StreamingFIFO_rtl_0.v
echo "diff exit: $?"
echo "--- MVAU_rtl_0 memory init files (weights) ---"
find /tmp/finn_dev_thelegendiv/code_gen_ipgen_MVAU_rtl_0_eo10d7_d -iname "*.dat" -o -iname "*.mif" -o -iname "*memblock*"
find /tmp/finn_dev_thelegendiv/code_gen_ipgen_MVAU_rtl_0_njzdpdz9 -iname "*.dat" -o -iname "*.mif" -o -iname "*memblock*"

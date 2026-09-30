#!/bin/bash
echo "--- wrapper file diff ---"
diff /tmp/finn_dev_thelegendiv/code_gen_ipgen_MVAU_rtl_0_eo10d7_d/MVAU_rtl_0_wrapper.v /tmp/finn_dev_thelegendiv/code_gen_ipgen_MVAU_rtl_0_njzdpdz9/MVAU_rtl_0_wrapper.v
echo "wrapper diff exit: $?"
echo "--- memblock.dat sizes ---"
wc -l /tmp/finn_dev_thelegendiv/code_gen_ipgen_MVAU_rtl_0_eo10d7_d/memblock.dat /tmp/finn_dev_thelegendiv/code_gen_ipgen_MVAU_rtl_0_njzdpdz9/memblock.dat
echo "--- memblock.dat diff (first mismatch) ---"
diff /tmp/finn_dev_thelegendiv/code_gen_ipgen_MVAU_rtl_0_eo10d7_d/memblock.dat /tmp/finn_dev_thelegendiv/code_gen_ipgen_MVAU_rtl_0_njzdpdz9/memblock.dat | head -5
echo "memblock diff exit: $?"
echo "--- is memblock.dat referenced by a FIXED relative filename inside the wrapper .v (readmem)? ---"
grep -n "readmem\|memblock" /tmp/finn_dev_thelegendiv/code_gen_ipgen_MVAU_rtl_0_eo10d7_d/MVAU_rtl_0_wrapper.v

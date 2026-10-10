cd /home/thelegendiv/finn
git diff HEAD FETCH_HEAD -- finn-rtllib/mvu/mvu_vvu_axi_wrapper.v > /tmp/alpha_wrapper.diff
git diff HEAD FETCH_HEAD -- src/finn/custom_op/fpgadataflow/rtl/matrixvectoractivation_rtl.py > /tmp/alpha_mvau_rtl.diff
git diff HEAD FETCH_HEAD -- src/finn/custom_op/fpgadataflow/rtl/vectorvectoractivation_rtl.py > /tmp/alpha_vvau_rtl.diff
git diff HEAD FETCH_HEAD -- finn-rtllib/mvu/mvu_vvu_axi.sv > /tmp/alpha_axi.diff
# upstream files to a scratch dir
rm -rf /tmp/alpha_rtllib && mkdir -p /tmp/alpha_rtllib
git archive FETCH_HEAD finn-rtllib/mvu | tar -x -C /tmp/alpha_rtllib
wc -l /tmp/alpha_*.diff

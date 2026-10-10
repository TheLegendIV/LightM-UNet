set -e
cd /home/thelegendiv/finn
git tag -f pre_mvu_alpha HEAD
# take the upstream rtllib/mvu tree, but keep our wrapper template
cp finn-rtllib/mvu/mvu_vvu_axi_wrapper.v /tmp/mvu_vvu_axi_wrapper.v.local
git checkout FETCH_HEAD -- finn-rtllib/mvu
git rm -q -f finn-rtllib/mvu/mvu_4sx4u.sv finn-rtllib/mvu/mvu_8sx8u_dsp48.sv \
  finn-rtllib/mvu/tb/mvu_8sx9_tb.sv finn-rtllib/mvu/tb/mvu_dsp58_tb.sv 2>&1 || true
cp /tmp/mvu_vvu_axi_wrapper.v.local finn-rtllib/mvu/mvu_vvu_axi_wrapper.v
python3 /tmp/_apply_mvu_alpha_py.py
git add finn-rtllib/mvu src/finn/custom_op/fpgadataflow/rtl/matrixvectoractivation_rtl.py src/finn/custom_op/fpgadataflow/rtl/vectorvectoractivation_rtl.py
git status --short | grep -E "^(A|M|D|R)" 

cd /home/thelegendiv/finn
F=src/finn/custom_op/fpgadataflow/rtl
echo "--- MVAU_rtl"
sed -n 145,165p $F/matrixvectoractivation_rtl.py
sed -n 196,232p $F/matrixvectoractivation_rtl.py
grep -n "COMPUTE_CORE\|_resolve_impl_style\|FORCE_BEHAVIORAL\|verilog_paths\|_wrapper_sim" $F/matrixvectoractivation_rtl.py
echo "--- VVAU_rtl"
sed -n 150,165p $F/vectorvectoractivation_rtl.py
grep -n "COMPUTE_CORE\|_resolve_impl_style\|FORCE_BEHAVIORAL" $F/vectorvectoractivation_rtl.py
echo "--- wrapper"
sed -n 30,100p finn-rtllib/mvu/mvu_vvu_axi_wrapper.v | cat -A | grep -n "COMPUTE_CORE\|PUMPED\|FORCE" 
echo "--- other refs in repo"
git grep -n "mvu_8sx8u\|mvu_4sx4u\|COMPUTE_CORE" -- . ':!finn-rtllib/mvu' | head -20

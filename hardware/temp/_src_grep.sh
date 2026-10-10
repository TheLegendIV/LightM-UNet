cd /home/thelegendiv/finn/src/finn/custom_op/fpgadataflow/rtl
wc -l matrixvectoractivation_rtl.py thresholding_rtl.py
echo == thr
grep -n -E "def |\.dat|memstream|thresholds|ActVal" thresholding_rtl.py | head -70
echo == mvau
grep -n -E "def |\.dat|memstream|wstrm" matrixvectoractivation_rtl.py | head -50

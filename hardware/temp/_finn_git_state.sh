cd /home/thelegendiv/finn
git remote -v
git status --short | head -20
echo --- tags
git tag | tail -5
echo --- network
timeout 20 git ls-remote --tags https://github.com/Xilinx/finn.git v1.0.0-alpha 2>&1 | head -3
echo --- py refs
grep -rn "mvu_8sx8u_dsp48\|mvu_4sx4u\|mvu_vvu_8sx9\|mvu_vvu_axi" src/finn --include=*.py | head -40
echo --- vivado
ls /tools/Xilinx/Vivado /tools/Xilinx/Vitis_HLS

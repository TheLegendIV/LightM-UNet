set -e
export HOME=/tmp/home_dir
source /tools/Xilinx/Vivado/2022.2/settings64.sh
D=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200/GenericPartition_7
W=$(ls -d $D/vivado_stitch_proj_*)/GenericPartition_7_wrapper.v
O=/tmp/p7_patched
rm -rf $O; mkdir -p $O/build
python3 - "$W" "$O/GenericPartition_7_wrapper.v" <<'EOF'
import sys
s = open(sys.argv[1]).read()
old = "if(zero)     B1  <= 0;\n"
n = s.count("if(zero)     B1  <= 0;")
print("occurrences", n)
import re
pat = re.compile(r"if\(zero\)(\s+)B1  <= 0;\s*\n(\s*)else if\(en\)(\s+)B1  <= bb;")
s2, k = pat.subn("if(en)   B1  <= zero ? 0 : bb;", s)
print("replaced", k)
assert k == n
open(sys.argv[2], "w").write(s2)
EOF
sed -n '/^#!/!p' $D/rtlsim_single/compile.sh > $O/c.sh
python3 - "$D" "$O" "$W" <<'EOF'
import re, sys
D, O, W = sys.argv[1:4]
lines = [l for l in open(f"{O}/c.sh").read().splitlines() if l.strip()]
vl = re.sub(r"-Mdir \S+", f"-Mdir {O}/build", lines[0])
vl = re.sub(r"verilator_fifosim_\w+\.cpp", f"{D}/rtlsim_taps/tap_tb.cpp", vl)
vl = vl.replace(W, f"{O}/GenericPartition_7_wrapper.v")
vl = vl.replace("-Wno-fatal", "-Wno-fatal --public-flat-rw", 1)
mk = re.sub(r"-j\d+", "-j12", lines[1])
open(f"{O}/run.sh", "w").write(vl + "\n" + mk + "\n")
EOF
cd $O/build
bash $O/run.sh > $O/build.log 2>&1
mkdir -p $O/taps
NIN=$(grep N_IN_TXNS $D/rtlsim_single/results.txt | cut -f2)
NOUT=$(grep N_OUT_TXNS $D/rtlsim_single/results.txt | cut -f2)
echo "NIN=$NIN NOUT=$NOUT"
$O/build/VGenericPartition_7_wrapper /tmp/golden_pp/case0/p7_in.raw $O/taps/top_out.raw $NIN $NOUT 100 100 1 3000000 $O/taps > $O/run.log 2>&1 || true
cp /tmp/taps_p7/taps.json $O/taps/taps.json
echo RUN_DONE

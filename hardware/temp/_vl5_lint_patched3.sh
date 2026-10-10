unset VERILATOR_ROOT
V5=/tmp/home_dir/.local/lib/python3.10/site-packages/verilator/bin/verilator
python3 /tmp/_patch_add_multi3.py /tmp/alpha_rtllib/finn-rtllib/mvu/add_multi.sv || exit 1
sed -n 48,70p /tmp/alpha_rtllib/finn-rtllib/mvu/add_multi.sv
W=/tmp/alpha_try
SRC=/tmp/alpha_rtllib/finn-rtllib/mvu
FILES="$SRC/mvu_pkg.sv $SRC/add_multi.sv $SRC/replay_buffer.sv $SRC/mvu.sv $SRC/mvu_vvu_8sx9_dsp58.sv $SRC/mvu_vvu_axi.sv"
n=2
top=$(grep -m1 -oE "^module [A-Za-z0-9_]+" $W/mvau${n}_wrapper.v | awk '{print $2}')
for v in $V5 /home/thelegendiv/tools/verilator5/bin/verilator; do
  echo "== mvau$n  $($v --version | cut -c1-30)"
  $v --lint-only -Wno-fatal -Wno-lint -Wno-style -Wno-STMTDLY --top-module $top $FILES $W/mvau${n}_wrapper_sim.v 2>&1 | grep -E "Error|error" | head -8
done
echo done

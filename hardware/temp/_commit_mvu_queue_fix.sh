set -e
cd /home/thelegendiv/finn
F=finn-rtllib/mvu/mvu_vvu_axi.sv
sed -i 's|// This is conservative and could be divided by a guaranteed minimum output interval, e.g. MW/SIMD.|// This is conservative and could be divided by a guaranteed minimum output interval, e.g. MW/SIMD.\n\t\t// +1: the registered OLock reacts one cycle after the first stalled output, so one more result than\n\t\t// the core pipeline depth can arrive; without it the queue overflows for SIMD=1 and MW=SIMD (output every cycle).|' $F
sed -i 's|localparam int unsigned  MAX_IN_FLIGHT = CORE_PIPELINE_DEPTH;|localparam int unsigned  MAX_IN_FLIGHT = CORE_PIPELINE_DEPTH + 1;|' $F
git diff $F
git add $F
git -c user.name=agent -c user.email=agent@local commit -q -m "mvu_vvu_axi: output queue one entry deeper (overflow under backpressure at output interval of 1 cycle)"
git log --oneline | head -3

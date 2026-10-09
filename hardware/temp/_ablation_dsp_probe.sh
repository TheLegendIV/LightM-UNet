#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet/finn_build_tmp
for s in h4m8tknc xtvm427_ o5mp0f8f jw16vngf qyg79oac 3e851e8z; do
  d=synth_out_of_context_$s
  f=$(find $d -name "vivado.log" -o -name "*.rpt" -path "*util*" 2>/dev/null | head -3 | tr '\n' ' ')
  echo "## $s: $f"
  for x in $(find $d -name vivado.log 2>/dev/null | head -1); do grep -E "^\| *(DSPs|CLB LUTs|Block RAM Tile|URAM) " $x | tail -4; done
  [ -z "$(find $d -name vivado.log 2>/dev/null)" ] && ls $d | head -5
done

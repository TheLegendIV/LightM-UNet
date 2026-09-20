#!/bin/bash
# Verify the 8 candidate partition dirs are pure kernel-only stitched IPs
# (no embedded PS7/PS8, and check whether their internal child IPs are
# already partition-prefixed or still generic/bare).
BASE=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
DIRS="vivado_stitch_proj_6_75nd4k vivado_stitch_proj_4ti661rj vivado_stitch_proj_e_z5561z vivado_stitch_proj_9qynp0ng vivado_stitch_proj_ohvum343 vivado_stitch_proj_anxxap8z vivado_stitch_proj_5kxs5a6b vivado_stitch_proj_btoactkv"

for d in $DIRS; do
  p="$BASE/$d"
  echo "===== $d ====="
  top_xml="$p/ip/component.xml"
  if [ -f "$top_xml" ]; then
    echo "top-level name: $(grep -m1 -oP '(?<=<spirit:name>)[^<]+' "$top_xml")"
    echo "top-level vlnv: $(grep -m1 -oP '(?<=<spirit:library>)[^<]+' "$top_xml")/$(grep -m1 -oP '(?<=<spirit:name>)[^<]+' "$top_xml")/$(grep -m1 -oP '(?<=<spirit:version>)[^<]+' "$top_xml")"
  else
    echo "MISSING $top_xml"
  fi

  # embedded PS check (a pure stitched IP should have NO zynq_ultra_ps_e cell)
  ps_hits=$(grep -rl "zynq_ultra_ps_e" "$p" 2>/dev/null | wc -l)
  echo "zynq_ultra_ps_e references found: $ps_hits (expect 0 for a pure stitched IP)"

  # every nested child component.xml's own name -> reveals prefixed vs bare
  echo "-- nested child component names --"
  find "$p" -path "*/ip/*/component.xml" 2>/dev/null | while read -r cx; do
    nm=$(grep -m1 -oP '(?<=<spirit:name>)[^<]+' "$cx")
    echo "  $nm"
  done | sort -u
  echo ""
done

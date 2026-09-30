#!/bin/bash
echo "--- appstore now ---"
find /tmp/home_dir/.Xilinx -maxdepth 4 | sort
echo "--- manifest.tcl content ---"
cat /tmp/home_dir/.Xilinx/Vivado/tclapp/manifest.tcl 2>&1 | head -20
echo "--- checking ALL code_gen_ipgen dirs from this run for missing component.xml ---"
for d in /tmp/finn_dev_thelegendiv/code_gen_ipgen_*; do
  if [ -d "$d" ]; then
    n=$(find "$d" -iname component.xml | wc -l)
    echo "$d : component.xml count=$n"
  fi
done

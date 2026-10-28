#!/bin/bash
cd /home/thelegendiv/finn/src/finn/custom_op/fpgadataflow
for f in matrixvectoractivation thresholding convolutioninputgenerator addstreams duplicatestreams fmpadding streamingdatawidthconverter streamingfifo streamingmaxpool; do
  echo "=== $f"
  grep -n "def execute_node" -A14 $f.py | cut -c1-140
done

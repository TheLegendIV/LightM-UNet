#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
nohup python3 -u run_variant_matrix.py 4 > /home/thelegendiv/finn/notebooks/enet/mvau_variant_matrix_run.log 2>&1 &
echo "launched pid $!"

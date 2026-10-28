#!/bin/bash
export HOME=/tmp/home_dir
unset VERILATOR_ROOT
cd /home/thelegendiv/finn/notebooks/enet
bash -n run_arms.sh && echo "syntax-ok"
python3 finn_s12_build.py --help 2>&1 | tail -3
python3 - <<'EOF'
import sys
sys.path.insert(0, ".")
import finn.transformation.fpgadataflow.set_fifo_depths as sfd
import finn_partition_build_steps as m
sfd.verilator_fifosim = m.verilator_fifosim_v5
print("patched:", sfd.verilator_fifosim.__module__, sfd.verilator_fifosim.__name__)
EOF

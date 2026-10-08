#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
nohup python3 rerun_combine_rtlsim.py > /tmp/recombine_v5sim.log 2>&1 &

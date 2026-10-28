#!/bin/bash
cd /mnt/c/DEV/repos/LightM-UNet/hardware/builds/bottleneck_probe_v1
export CONTAINER=finn_persistent
export JOBS=1
export EXTRA="--ooc"
export CASES="init_cin1_cout4_in256_int4_pool"
exec bash run_probes.sh

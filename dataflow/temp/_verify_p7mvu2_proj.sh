#!/bin/bash
L=/tmp/make_proj_p7mvu2.log
grep -c MAKE_PROJ_P7MVU2_DONE $L
P=/home/thelegendiv/finn/vivado_projects/S12_256_analytical_p7mvu2/s12_256_analytical_p7mvu2
ls -la $P/*.xpr
grep -a IPRepoPath $P/s12_256_analytical_p7mvu2.xpr | grep -c 'u8in_int6_fps250_lat200/GenericPartition_7'
grep -a IPRepoPath $P/s12_256_analytical_p7mvu2.xpr | grep -c 'u8in_int6_p7mvu2/GenericPartition_7'
grep -a IPRepoPath $P/s12_256_analytical_p7mvu2.xpr | head -2 | cut -c1-160

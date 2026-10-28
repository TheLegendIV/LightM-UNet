#!/bin/bash
T=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_512_u4_analytical_v1_ft15ep_int6_fps250_lat200
echo "== IODMA dirs"; ls -d $T/GenericPartition_*/code_gen_ipgen_IODMA* 
echo "== stitch dirs"; ls -d $T/GenericPartition_*/vivado_stitch_proj_*
echo "== top of T"; ls $T | head
echo "== iodma verilog dirs"; for d in $T/GenericPartition_*/code_gen_ipgen_IODMA*; do echo $d; ls $d/project_IODMA_hls_0/sol1/impl/verilog | head -20; ls $d/project_IODMA_hls_0/sol1/impl/ip | head; done
echo "== proj dirs mtime"
ls -ld --time-style=long-iso /home/thelegendiv/finn/vivado_projects/*/* | cut -c1-200
for p in S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr S12_256_analytical_p7mvu2/s12_256_analytical_p7mvu2/s12_256_analytical_p7mvu2.xpr; do
  echo "== $p"; ls -l --time-style=long-iso /home/thelegendiv/finn/vivado_projects/$p
  grep -o 'finn_build_tmp/[A-Za-z0-9_]*' /home/thelegendiv/finn/vivado_projects/$p | sort | uniq -c
done

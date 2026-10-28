P=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical
ls -d --time-style=full-iso -l $P/s12_256_analytical.runs/*GenericPartition_1* 2>&1 | cut -c1-200
grep -o 'Run Id="[^"]*GenericPartition_1[^"]*"[^>]*' $P/s12_256_analytical.xpr | cut -c1-200
ls -l --time-style=full-iso $P/s12_256_analytical.runs/top_GenericPartition_1_0_0_synth_1 2>&1 | head -8

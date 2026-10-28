P=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical
ls -1 $P/s12_256_analytical.runs
echo ---
grep -o 'Run Id="[^"]*"' $P/s12_256_analytical.xpr | head -30

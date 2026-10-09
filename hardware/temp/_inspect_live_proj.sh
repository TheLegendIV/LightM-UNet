P=/home/thelegendiv/finn/vivado_projects/S12_256_analytical
ls -la $P $P/s12_256_analytical | head -40
echo ---- ip_repo_paths
grep -o 'IPRepoPath[^/]*Path="[^"]*"' $P/s12_256_analytical/s12_256_analytical.xpr | head -50
grep -o 'Option Name="IPRepoPath"[^>]*>' $P/s12_256_analytical/s12_256_analytical.xpr | head
echo ---- bd cells
grep -o '"VLNV": "[^"]*"' $P/s12_256_analytical/s12_256_analytical.srcs/sources_1/bd/top/top.bd | sort | uniq -c
echo ---- ps
ps aux | grep -i vivado | grep -v grep | cut -c1-200

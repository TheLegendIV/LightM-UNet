P=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical
cd "$P" || exit 1
echo "-- timestamps"
ls -l --time-style=+%F_%T ip_upgrade.log top_wrapper*.xsa s12_256_analytical.xpr 2>/dev/null | cut -c30-140
ls -l --time-style=+%F_%T s12_256_analytical.runs/synth_1/*.dcp s12_256_analytical.runs/synth_1/runme.log s12_256_analytical.runs/synth_1/.vivado.begin.rst s12_256_analytical.runs/impl_1/*.bit s12_256_analytical.runs/impl_1/*routed.dcp 2>/dev/null | cut -c30-170
echo "-- ip_upgrade.log tail"
tail -5 ip_upgrade.log
echo "-- build dirs referenced in synth_1/runme.log"
grep -o -E '2026[0-9]{4}_[0-9]{6}' s12_256_analytical.runs/synth_1/runme.log | sort | uniq -c
echo "-- build dirs referenced in the .xpr ip_repo_paths"
grep -o -E 'S12_dense_256[A-Za-z0-9_]*_20[0-9]{6}_[0-9]{6}' s12_256_analytical.xpr | sort | uniq -c | head
echo "-- build dirs referenced by the 8 GenericPartition cells (xci)"
find s12_256_analytical.srcs -name 'GenericPartition_*.xci' -o -name 'top_GenericPartition_*_0.xci' | head -3
grep -rho -E 'S12_dense_256[A-Za-z0-9_]*_20[0-9]{6}_[0-9]{6}' s12_256_analytical.gen 2>/dev/null | sort | uniq -c | head

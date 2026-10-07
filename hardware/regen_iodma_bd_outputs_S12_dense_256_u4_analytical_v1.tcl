# STEP 2 of 2: regenerates BD output products for both IODMA cells (now
# repackaged with collision-free module names by
# fix_iodma_rtl_module_collision_S12_dense_256_u4_analytical_v1.tcl) and
# re-runs the top-level synth_1 run, where the module-name collision was
# actually observed (both IODMA cells get elaborated inline as part of
# synth_1 -- they have no separate per-cell OOC run directory, unlike the
# 8 GenericPartition cells which are FINN's own pre-synthesized
# checkpoints). Does NOT re-run impl_1 -- check the synth_1 result for
# the Synth 8-448 errors first before committing to a full re-implement.

set PROJ /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
open_project $PROJ

update_ip_catalog -rebuild

# Nested BD sub-design IP can only be reset/regenerated through their
# parent BD file, not directly by their own .xci (Vivado 12-3564).
set bd_file [get_files top.bd]
puts "regenerating output products for $bd_file (covers both IODMA cells)"
reset_target {all} $bd_file
generate_target {all} $bd_file -force

reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1

set status [get_property STATUS [get_runs synth_1]]
set progress [get_property PROGRESS [get_runs synth_1]]
puts "SYNTH_1_STATUS: $status ($progress)"

close_project

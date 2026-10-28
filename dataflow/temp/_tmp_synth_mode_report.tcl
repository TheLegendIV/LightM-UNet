open_checkpoint /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.runs/impl_1/top_wrapper_routed.dcp
set fp [open /tmp/synth_mode_report.txt w]
foreach n {0 1 2 3 4 5 6 7} {
    set cell "top_i/GenericPartition_${n}_0"
    puts $fp "=== $cell ==="
    catch {puts $fp "  HD.PARTITION=[get_property HD.PARTITION [get_cells $cell]]"}
}
close $fp
exit

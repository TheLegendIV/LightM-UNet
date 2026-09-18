open_project /tmp/finn_dev_thelegendiv/vivado_stitch_proj_d9ohtyc0/finn_vivado_stitch_proj.xpr

set bd_file [get_files GenericPartition_2.bd]
open_bd_design $bd_file

set fifo_names {StreamingFIFO_rtl_3 StreamingFIFO_rtl_6 StreamingFIFO_rtl_12 StreamingFIFO_rtl_28 StreamingFIFO_rtl_34 StreamingFIFO_rtl_50 StreamingFIFO_rtl_55 StreamingFIFO_rtl_56 StreamingFIFO_rtl_69 StreamingFIFO_rtl_72 StreamingFIFO_rtl_77 StreamingFIFO_rtl_78 StreamingFIFO_rtl_94 StreamingFIFO_rtl_100}

set n_set 0
foreach n $fifo_names {
    set cellpath "/$n/fifo"
    set cell [get_bd_cells -quiet $cellpath]
    if {$cell eq ""} {
        puts "WARNING: cell not found for $cellpath"
        continue
    }
    set_property CONFIG.FIFO_MEMORY_TYPE {ultra} $cell
    puts "SET $cellpath FIFO_MEMORY_TYPE=ultra"
    incr n_set
}
puts "TOTAL_SET=$n_set"

validate_bd_design
save_bd_design

generate_target all [get_files GenericPartition_2.bd] -force
export_ip_user_files -of_objects [get_files GenericPartition_2.bd] -no_script -sync -force -quiet

# regenerate the all_verilog_srcs.txt manifest style file list is unchanged (paths same),
# but re-list synth outputs to confirm they were touched
close_project
puts "DONE_PATCH_FIFO_URAM"

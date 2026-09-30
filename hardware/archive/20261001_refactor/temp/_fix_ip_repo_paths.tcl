set proj_path /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_zynq_proj_ivzy_6_z/finn_zynq_link.xpr
open_project $proj_path

set fp [open /tmp/_resolved_repo_dirs.txt r]
set new_dirs [split [string trim [read $fp]] "\n"]
close $fp
puts "Adding [llength $new_dirs] repo dirs"

set cur [get_property ip_repo_paths [current_project]]
set combined [concat $cur $new_dirs]
set combined [lsort -unique $combined]
set_property ip_repo_paths $combined [current_project]
update_ip_catalog -rebuild

open_bd_design [get_files top.bd]

puts "=== report_ip_status (after repo-path fix) ==="
report_ip_status -file /tmp/ip_status_after_fix.rpt
set fp2 [open /tmp/ip_status_after_fix.rpt r]
set content [read $fp2]
close $fp2
# only print lines that are interesting (not up-to-date)
foreach line [split $content "\n"] {
    if {[string match "*not found*" $line] || [string match "*locked*" $line] || [string match "*Instance Name*" $line]} {
        puts $line
    }
}
puts "=== end report ==="

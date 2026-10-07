set fp [open "/tmp/ip_repo_paths_list.txt" r]
set data [read $fp]
close $fp
set new_paths [list]
foreach line [split $data "\n"] {
    set line [string trim $line]
    if {$line ne ""} {
        lappend new_paths $line
    }
}
puts "Collected [llength $new_paths] candidate ip_repo_paths entries"

open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr

set existing [get_property ip_repo_paths [current_project]]
puts "Existing ip_repo_paths: $existing"

set all_paths $existing
foreach p $new_paths {
    if {[lsearch -exact $all_paths $p] == -1} {
        lappend all_paths $p
    }
}

set_property ip_repo_paths $all_paths [current_project]
update_ip_catalog -rebuild -scan_changes

puts "Final ip_repo_paths ([llength $all_paths] entries):"
foreach p $all_paths { puts "  $p" }

close_project

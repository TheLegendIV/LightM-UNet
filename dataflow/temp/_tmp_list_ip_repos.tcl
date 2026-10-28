open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
set paths [get_property ip_repo_paths [current_project]]
puts "COUNT=[llength $paths]"
set fp [open /tmp/live_ip_repo_paths.txt w]
foreach p $paths { puts $fp $p }
close $fp
puts "BD_FILES: [get_files -quiet *.bd]"
close_project

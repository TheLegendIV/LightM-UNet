open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
puts "=== CURRENT ip_repo_paths (count + first few) ==="
set paths [get_property ip_repo_paths [current_project]]
puts "COUNT=[llength $paths]"
foreach p [lrange $paths 0 4] { puts "  $p" }
puts "=== does path list include a DuplicateStreams_hls_0 entry for p0? ==="
puts [lsearch -all -inline $paths "*DuplicateStreams_hls*"]
puts "=== IP_REPO_PATHS_QUERY_DONE ==="

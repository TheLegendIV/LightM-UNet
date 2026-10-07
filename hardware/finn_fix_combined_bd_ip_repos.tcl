# Reusable: registers ip_repo_paths for a combined multi-partition FINN BD and
# (re)generates all its IP output products. Supersedes the per-build, hardcoded,
# INCOMPLETE `add_ip_repos_*.tcl` scripts -- those only registered each
# partition's own top-level packaged `ip/` dir (plus, inconsistently, one or
# two raw IODMA code_gen dirs), which is NOT enough: Vivado's generate_target
# also needs, as SEPARATE ip_repo_paths entries, every HLS-backend child
# node's own standalone exported ip dir, and FINN's custom
# `amd.com:finn:memstream:1.0` weight-memory IP (finn-rtllib/memstream) --
# without these, generate_target fails with
# "Cannot upgrade to invalid target ''" on the first HLS child it can't
# resolve. See /memories/repo/finn_gotchas.md entry dated 2026-10-06 for the
# full root-cause writeup. Confirmed working end-to-end (zero errors, all N
# partitions regenerated) on the 8-partition S12_256_analytical combined BD.
#
# Usage (run after open_project, with the target .bd already open or
# openable via BD_FILE_GLOB):
#   vivado -mode batch -source finn_fix_combined_bd_ip_repos.tcl -tclargs \
#       <finn_build_dir> [bd_file_glob] [finn_root]
#
# <finn_build_dir>: the FINN builder's finn_build_tmp/<build_name> dir
#                   containing GenericPartition_0, GenericPartition_1, ...
# [bd_file_glob]:   glob pattern (relative to the project) identifying the
#                   combined .bd file, default "*/top.bd"
# [finn_root]:      FINN_ROOT dir containing finn-rtllib/memstream, default
#                   $::env(FINN_ROOT) if set, else /home/thelegendiv/finn

if {[llength $argv] < 1} {
    error "usage: -tclargs <finn_build_dir> \[bd_file_glob\] \[finn_root\]"
}
set finn_build [lindex $argv 0]
set bd_file_glob [expr {[llength $argv] >= 2 ? [lindex $argv 1] : "*/top.bd"}]
if {[llength $argv] >= 3} {
    set finn_root [lindex $argv 2]
} elseif {[info exists ::env(FINN_ROOT)]} {
    set finn_root $::env(FINN_ROOT)
} else {
    set finn_root /home/thelegendiv/finn
}

if {![file isdirectory $finn_build]} {
    error "finn_build_dir not found: $finn_build"
}

# 1) each partition's own packaged top-level stitched-IP dir
set ip_paths {}
foreach part_dir [lsort [glob -nocomplain -type d $finn_build/GenericPartition_*]] {
    foreach d [glob -nocomplain -type d $part_dir/vivado_stitch_proj_*/ip] {
        lappend ip_paths $d
    }
}
# 2) every HLS-backend child node's own standalone exported ip dir
foreach part_dir [lsort [glob -nocomplain -type d $finn_build/GenericPartition_*]] {
    foreach d [glob -nocomplain -type d $part_dir/code_gen_ipgen_*_hls_*/*/sol1/impl/ip] {
        lappend ip_paths $d
    }
}
# 3) FINN's own custom weight-memory IP (shared across all partitions)
set memstream_dir $finn_root/finn-rtllib/memstream
if {[file isdirectory $memstream_dir]} {
    lappend ip_paths $memstream_dir
} else {
    puts "WARNING: memstream dir not found, skipping: $memstream_dir"
}
set ip_paths [lsort -unique $ip_paths]

puts "=== setting [llength $ip_paths] ip_repo_paths entries from $finn_build ==="
set_property ip_repo_paths $ip_paths [current_project]
update_ip_catalog -rebuild -scan_changes

set bd_files [get_files -quiet $bd_file_glob]
if {$bd_files eq ""} {
    error "no .bd file matched glob '$bd_file_glob' in the current project"
}
open_bd_design $bd_files
validate_bd_design -force
generate_target all $bd_files -force

puts "=== finn_fix_combined_bd_ip_repos.tcl: done, [llength $ip_paths] ip_repo_paths registered ==="

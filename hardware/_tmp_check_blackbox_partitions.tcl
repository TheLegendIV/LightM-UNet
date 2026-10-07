open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr

puts "=== RUNS ==="
foreach r [get_runs] {
    puts "$r: progress=[get_property PROGRESS $r] status=[get_property STATUS $r] is_synthesis=[get_property IS_SYNTHESIS $r]"
}

puts "=== OPENING impl_1 ==="
open_run impl_1

puts "=== CELL COUNT PER PARTITION (hierarchical) ==="
for {set i 0} {$i < 8} {incr i} {
    set cells [get_cells -hierarchical -filter "NAME =~ *GenericPartition_${i}_0*" -quiet]
    puts "GenericPartition_$i : cell_count=[llength $cells]"
}

puts "=== BLACKBOX CHECK ==="
set bb [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ "*BLACKBOX*" || IS_BLACKBOX} -quiet]
puts "blackbox cells found: [llength $bb]"
foreach c $bb { puts "  BLACKBOX: $c" }

puts "=== STUB CHECK ON BD CELLS ==="
foreach c [get_bd_cells -hierarchical -filter {VLNV =~ "*GenericPartition*"} -quiet] {
    puts "$c : [get_property CONFIG.SYNTH_CHECKPOINT_MODE $c -quiet]"
}

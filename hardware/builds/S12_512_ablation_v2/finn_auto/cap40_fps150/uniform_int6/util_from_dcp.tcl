set dcp [lindex $argv 0]
set out [lindex $argv 1]
open_checkpoint $dcp
set util [report_utilization -return_string]
set util_lut [exec echo $util | grep LUT | head -n 1 | cut -d| -f3 | tr -d " "]
set util_lutram [exec echo $util | grep LUT | head -n 3 | tail -1 | cut -d| -f3 | tr -d " "]
set util_ff [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ *.F*E*}]]
set util_dsp [llength [get_cells -hier -filter {PRIMITIVE_GROUP == DSP}]]
set util_bram18 [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ *RAMB18*}]]
set util_bram36 [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ *RAMB36*}]]
set util_carry [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ *CARRY*}]]
set util_uram [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ *URAM*}]]
set vivado_version [version -short]
set vivado_build_no [exec echo $util | grep Build | head -n 1 | cut -d\  -f9]
set util_bram [expr $util_bram18/2 + $util_bram36]
set fp [open $out w]
puts $fp "LUT=$util_lut"
puts $fp "LUTRAM=$util_lutram"
puts $fp "FF=$util_ff"
puts $fp "DSP=$util_dsp"
puts $fp "BRAM=$util_bram"
puts $fp "BRAM_18K=$util_bram18"
puts $fp "BRAM_36K=$util_bram36"
puts $fp "URAM=$util_uram"
puts $fp "Carry=$util_carry"
puts $fp "vivado_version=$vivado_version"
puts $fp "vivado_build_no=$vivado_build_no"
close $fp
puts "SYNTH-ONLY UTILIZATION: LUT $util_lut FF $util_ff BRAM18 $util_bram18 BRAM36 $util_bram36 URAM $util_uram"

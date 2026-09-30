set dcp [lindex $argv 0]
set outfile [lindex $argv 1]
open_checkpoint $dcp
report_utilization -hierarchical -hierarchical_depth 3 -file $outfile
exit

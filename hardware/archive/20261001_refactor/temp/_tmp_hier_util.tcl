open_checkpoint [lindex $argv 0]
set rpt_path [lindex $argv 1]
report_utilization -hierarchical -hierarchical_depth 3 -file $rpt_path
exit

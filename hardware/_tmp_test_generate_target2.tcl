open_project /tmp/finn_dev_thelegendiv/combined_stitch_proj_k3gshz1u/finn_design.xpr
open_bd_design /tmp/finn_dev_thelegendiv/combined_stitch_proj_k3gshz1u/finn_design.srcs/sources_1/bd/finn_design/finn_design.bd
report_ip_status -name ip_status
report_property [get_ips -all] -all
puts "=== ip list ==="
puts [get_ips -all]

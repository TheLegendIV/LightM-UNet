open_checkpoint /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.runs/synth_1/top_wrapper.dcp
foreach pin {top_i/GenericPartition_1_0/m_axis_0_tdata[0] top_i/GenericPartition_1_0/s_axis_0_tdata[0] top_i/GenericPartition_2_0/s_axis_0_tdata[0] top_i/GenericPartition_0_0/s_axis_0_tdata[0]} {
  set p [get_pins -quiet $pin]
  if {$p eq ""} { puts "$pin : PIN NOT FOUND POST-SYNTH"; continue }
  set net [get_nets -quiet -of_objects $p]
  puts "$pin : NET=$net"
}
report_utilization -file /tmp/postsynth_util.rpt
close_project

\
# Generates hierarchical utilization reports for partitions 3-7 of the
# 12_separable_dense_relu_alpha025_trained_8way build (already-completed
# partitions 0/2 are NOT regenerated here).
set parts {
  {3 /tmp/finn_dev_thelegendiv/synth_out_of_context_m2mopup2/results_GenericPartition_3_wrapper}
  {4 /tmp/finn_dev_thelegendiv/synth_out_of_context_nfdc7hx6/results_GenericPartition_4_wrapper}
  {5 /tmp/finn_dev_thelegendiv/synth_out_of_context_jdxc2ja8/results_GenericPartition_5_wrapper}
  {6 /tmp/finn_dev_thelegendiv/synth_out_of_context_9e79vd55/results_GenericPartition_6_wrapper}
  {7 /tmp/finn_dev_thelegendiv/synth_out_of_context_oxzfqjq9/results_GenericPartition_7_wrapper}
}

foreach p $parts {
  set n [lindex $p 0]
  set proj_folder [lindex $p 1]
  set dcp "$proj_folder/vivadocompile/vivadocompile.runs/impl_1/GenericPartition_${n}_wrapper_routed.dcp"
  puts "=== partition $n: opening $dcp ==="
  open_checkpoint $dcp
  report_utilization -hierarchical -hierarchical_depth 4 -file /tmp/hier_partition_${n}.rpt
  close_design
}

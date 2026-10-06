# Creates a clean Vivado project targeting ZCU106 and adds ALL dataflow
# sources needed for the combined S12_dense_256_u4_analytical_v1 design,
# pre-resolved for name collisions, as plain (non-BD) Verilog sources:
#   - the 8 partitions' already-merged/renamed flat bundle from
#     step_combine_partitions (finn_design_wrapper.v + 4086 files)
#   - both standalone IODMA cores, with every module renamed per-instance
#     (p0_input_* / p7_output_*) so the two otherwise-identical
#     "IODMA_hls_0" top modules (and their submodules) don't collide
#
# No BD / IP Integrator is used anywhere (see
# hardware/finn_partition_build_steps.py step_combine_partitions docstring
# for why combining FINN's partition stitched IPs via BD is unreliable).
# Wiring the PS, IODMAs and finn_design_wrapper together is left to you.

set PROJ_NAME zcu7ev_S12_dense_256_u4_analytical_v1
set PROJ_DIR /home/thelegendiv/finn/notebooks/enet/vivado_projects/$PROJ_NAME
set FPGA_PART xczu7ev-ffvc1156-2-e

set COMBINED_PARTITIONS_SRCS /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200/combined_stitch_proj_90htlxvu/all_verilog_srcs.txt
set IODMA_SRCS /home/thelegendiv/finn/notebooks/enet/vivado_projects/zcu7ev_S12_dense_256_u4_analytical_v1_srcs/iodma_renamed/iodma_verilog_srcs.txt

file mkdir /home/thelegendiv/finn/notebooks/enet/vivado_projects
create_project $PROJ_NAME $PROJ_DIR -part $FPGA_PART -force

set paths_prop [get_property BOARD_PART_REPO_PATHS [current_project]]
lappend paths_prop /home/thelegendiv/finn/deps/xil-bdf/boards
lappend paths_prop /home/thelegendiv/finn/deps/board_files
set_property BOARD_PART_REPO_PATHS $paths_prop [current_project]
set_param board.repoPaths $paths_prop
set_property board_part xilinx.com:zcu106:part0:2.6 [current_project]

proc add_srcs_from_list {list_path} {
    set fp [open $list_path r]
    set data [read $fp]
    close $fp
    set files {}
    foreach line [split $data "\n"] {
        set line [string trim $line]
        if {$line ne ""} {
            lappend files $line
        }
    }
    add_files -norecurse -fileset sources_1 $files
}

add_srcs_from_list $COMBINED_PARTITIONS_SRCS
add_srcs_from_list $IODMA_SRCS

update_compile_order -fileset sources_1

puts "ADD_SOURCES_DONE: project at $PROJ_DIR"
puts "  partition bundle top module: finn_design_wrapper (from $COMBINED_PARTITIONS_SRCS)"
puts "  input  DMA top module: p0_input_IODMA_hls_0  (Mem2Stream, gmem master + control + out_V stream)"
puts "  output DMA top module: p7_output_IODMA_hls_0 (Stream2Mem,  gmem master + control + in0_V stream)"
close_project

# Undo the leftover top-module setting from the earlier RTL-elaboration
# verification check (it persisted to sources_1's TopModule property
# without an explicit reset, which is why finn_design_wrapper stopped
# appearing in the BD canvas's Add Module dialog -- Vivado excludes the
# fileset's current top from that list).

open_project /home/thelegendiv/finn/notebooks/enet/vivado_projects/zcu7ev_S12_dense_256_u4_analytical_v1/zcu7ev_S12_dense_256_u4_analytical_v1.xpr

# automatic hierarchy update mode keeps re-picking *some* leaf module as
# top on every update_compile_order (whichever it is gets hidden from Add
# Module) -- switch to manual compile order so nothing gets auto-excluded.
set_property source_mgmt_mode None [current_project]
set_property top {} [current_fileset]
update_compile_order -fileset sources_1

puts "TOP_RESET_DONE: current top = [get_property top [current_fileset]]"
close_project

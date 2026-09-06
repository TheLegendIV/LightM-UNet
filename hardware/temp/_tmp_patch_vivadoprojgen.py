path = "/home/thelegendiv/finn/deps/oh-my-xilinx/vivadoprojgen.sh"
with open(path) as f:
    lines = f.readlines()

# find the header-marking loop's closing "done" (the one right after the
# "Verilog Header" file_type line) and insert the swg_pkg global-include
# block immediately after it.
idx = None
for i, l in enumerate(lines):
    if "Verilog Header" in l:
        idx = i
        break
assert idx is not None, "anchor line not found"
# next non-empty line should be "done"
j = idx + 1
while lines[j].strip() == "":
    j += 1
assert lines[j].strip() == "done", "expected done, got %r" % lines[j]

addition = [
    "\n",
    "# swg_pkg.sv declares SV package \"swg\" used by many swg_common\n",
    "# instances; mark it Global Include so Vivado always elaborates it\n",
    "# before any file that references the package, regardless of the\n",
    "# add_files/compile order.\n",
    "if [ -f 0_swg_pkg.sv ]; then\n",
    "\techo \"set file \\\"\\$origin_dir/0_swg_pkg.sv\\\"\" >> headers.tcl\n",
    "\techo \"set file [file normalize \\$file]\" >> headers.tcl\n",
    "\techo \"set file_obj [get_files -of_objects [get_filesets sources_1] "
    "[list \\\"*\\$file\\\"]]\" >> headers.tcl\n",
    "\techo \"set_property \\\"file_type\\\" \\\"Verilog Header\\\" \\$file_obj\" >> headers.tcl\n",
    "\techo \"set_property \\\"is_global_include\\\" \\\"1\\\" \\$file_obj\" >> headers.tcl\n",
    "fi\n",
]

lines = lines[: j + 1] + addition + lines[j + 1 :]
with open(path, "w") as f:
    f.writelines(lines)
print("patched at line", j + 1)

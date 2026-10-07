"""Phase 1: dump the VCD scope hierarchy (just $scope/$upscope/$var lines,
not the value-change data) so we can see how instance names map to node
names and what axis signal names look like at this trace depth."""
import sys

vcd_path = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/"
    "S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix/"
    "GenericPartition_1/rtlsim_single_vcd/trace.vcd"
)

scope_stack = []
count = 0
with open(vcd_path, "r", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if line.startswith("$scope"):
            parts = line.split()
            # $scope module <name> $end
            name = parts[2]
            scope_stack.append(name)
            print("  " * (len(scope_stack) - 1) + "SCOPE " + "/".join(scope_stack))
            count += 1
        elif line.startswith("$upscope"):
            scope_stack.pop()
        elif line.startswith("$var"):
            parts = line.split()
            # $var wire 1 ! tvalid $end  (or similar)
            vtype = parts[1]
            width = parts[2]
            code = parts[3]
            sig_name = " ".join(parts[4:-1])
            if "tvalid" in sig_name.lower() or "tready" in sig_name.lower():
                print("  " * len(scope_stack) + "VAR %s width=%s code=%s path=%s/%s" % (
                    vtype, width, code, "/".join(scope_stack), sig_name))
        elif line.startswith("$enddefinitions"):
            break
print("total scopes:", count)

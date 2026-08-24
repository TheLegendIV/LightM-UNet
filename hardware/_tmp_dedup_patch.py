path = "/home/thelegendiv/finn/deps/oh-my-xilinx/vivadoprojgen.sh"
with open(path) as f:
    lines = f.readlines()

# find both occurrences of the anchor comment line and drop the second block
anchor = "# swg_pkg.sv declares SV package \"swg\" used by many swg_common\n"
idxs = [i for i, l in enumerate(lines) if l == anchor]
print("anchor occurrences:", idxs)
if len(idxs) > 1:
    first_start = idxs[0]
    second_start = idxs[1]
    # each block is 12 lines long (comment*4 + if/echo*5 + fi) -- find its end
    # by locating the "fi\n" line following second_start
    end = second_start
    while lines[end].strip() != "fi":
        end += 1
    # remove lines[second_start .. end] inclusive, plus the blank line before it if present
    del lines[second_start : end + 1]
    with open(path, "w") as f:
        f.writelines(lines)
    print("removed duplicate block")
else:
    print("no duplicate found")

import sys
p = sys.argv[1]
s = open(p).read()
a = s.index("// Compute the count of decendents")
a = s.rindex("\n", 0, a) + 1
e = s.index("endfunction : leaf_load")
e = s.index("\n", e) + 1
block = s[a:e]
s = s[:a] + s[e:]
# de-indent one level (block was inside genTree), re-indent for module scope
lines = block.split("\n")
fixed = []
for l in lines:
    fixed.append(l[8:] if l.startswith("        ") else l)
block = "\n".join(fixed)
anchor = "localparam int unsigned  L = $clog2(N);"
i = s.index(anchor)
i = s.index("\n", i) + 1
s = s[:i] + "\n" + block + s[i:]
open(p, "w").write(s)
print("moved")

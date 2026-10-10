import sys
p = sys.argv[1]
s = open(p).read()
a = s.index("// Compute the count of decendents")
a = s.rindex("\n", 0, a) + 1
b = s.index("// Adder Tree")
b = s.rindex("\n", 0, b) + 1
ind = s[a:s.index("//", a)]
new = "".join(ind + l + "\n" if l else "\n" for l in [
    "// Compute the count of decendents for all nodes in the reduction trees.",
    "// Scalar-valued constant function (an unpacked-array-valued constant function under",
    "// generate is rejected by Verilator 5).",
    "function automatic int unsigned leaf_load(input int unsigned  idx);",
    "\tint unsigned  res[2*N-1];",
    "\tfor(int unsigned  i = 2*N-1; i-- > N-1;)  res[i] = 1;",
    "\tfor(int unsigned  i =   N-1; i-- >   0;)  res[i] = res[2*i+1] + res[2*i+2];",
    "\treturn  res[idx];",
    "endfunction : leaf_load",
    "",
])
s = s[:a] + new + s[b:]
s = s.replace("LEAF_LOAD[i]", "leaf_load(i)").replace("LEAF_LOAD[LIDX]", "leaf_load(LIDX)").replace("LEAF_LOAD[RIDX]", "leaf_load(RIDX)")
assert "LEAF_LOAD" not in s, "leftover"
open(p, "w").write(s)
print("patched", p)

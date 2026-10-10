import re, sys
p = sys.argv[1]
s = open(p).read()
old_start = s.index("                // Compute the count of decendents")
old_end = s.index("                // Adder Tree")
new = """                // Compute the count of decendents for all nodes in the reduction trees.
                // Scalar-valued constant function (an unpacked-array-valued constant function under
                // generate is rejected by Verilator 5).
                function automatic int unsigned leaf_load(input int unsigned  idx);
                        int unsigned  res[2*N-1];
                        for(int unsigned  i = 2*N-1; i-- > N-1;)  res[i] = 1;
                        for(int unsigned  i =   N-1; i-- >   0;)  res[i] = res[2*i+1] + res[2*i+2];
                        return  res[idx];
                endfunction : leaf_load

"""
s = s[:old_start] + new + s[old_end:]
s = s.replace("LEAF_LOAD[i]", "leaf_load(i)").replace("LEAF_LOAD[LIDX]", "leaf_load(LIDX)").replace("LEAF_LOAD[RIDX]", "leaf_load(RIDX)")
assert "LEAF_LOAD" not in s
open(p, "w").write(s)
print("patched", p)

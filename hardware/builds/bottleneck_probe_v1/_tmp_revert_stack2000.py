p = '/home/thelegendiv/finn/deps/oh-my-xilinx/vivadocompile.sh'
s = open(p).read()
old = 'vivado -stack 2000 -mode tcl -source $1.tcl -tclargs $1'
new = 'vivado -mode tcl -source $1.tcl -tclargs $1'
assert old in s, 'pattern not found'
s2 = s.replace(old, new, 1)
open(p, 'w').write(s2)
print('reverted:', new in s2, old not in s2)

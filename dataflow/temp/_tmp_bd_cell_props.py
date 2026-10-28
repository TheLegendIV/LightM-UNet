import re
f = '/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.srcs/sources_1/bd/top/top.bd'
txt = open(f).read()
idx = txt.find('"GenericPartition_0_0"')
print(txt[idx-300:idx+2000])

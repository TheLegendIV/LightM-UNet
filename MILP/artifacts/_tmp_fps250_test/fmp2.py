import onnx
d='/workspace/LightM-UNet/hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/autofold_configs_all_partitions/'
from collections import Counter
c=Counter()
for i in range(8):
    m=onnx.load(d+f'autofold_partition{i}.onnx')
    for n in m.graph.node:
        c[(i,n.op_type)]+=1
        if n.op_type.startswith('FMPadding') and not any(x.name=='Padding' for x in n.attribute):
            print('NO Padding',i,n.name,n.op_type,[(x.name,x.i if x.type==2 else list(x.ints) if x.type==7 else None) for x in n.attribute])
for k,v in sorted(c.items()): print(k,v)

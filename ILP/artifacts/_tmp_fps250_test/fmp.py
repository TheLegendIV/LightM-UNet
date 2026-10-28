import onnx,glob
d='/workspace/LightM-UNet/hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/autofold_configs_all_partitions/'
for i in range(8):
    m=onnx.load(d+f'autofold_partition{i}.onnx')
    for n in m.graph.node:
        if n.op_type.startswith('FMPadding'):
            a={x.name:(x.i if x.type==2 else list(x.ints) if x.type==7 else x.s.decode() if x.type==3 else None) for x in n.attribute}
            print(i,n.name,n.op_type,{k:v for k,v in a.items() if k in('Padding','ImgDim','NumChannels','SIMD','padding','PaddingStyle')}, sorted(a)[:30])
            break

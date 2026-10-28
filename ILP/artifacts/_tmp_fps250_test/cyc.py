import onnx
d='/workspace/LightM-UNet/hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/autofold_configs_all_partitions/'
for i in range(8):
    m=onnx.load(d+f'autofold_partition{i}.onnx'); best=(0,None,None)
    for n in m.graph.node:
        if n.op_type.startswith('StreamingFIFO'): continue
        for a in n.attribute:
            if a.name=='cycles_estimate' and a.i>best[0]: best=(a.i,n.op_type,n.name)
    print(i,best,'FPS %.1f'%(1e8/best[0]))

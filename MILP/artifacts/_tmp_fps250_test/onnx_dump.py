import onnx,json,sys
d='/workspace/LightM-UNet/hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/post_fifo_autosize_checkpoints/'
m=onnx.load(d+'partition2_dsrSweep_pbiOff_dsr1.5_milpfold_prefifo_autosize.onnx')
print(len(m.graph.node))
for n in m.graph.node:
    if n.op_type.startswith('StreamingFIFO'): continue
    a={x.name:(x.i if x.type==2 else x.s.decode() if x.type==3 else None) for x in n.attribute}
    print(n.op_type,n.name,{k:a.get(k) for k in('PE','SIMD','cycles_estimate','originalName','inFIFODepths') if k in a})

import onnx
d='/workspace/LightM-UNet/hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/autofold_configs_all_partitions/'
m=onnx.load(d+'autofold_partition7.onnx')
def at(n):
    o={}
    for x in n.attribute:
        if x.type==2:o[x.name]=x.i
        elif x.type==3:o[x.name]=x.s.decode()
        elif x.type==7:o[x.name]=list(x.ints)
    return o
for n in m.graph.node:
    if n.op_type.startswith(('MVAU','VVAU','Channelwise','ConvolutionInput')):
        a=at(n); print(n.name,{k:a.get(k) for k in('MW','MH','PE','SIMD','cycles_estimate','weightDataType','inputDataType','IFMChannels','ConvKernelDim','Channels')})

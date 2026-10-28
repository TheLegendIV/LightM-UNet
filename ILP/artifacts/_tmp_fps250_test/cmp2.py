import json
d='/workspace/LightM-UNet/MILP/artifacts/'
L=lambda p: json.load(open(d+p))
c=L('_tmp_fps250_test/control.json'); f=L('_tmp_fps250_test/fps250.json')
pc,pf=c['per_layer'],f['per_layer']
print(type(pc), list(pc.items())[0] if isinstance(pc,dict) else pc[0])
def fold(e): return {k:v for k,v in e.items() if k in('pe','simd','PE','SIMD','ram_style','variant','cycles','impl_style')}
n=0
for k in pc:
    if fold(pc[k])!=fold(pf[k]): n+=1; print(k,fold(pc[k]),fold(pf[k])) if n<=8 else None
print('layers',len(pc),'fold differ',n)
wd=sum(c['layer_weight_bits'][k]!=f['layer_weight_bits'][k] for k in c['layer_weight_bits']); print('w bits differ',wd)
print(c['_diagnostics'].get('bottleneck_node'),f['_diagnostics'].get('bottleneck_node'))

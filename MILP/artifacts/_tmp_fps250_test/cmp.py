import json,sys
d='/workspace/LightM-UNet/MILP/artifacts/'
L=lambda p: json.load(open(d+p))
c=L('_tmp_fps250_test/control.json'); f=L('_tmp_fps250_test/fps250.json'); s=L('S12_dense_nearest_upsample_512_hwsweep_partition2/dsrSweep_pbiOff_dsr1.5/layer_bits_folding_dsrSweep_pbiOff_dsr1.5.json')
print(list(c)[:8])
def flat(x):
    x=x.get('layers',x) if isinstance(x,dict) else x
    return x
for n,a,b in [('control vs stored',c,s),('fps250 vs control',f,c),('fps250 vs stored',f,s)]:
    A,B=flat(a),flat(b)
    if isinstance(A,dict):
        diff=[k for k in set(A)|set(B) if A.get(k)!=B.get(k)]
        print(n,'entries',len(A),'differ',len(diff),diff[:5])

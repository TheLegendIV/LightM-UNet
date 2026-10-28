import json
s=json.load(open('/workspace/LightM-UNet/MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_upsample_wm.json'))
L={k:v for k,v in s.items() if isinstance(v,dict) and 'trace_w' in v}
nw=[(k,v['trace_w']) for k,v in L.items() if v['trace_w']<0]; na=[(k,v['trace_a']) for k,v in L.items() if v['trace_a']<0]
print('layers',len(L),'neg trace_w',len(nw),'neg trace_a',len(na))
tw=sorted(v['trace_w'] for v in L.values()); ta=sorted(v['trace_a'] for v in L.values())
print('trace_w min/median/max %.3g %.3g %.3g'%(tw[0],tw[len(tw)//2],tw[-1]))
print('trace_a min/median/max %.3g %.3g %.3g'%(ta[0],ta[len(ta)//2],ta[-1]))
print('most negative w',sorted(nw,key=lambda x:x[1])[:4]); print('most negative a',sorted(na,key=lambda x:x[1])[:4])
mag=sum(abs(x[1]) for x in nw); tot=sum(abs(v['trace_w']) for v in L.values()); print('neg share of |trace_w| mass %.1f%%'%(100*mag/tot))
import itertools
k=next(iter(L)); print(k,{kk:L[k][kk] for kk in ('trace_w','trace_a')}, {b:L[k]['sensitivity_w'][b] for b in L[k]['sensitivity_w']})

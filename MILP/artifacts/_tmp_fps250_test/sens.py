import sys,json,glob
sys.path.insert(0,'/workspace/LightM-UNet/MILP')
import finn_milp as fm
from pathlib import Path
fm.load_config('config_12_dense_relu_nearest_upsample_wm')
fm.CANDIDATE_BITS=(4,6,8)
sens=json.load(open('/workspace/LightM-UNet/MILP/artifacts/layer_sensitivity_12_dense_relu_nearest_upsample_wm.json'))
sens.pop('_zero_sensitivity_layers',None)
model,geoms,extra,pred,dmap,kinds=fm.build_model_and_graph()
names=[g.name for g in geoms]; pairs=[(w,a) for w in fm.CANDIDATE_BITS for a in fm.CANDIDATE_BITS]
raw={}
for n in names:
    src=[s for s in fm._act_sensitivity_sources(n,pred) if s in sens]
    for w,a in pairs:
        sw=sens[n]['sensitivity_w'][str(w)] if n in sens else 0.0
        sa=max((sens[s]['sensitivity_a'][str(a)] for s in src),default=0.0)
        raw[(n,w,a)]=sw+sa
norm=fm._normalize(raw)
def score(wb,ab):
    r=sum(raw[(n,wb[n],ab[n])] for n in names); m=sum(norm[(n,wb[n],ab[n])] for n in names)/len(names); return r,m
A='/workspace/LightM-UNet/MILP/artifacts/'
rows=[]
def add(label,path):
    j=json.load(open(path)); 
    if not j.get('layer_weight_bits'): return
    r,m=score(j['layer_weight_bits'],j['layer_act_bits']); rows.append((label,r,m,sum(j['layer_weight_bits'].values())/len(names),sum(j['layer_act_bits'].values())/len(names)))
add('NO width cap (DSR2,305fps)','/workspace/LightM-UNet/MILP/artifacts/_tmp_fps250_test/nocap/l.json')
add('THIS: lex fps305 (caps1.0,DSR2)',A+'S12_dense_arms_bc_v1/lex_dsr2_fps305/layer_bits_folding_lex_dsr2_fps305.json')
add('lex fps250 (caps1.0,DSR2)',A+'S12_dense_arms_bc_v1/lex_dsr2/layer_bits_folding_lex_dsr2.json')
for t in ['baseline_both_off','dsrSweep_pbiOff_dsr1.5','dsrSweep_pbiOff_dsr3.0','dsrSweep_pbiOff_dsr7.5','dsrSweep_pbiOff_dsr15.0','dsrSweep_pbiOff_dsr150.0','pbiSweep_dsrOff_pbi1.5']:
    add('wm_v2 '+t+' (caps1.0,250fps)',A+f'archive/S12_dense_nearest_upsample_512_hwsweep_partition2_wm_v2/{t}/layer_bits_folding_{t}.json')
for t in ['baseline_both_off','dsrSweep_pbiOff_dsr7.5','dsrSweep_pbiOff_dsr15.0']:
    add('finncaps '+t+' (FINN cost caps)',A+f'S12_dense_nearest_upsample_512_hwsweep_partition2_wm_finncaps/{t}/layer_bits_folding_{t}.json')
for t in ['baseline_both_off','dsrSweep_pbiOff_dsr5.0']:
    add('orig _wm '+t+' (0.9/0.9/0.95)',A+f'S12_dense_nearest_upsample_512_hwsweep_partition2_wm/{t}/layer_bits_folding_{t}.json')
for b in (4,6,8):
    r,m=score({n:b for n in names},{n:b for n in names}); rows.append((f'uniform INT{b}',r,m,b,b))
print('%-46s %12s %10s %6s %6s'%('solve','raw_sum','norm_mean','avg_w','avg_a'))
for l,r,m,w,a in rows: print('%-46s %12.5f %10.5f %6.2f %6.2f'%(l,r,m,w,a))

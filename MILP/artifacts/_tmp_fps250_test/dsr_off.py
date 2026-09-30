import json
d='/workspace/LightM-UNet/MILP/artifacts/'
for p in ['S12_dense_nearest_upsample_512_hwsweep_partition2_wm_fps250/baseline_both_off','S12_dense_nearest_upsample_512_hwsweep_partition2_wm/baseline_both_off']:
    j=json.load(open(d+p+'/layer_bits_folding_baseline_both_off.json'))
    dg=j['_diagnostics']
    print(p); print({k:v for k,v in dg.items() if 'chain' in k or 'downstream' in k or 'dsr' in k})

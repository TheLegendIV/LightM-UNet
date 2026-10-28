import json
d='/workspace/LightM-UNet/MILP/artifacts/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/dsrSweep_pbiOff_dsr1.5/'
import glob
f=[p for p in glob.glob(d+'layer_bits_folding_*.json')][0]
j=json.load(open(f))
for k,e in j['per_layer'].items():
    print(k,'PE',e['pe'],'SIMD',e['simd'],'simd_swu',e.get('simd_swu'),'cyc',e['cycles'],'mvu',e.get('mvu_cycles'),'swu',e.get('swu_cycles'))

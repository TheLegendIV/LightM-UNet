import json
d='/workspace/LightM-UNet/MILP/artifacts/S12_dense_nearest_upsample_512_hwsweep_partition2_wm_finncaps_pinbits/'
j=json.load(open(d+'baseline_both_off/layer_bits_folding_baseline_both_off.json'))['per_layer']
rows=[(k,e['pe'],e['simd'],e['weight_bits'],e['simd']*e['weight_bits'],e['pe']*e['simd']*e['weight_bits']) for k,e in j.items() if e.get('mvu_cycles') is not None]
print('layers',len(rows),'| SIMD*wbits>80:',sum(r[4]>80 for r in rows),'| PE*SIMD*wbits>80:',sum(r[5]>80 for r in rows),'| max PE*SIMD*wbits',max(r[5] for r in rows))
for r in sorted(rows,key=lambda r:-r[5])[:4]: print(r)
print([r for r in rows if r[0].startswith(('out','final','head'))][:3], rows[-1])

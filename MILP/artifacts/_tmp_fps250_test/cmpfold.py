import json
for t in ['dsrmin_1x','dsrmin_2x','dsrmin_4x','dsrmin_8x','dsr_off']:
    a=json.load(open(f'/workspace/LightM-UNet/MILP/artifacts/_tmp_fps250_test/prev/prev_{t}.json'))['per_layer']; b=json.load(open(f'/workspace/LightM-UNet/MILP/artifacts/S12_dense_dsr_ablation_v1/{t}/layer_bits_folding_{t}.json'))['per_layer']
    d=[k for k in a if (a[k]['pe'],a[k]['simd'])!=(b[k]['pe'],b[k]['simd'])]
    print(t,'layers with different PE/SIMD:',len(d),d[:4])

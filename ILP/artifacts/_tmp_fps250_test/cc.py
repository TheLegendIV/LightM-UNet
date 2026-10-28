import csv,json
b='/workspace/LightM-UNet/MILP/artifacts/S12_dense_arms_bc_v1/'
for t in ['lex_dsr2_fps305','lex_nodsr_fps305']:
    r=next(csv.DictReader(open(b+t+'/summary.csv')))
    print(t.ljust(18),'LUT %.1f BRAM %.1f DSP %.1f fps %.2f'%(float(r['lut_pct_of_budget']),float(r['bram_pct_of_budget']),float(r['dsp_pct_of_budget']),float(r['fps'])),'DSRfoldable med %.2f max %.2f'%(float(r['dsr_foldable_median']),float(r['dsr_foldable_max'])),'sens_norm %.5f'%float(r['sensitivity_norm_mean']))
x=json.load(open(b+'lex_dsr2_fps305/layer_bits_folding_lex_dsr2_fps305.json')); y=json.load(open(b+'lex_nodsr_fps305/layer_bits_folding_lex_nodsr_fps305.json'))
print('same bits',x['layer_weight_bits']==y['layer_weight_bits'] and x['layer_act_bits']==y['layer_act_bits'])
print('layers with different PE/SIMD',sum((x['per_layer'][k]['pe'],x['per_layer'][k]['simd'])!=(y['per_layer'][k]['pe'],y['per_layer'][k]['simd']) for k in x['per_layer']))

import csv
for r in csv.DictReader(open('/workspace/LightM-UNet/compression/results.csv')):
    n=r['config_name']
    if 'lex_dsr2_fps305' in n or ('wm_uniform_int' in n and 'samefolding' in n and n.endswith('ft15ep')) or ('wm_joint_alpha1.0_perlayer_candidatebits468_forcedsp_lut50_bram50_dsp90_ft15ep' in n and n.endswith('ft15ep')):
        print(n[-62:].ljust(62),'dice %.4f cldice %.4f LAD %.3f RCA %.3f LCX %.3f LM %.3f'%(float(r['dice']),float(r['cldice']),float(r['dice_LAD']),float(r['dice_RCA']),float(r['dice_LCX']),float(r['dice_LM'])))

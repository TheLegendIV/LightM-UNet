import csv,os
b='/workspace/LightM-UNet/MILP/artifacts/S12_dense_arms_bc_v1/'
def row(label,path):
    if not os.path.exists(path+'/summary.csv'):
        st=[l.strip() for l in open(path+'/solve.log') if 'ILP status' in l]; print(label.ljust(34),st or 'no summary'); return
    r=next(csv.DictReader(open(path+'/summary.csv')))
    print(label.ljust(34),'LUT %5.1f BRAM %5.1f DSP %5.1f fps %.1f DSRmax %.2f'%(float(r['lut_pct_of_budget']),float(r['bram_pct_of_budget']),float(r['dsp_pct_of_budget']),float(r['fps']),float(r['dsr_foldable_max'])))
row('lex mixed 6.55/6.86  DSR2',b+'lex_dsr2_fps305'); row('lex mixed 6.55/6.86  noDSR',b+'lex_nodsr_fps305')
for bits in (4,6,8):
    for d in ('dsr2','nodsr'): row(f'uniform INT{bits} {d}',b+f'uniform_ref/int{bits}_{d}')

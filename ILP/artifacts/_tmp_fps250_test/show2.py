import csv,glob,os
for f in sorted(glob.glob('/workspace/LightM-UNet/MILP/artifacts/S12_dense_nearest_upsample_512_hwsweep_partition2_wm_finncaps_pinbits/*/')):
    t=f.rstrip('/').split('/')[-1]
    s=f+'summary.csv'
    if not os.path.exists(s):
        print(t.ljust(26),'INFEASIBLE / no summary:',[l.strip() for l in open(f+'solve.log') if 'status' in l]); continue
    r=next(csv.DictReader(open(s)))
    print(t.ljust(26),r['status'],'wbits %.2f abits %.2f'%(float(r['avg_weight_bits']),float(r['avg_act_bits'])),'LUT %.1f BRAM %.1f DSP %.1f (%% of board)'%(float(r['lut_pct_of_budget'])*0.9 if False else float(r['lut_pct_of_budget']),float(r['bram_pct_of_budget']),float(r['dsp_pct_of_budget'])),'fps %.1f'%float(r['fps']),'DSRmed %.2f'%float(r['dsr_median']))

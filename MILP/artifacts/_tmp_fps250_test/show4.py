import csv,glob,os
for f in sorted(glob.glob('/workspace/LightM-UNet/MILP/artifacts/S12_dense_dsr_ablation_v1/fps305/*/'),key=lambda x:x):
    t=f.rstrip('/').split('/')[-1]; s=f+'summary.csv'
    if not os.path.exists(s): print(t.ljust(18),[l.strip() for l in open(f+'solve.log') if 'status' in l or 'Error' in l]); continue
    r=next(csv.DictReader(open(s)))
    print(t.ljust(18),r['status'],'LUT %.1f BRAM %.1f DSP %.1f'%(float(r['lut_pct_of_budget']),float(r['bram_pct_of_budget']),float(r['dsp_pct_of_budget'])),'fps %.1f'%float(r['fps']),'DSRfoldable med %s max %.2f (%s)'%(r['dsr_foldable_median'][:5],float(r['dsr_foldable_max']),r['dsr_foldable_worst_node']))

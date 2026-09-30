import csv,glob
for f in sorted(glob.glob('/workspace/LightM-UNet/MILP/artifacts/S12_dense_nearest_upsample_512_hwsweep_partition2_wm_v2/*/summary.csv')):
    r=next(csv.DictReader(open(f)))
    print(f.split('/')[-2].ljust(26),'fps %.1f'%float(r['fps']),'dsr_set',r['dsr_ratio_setting'] or '-','| DSR med %.2f mean %.2f max %.2f'%(float(r['dsr_median']),float(r['dsr_mean']),float(r['dsr_max'])),r['dsr_worst_node'])

import csv
rows=list(csv.DictReader(open('/workspace/LightM-UNet/compression/results.csv')))
for r in rows:
    n=r['config_name']
    if 'nearest_upsample_wm_' in n and 'ft15ep' in n and ('uniform' in n or 'joint' in n):
        print(n.split('perlayer_12_dense_relu_nearest_upsample_wm_')[-1][:70].ljust(70),'stage',r['stage'][-24:],'dice',r['dice'],'cldice',r['cldice'],'ep',r['epochs'],'conv',r['converged_flag'])

p='/workspace/LightM-UNet/hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/finn_hawq_folding_bridge_nearest_upsample.py'
b=open(p,'rb').read().replace(b'\r\n',b'\n')
if b'import dataclasses\n' not in b: b=b.replace(b'import json\n',b'import dataclasses\nimport json\n',1)
open(p,'wb').write(b.replace(b'\n',b'\r\n')); print('done')

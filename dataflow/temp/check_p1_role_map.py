import json
with open('/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/partition1_refix_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_refix1/fifo_plan_partition1.json') as f:
    plan = json.load(f)
role_of_node = plan['role_of_node']
node_of_role = {v: k for k, v in role_of_node.items()}
for b in ('down1', 'down2', 'regular1.0'):
    for suffix in ('thr_s', 'skip_quant', 'add', 'maxpool', 'thr_r', 'dup'):
        key = f'{b}.{suffix}'
        print(key, '->', node_of_role.get(key))
print('---')
print('Thresholding_rtl_1 role:', role_of_node.get('Thresholding_rtl_1'))
print('Thresholding_rtl_26 role:', role_of_node.get('Thresholding_rtl_26'))
print('total role_of_node entries:', len(role_of_node))
print('all down1.* roles:', sorted(k for k in node_of_role if k.startswith('down1.')))
print('all down2.* roles:', sorted(k for k in node_of_role if k.startswith('down2.')))

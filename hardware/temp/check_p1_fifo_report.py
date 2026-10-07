import json
with open('/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/partition1_refix_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_refix1/fifo_force_report_partition_1.json') as f:
    rep = json.load(f)
print('total entries:', len(rep))
forced = [e for e in rep if e.get('forced_depth')]
print('forced count:', len(forced))
big = sorted(forced, key=lambda e: -e['forced_depth'])[:10]
for e in big:
    print(e['fifo'], e['producer_role'], '=>', e['consumer_role'], 'stock=', e['stock_depth'], 'forced=', e['forced_depth'])
print('--- unmatched (forced_depth null) ---')
unmatched = [e for e in rep if not e.get('forced_depth')]
print('unmatched count:', len(unmatched))
for e in unmatched:
    print(e['fifo'], e['producer_node'], '=>', e['consumer_node'], 'roles:', e['producer_role'], '=>', e['consumer_role'], 'stock=', e['stock_depth'])

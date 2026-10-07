import json
d = json.load(open('/tmp/fifo_compare.json'))
rows = d['fifo_comparison']

def fnum(x):
    return x if isinstance(x, (int, float)) else -1

print(f"{'fifo':24s} {'analytical':>10s} {'autosize':>9s} {'producer':>32s} {'consumers'}")
for r in sorted(rows, key=lambda r: -fnum(r['autosize_depth'])):
    a = r['analytical_depth']
    b = r['autosize_depth']
    prod = r['producer_op'] or '?'
    cons = ','.join(r['consumer_ops'] or []) or '?'
    print(f"{r['fifo_name']:24s} {str(a):>10s} {str(b):>9s} {prod:>32s} {cons}  [{r['only_in']}]")

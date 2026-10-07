import json
d = json.load(open('/tmp/fifo_compare.json'))
print('=== MVAU_rtl_8 neighborhood (analytical) ===')
print(json.dumps(d['mvau8_neighborhood_analytical'], indent=2))
print('=== MVAU_rtl_8 neighborhood (autosize) ===')
print(json.dumps(d['mvau8_neighborhood_autosize'], indent=2))

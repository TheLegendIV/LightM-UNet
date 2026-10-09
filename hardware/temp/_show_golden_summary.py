import json
for tag, path in (("old", "/tmp/golden_pp/summary.json"), ("new", "/tmp/golden_pp_p1fix/summary.json")):
    d = json.load(open(path))
    print(tag, json.dumps(d)[:1500])

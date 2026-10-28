import re
import sys
from collections import Counter

log_path = sys.argv[1]
names = Counter()
pairs = []
with open(log_path, "r", errors="ignore") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    m = re.search(r"File '([^']+)' is used", line)
    if m:
        fname = m.group(1)
        names[fname] += 1
        # collect the following '*' bullet lines (the colliding source paths)
        paths = []
        j = i + 1
        while j < len(lines) and lines[j].strip().startswith("*"):
            paths.append(lines[j].strip())
            j += 1
        pairs.append((fname, paths))

print("=== unique filenames with mismatched content, by count ===")
for fname, cnt in names.most_common():
    print(f"{cnt:3d}  {fname}")

print()
print("=== first example per unique filename ===")
seen = set()
for fname, paths in pairs:
    if fname in seen:
        continue
    seen.add(fname)
    print(fname)
    for p in paths:
        print("    ", p)

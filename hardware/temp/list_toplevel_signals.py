import sys
path = sys.argv[1]
scope = []
with open(path, errors="replace") as f:
    for line in f:
        line = line.strip()
        if line.startswith("$scope"):
            scope.append(line.split()[2])
        elif line.startswith("$upscope"):
            if scope:
                scope.pop()
        elif line.startswith("$var"):
            parts = line.split()
            name = ".".join(scope + [parts[4]])
            low = name.lower()
            if ("tvalid" in low or "tready" in low) and len(scope) <= 4:
                print(name)
        elif line.startswith("$enddefinitions"):
            break

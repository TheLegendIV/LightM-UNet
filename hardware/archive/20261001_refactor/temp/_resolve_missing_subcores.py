import os, re, json

BASE = "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp"

# partition -> list of core names (without the p{N}_ prefix)
CORES = {
    0: ["DuplicateStreams_hls_0", "StreamingConcat_hls_0", "StreamingMaxPool_hls_0"],
    1: ["AddStreams_hls_0","AddStreams_hls_1","AddStreams_hls_2","AddStreams_hls_3","AddStreams_hls_4","AddStreams_hls_5",
        "DuplicateStreams_hls_0","DuplicateStreams_hls_1","DuplicateStreams_hls_2","DuplicateStreams_hls_3","DuplicateStreams_hls_4","DuplicateStreams_hls_5",
        "StreamingMaxPool_hls_0","StreamingMaxPool_hls_1"],
    2: ["AddStreams_hls_0","AddStreams_hls_1","AddStreams_hls_2","AddStreams_hls_3","AddStreams_hls_4",
        "DuplicateStreams_hls_0","DuplicateStreams_hls_1","DuplicateStreams_hls_2","DuplicateStreams_hls_3","DuplicateStreams_hls_4"],
    3: ["AddStreams_hls_0","AddStreams_hls_1","AddStreams_hls_2","AddStreams_hls_3",
        "DuplicateStreams_hls_0","DuplicateStreams_hls_1","DuplicateStreams_hls_2","DuplicateStreams_hls_3"],
    4: ["AddStreams_hls_0","AddStreams_hls_1","AddStreams_hls_2","AddStreams_hls_3",
        "DuplicateStreams_hls_0","DuplicateStreams_hls_1","DuplicateStreams_hls_2","DuplicateStreams_hls_3"],
    5: ["AddStreams_hls_0","AddStreams_hls_1","AddStreams_hls_2","AddStreams_hls_3",
        "DuplicateStreams_hls_0","DuplicateStreams_hls_1","DuplicateStreams_hls_2","DuplicateStreams_hls_3",
        "FMPadding_Pixel_hls_0","UpsampleNearestNeighbour_hls_0","VVAU_hls_0"],
    6: ["AddStreams_hls_0","AddStreams_hls_1","AddStreams_hls_2",
        "DuplicateStreams_hls_0","DuplicateStreams_hls_1","DuplicateStreams_hls_2",
        "FMPadding_Pixel_hls_0","UpsampleNearestNeighbour_hls_0","VVAU_hls_0"],
    7: ["AddStreams_hls_0","ChannelwiseOp_hls_0","DuplicateStreams_hls_0","FMPadding_Pixel_hls_0","VVAU_hls_0"],
}

# build full list of (partition, expected_vlnv_name)
targets = []
for n, cores in CORES.items():
    for c in cores:
        targets.append((n, f"p{n}_{c}"))
# special-case partition-0/2 IODMA cores (not prefixed per report)
targets.append((0, "StreamingDataflowPartition_0_IODMA_hls_0"))
targets.append((2, "StreamingDataflowPartition_2_IODMA_hls_0"))

# index all component.xml files once (content + mtime), skip archive dirs
name_pat = re.compile(rb"<spirit:name>([^<]+)</spirit:name>")
index = {}  # vlnv_name -> list of (mtime, path)
for root, dirs, files in os.walk(BASE):
    dirs[:] = [d for d in dirs if "archive_pre_" not in d and d != "archive"]
    if "component.xml" in files:
        p = os.path.join(root, "component.xml")
        try:
            with open(p, "rb") as f:
                data = f.read()
            m = name_pat.search(data)
            if m:
                nm = m.group(1).decode()
                index.setdefault(nm, []).append((os.path.getmtime(p), root))
        except Exception as e:
            pass

results = {}
missing = []
for n, vlnv in targets:
    matches = index.get(vlnv)
    if not matches:
        missing.append((n, vlnv))
        continue
    matches.sort(reverse=True)  # newest first
    results[vlnv] = matches[0][1]  # dirname containing component.xml

print("=== RESOLVED (%d) ===" % len(results))
for vlnv, d in results.items():
    print(f"{vlnv} -> {d}")

print("=== MISSING (%d) ===" % len(missing))
for n, vlnv in missing:
    print(f"partition {n}: {vlnv}")

# write out unique dirs to a tcl-friendly file
dirs_unique = sorted(set(results.values()))
with open("/tmp/_resolved_repo_dirs.txt", "w") as f:
    for d in dirs_unique:
        f.write(d + "\n")
print("=== wrote %d unique dirs to /tmp/_resolved_repo_dirs.txt ===" % len(dirs_unique))

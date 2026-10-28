import os

dirs = [
    "combined_stitch_proj_urzh2_bm",
    "combined_stitch_proj_dud9_c9n",
    "combined_stitch_proj_a_1k75yq",
    "combined_stitch_proj_o5ahyfgl",
]
base = "/tmp/finn_dev_thelegendiv"
for d in dirs:
    p = os.path.join(base, d)
    print(d, "->", "EXISTS" if os.path.isdir(p) else "NOT_FOUND")

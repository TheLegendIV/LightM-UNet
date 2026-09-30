import re, glob

base = "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_zynq_proj_ivzy_6_z/finn_zynq_link.srcs/sources_1/bd/top/ip"
dirs = [
    "top_StreamingDataflowPar_0_0",
    "top_StreamingDataflowPar_1_0",
    "top_StreamingDataflowPar_10_1",
    "top_StreamingDataflowPar_3_0",
    "top_StreamingDataflowPar_4_0",
    "top_StreamingDataflowPar_5_0",
    "top_StreamingDataflowPar_6_0",
    "top_StreamingDataflowPar_7_0",
]

pattern = re.compile(r'("FREQ_HZ":\s*\[\s*\{\s*"value":\s*")100000000(")')

for d in dirs:
    path = f"{base}/{d}/{d}.xci"
    with open(path) as f:
        content = f.read()
    n = len(pattern.findall(content))
    new_content = pattern.sub(r"\g<1>99990005\g<2>", content)
    with open(path, "w") as f:
        f.write(new_content)
    print(f"{path}: patched {n} occurrence(s)")

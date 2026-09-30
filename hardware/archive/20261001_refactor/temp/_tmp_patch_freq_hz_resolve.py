import glob
import re
import sys

BASE = "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp"

paths = glob.glob(BASE + "/**/ip/component.xml", recursive=True)
paths += glob.glob(BASE + "/**/vivado_stitch_proj_*/component.xml", recursive=True)
paths = sorted(set(paths))

print(f"found {len(paths)} component.xml candidates")

n_patched = 0
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()

    if "BUSIFPARAM_VALUE.CLK.AP_CLK.FREQ_HZ" not in text:
        continue

    orig = text

    # 1. flip the busif parameter's own resolve type from user (fixed) to generated (inherited)
    text = text.replace(
        'spirit:resolve="user" spirit:id="BUSIFPARAM_VALUE.CLK.AP_CLK.FREQ_HZ"',
        'spirit:resolve="generated" spirit:id="BUSIFPARAM_VALUE.CLK.AP_CLK.FREQ_HZ"',
    )

    # 2. widen the tolerance too, as a belt-and-suspenders fallback (covers the ~9995 Hz PS PLL gap)
    text = re.sub(
        r'(spirit:id="BUSIFPARAM_VALUE\.CLK\.AP_CLK\.FREQ_TOLERANCE_HZ">)0(</spirit:value>)',
        r"\g<1>30000\g<2>",
        text,
    )

    # 3. matching config-element-info bookkeeping entry (per-instance customization source)
    text = text.replace(
        'xilinx:referenceId="BUSIFPARAM_VALUE.CLK.AP_CLK.FREQ_HZ" xilinx:valueSource="user"',
        'xilinx:referenceId="BUSIFPARAM_VALUE.CLK.AP_CLK.FREQ_HZ" xilinx:valueSource="generated"',
    )

    if text != orig:
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        n_patched += 1
        print(f"patched: {p}")

print(f"DONE, patched {n_patched} file(s)")

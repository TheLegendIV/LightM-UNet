"""Create a locally-patched copy of Xilinx's xpm_memory.sv with the simulation-only
$info(...) calls in config_drc neutered. These calls build long %m-hierarchy-substituted
strings; on the S12_256_analytical 8-partition combined design (unusually deep hierarchy),
Verilator 4.224 crashes with 'stack smashing detected' while executing one of them during
elaboration. The $info calls are purely informational (not DRC errors), safe to drop for
simulation. Does NOT touch the real Xilinx install -- writes a separate file.
"""
SRC = "/tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv"
DST = "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/xpm_memory_patched.sv"

TARGETS = [
    '      $info("[%s %0d-%0d] MEMORY_PRIMITIVE (%0d) instructs Vivado Synthesis to choose the memory primitive type. Depending on their values, other XPM_MEMORY parameters may preclude the choice of certain memory primitive types. Review XPM_MEMORY documentation and parameter values to understand any limitations, or set MEMORY_PRIMITIVE to a different value. %m", "XPM_MEMORY", 20, 1, MEMORY_PRIMITIVE);',
    '      $info("[%s %0d-%0d] MEMORY_INIT_FILE (%0s), MEMORY_INIT_PARAM together specify no memory initialization. Initial memory contents will be all 0\'s. %m", "XPM_MEMORY", 20, 2, MEMORY_INIT_FILE,MEMORY_INIT_PARAM);',
    '      $info("[%s %0d-%0d] XPM_MEMORY behaviorally models the port operation ordering of true dual port UltraRAM configurations by slightly delaying the common clock for port B operations only. Refer to UltraRAM documentation for details. %m", "XPM_MEMORY", 20, 3);',
    '      $info("[%s %0d-%0d] Non-zero AUTO_SLEEP_TIME (%0d) is specifed for this configuration, An input pipeline having the number of register stages equal to AUTO_SLEEP_TIME will be introduced on all the input control/data signals path except for the port-enables(en[a|b]) and reset(rst[a|b]). %m", "XPM_MEMORY", 20, 4, AUTO_SLEEP_TIME);',
]

with open(SRC, "r") as f:
    text = f.read()

n_replaced = 0
for t in TARGETS:
    count = text.count(t)
    if count != 1:
        print(f"WARNING: expected exactly 1 occurrence, found {count} for: {t[:80]}...")
    text = text.replace(t, "      ; // PATCHED OUT: stack-smashing in Verilator on deep combined hierarchy")
    n_replaced += count

with open(DST, "w") as f:
    f.write(text)

print(f"Replaced {n_replaced}/4 $info calls. Wrote patched file to {DST}")

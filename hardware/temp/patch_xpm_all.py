"""Broader patch: strip the '%m' hierarchical-path token from EVERY Verilog
system-task format string (display/info/warning/error/fatal) across all 3
Xilinx XPM source files fed into this combined-design Verilator compile.
Rationale: narrowly neutering xpm_memory.sv's 4 "Infos" $info(...) calls did
NOT fix the 'stack smashing detected' crash (still crashes at the identical
point -- first eval_initial_loop invocation, main thread, even with
ulimit -s unlimited and without --threads) -- strongly suggesting the real
trigger is a DIFFERENT (earlier-executing, e.g. a DRC $error/$warning) %m
call, possibly in xpm_cdc.sv/xpm_fifo.sv rather than xpm_memory.sv's "Infos"
section. Removing ' %m"' (the consistent Xilinx-authored suffix pattern used
right before the closing quote) from every message, globally, across all 3
files is a blunt but safe way to eliminate every %m-substitution call site at
once -- these are simulation-only DRC/info messages, not functionally
relevant RTL behavior.
"""
FILES = [
    ("/tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv",
     "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/xpm_memory_patched.sv"),
    ("/tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_cdc/hdl/xpm_cdc.sv",
     "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/xpm_cdc_patched.sv"),
    ("/tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_fifo/hdl/xpm_fifo.sv",
     "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/xpm_fifo_patched.sv"),
]

for src, dst in FILES:
    with open(src, "r") as f:
        text = f.read()
    n1 = text.count(' %m"')
    text = text.replace(' %m"', '"')
    # also handle the no-leading-space variant "...%m" (no space before %m)
    n2 = text.count('%m"')
    text = text.replace('%m"', '"')
    with open(dst, "w") as f:
        f.write(text)
    print(f"{src}: removed {n1} ' %m\"' occurrences + {n2} remaining '%m\"' occurrences -> {dst}")

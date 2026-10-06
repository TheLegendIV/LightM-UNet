#!/usr/bin/env python3
"""Renames module declarations in the two standalone IODMA ipgen source
sets so they can coexist as flat Verilog sources in one Vivado project
(both instances independently generated a module literally named
IODMA_hls_0, plus several identically-named internal submodules --
e.g. IODMA_hls_0_control_s_axi, IODMA_hls_0_fifo_w32_d2_S -- which would
collide if added to the same project/fileset without renaming).

Usage: python3 _rename_iodma_sources.py <out_dir>
Writes <out_dir>/p0_input/*.v, <out_dir>/p7_output/*.v and
<out_dir>/iodma_verilog_srcs.txt (merged file list, one path per line).
"""
import os
import re
import sys

IODMA_DIRS = {
    "p0_input": (
        "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/"
        "S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200/GenericPartition_0/"
        "code_gen_ipgen_IODMA_hls_0_2aozvq23/project_IODMA_hls_0/sol1/impl/verilog"
    ),
    "p7_output": (
        "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/"
        "S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200/GenericPartition_7/"
        "code_gen_ipgen_IODMA_hls_0_094h060g/project_IODMA_hls_0/sol1/impl/verilog"
    ),
}


def collect_module_decls(path):
    with open(path, errors="ignore") as f:
        text = f.read()
    return set(re.findall(r"^\s*module\s+(\w+)", text, re.MULTILINE))


def main():
    out_root = sys.argv[1]
    merged = []
    for prefix, src_dir in IODMA_DIRS.items():
        src_files = sorted(
            os.path.join(src_dir, fn) for fn in os.listdir(src_dir) if fn.endswith(".v")
        )
        module_names = set()
        for fn in src_files:
            module_names |= collect_module_decls(fn)
        rename_map = {m: "%s_%s" % (prefix, m) for m in module_names}
        sorted_names = sorted(rename_map, key=len, reverse=True)
        combined_re = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(n) for n in sorted_names))

        out_dir = os.path.join(out_root, prefix)
        os.makedirs(out_dir, exist_ok=True)
        for fn in src_files:
            with open(fn, errors="ignore") as f:
                text = f.read()
            text = combined_re.sub(lambda m: rename_map[m.group(0)], text)
            out_path = os.path.join(out_dir, os.path.basename(fn))
            with open(out_path, "w") as f:
                f.write(text)
            merged.append(out_path)
        print("[%s] renamed %d modules across %d files -> %s" % (
            prefix, len(rename_map), len(src_files), out_dir))

    list_path = os.path.join(out_root, "iodma_verilog_srcs.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(merged) + "\n")
    print("wrote %s (%d files)" % (list_path, len(merged)))
    print("top modules: p0_input_IODMA_hls_0 (Mem2Stream/input DMA), "
          "p7_output_IODMA_hls_0 (Stream2Mem/output DMA)")


if __name__ == "__main__":
    main()

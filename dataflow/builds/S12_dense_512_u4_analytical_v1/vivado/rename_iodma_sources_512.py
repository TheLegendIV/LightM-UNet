#!/usr/bin/env python3
"""512 variant of hardware/_rename_iodma_sources.py.

Renames the Verilog module declarations of the two standalone IODMA ipgen
source sets (input = GenericPartition_0, output = GenericPartition_7) with a
p0_input_ / p7_output_ prefix so both can live in one Vivado project.
IODMA ipgen dir hashes are discovered by glob.

Usage: python3 rename_iodma_sources_512.py <finn_build_dir> <out_dir>
Writes <out_dir>/p0_input/*.v and <out_dir>/p7_output/*.v
"""
import glob
import os
import re
import sys

PARTS = {"p0_input": "GenericPartition_0", "p7_output": "GenericPartition_7"}


def main():
    build_dir, out_root = sys.argv[1], sys.argv[2]
    for prefix, part in PARTS.items():
        pattern = os.path.join(
            build_dir, part, "code_gen_ipgen_IODMA_hls_0_*",
            "project_IODMA_hls_0", "sol1", "impl", "verilog")
        hits = glob.glob(pattern)
        if len(hits) != 1:
            raise SystemExit("expected exactly 1 match for %s, got %s" % (pattern, hits))
        src_dir = hits[0]
        src_files = sorted(os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".v"))
        names = set()
        for fn in src_files:
            with open(fn, errors="ignore") as f:
                names |= set(re.findall(r"^\s*module\s+(\w+)", f.read(), re.MULTILINE))
        rename_map = {m: "%s_%s" % (prefix, m) for m in names}
        rx = re.compile(r"\b(?:%s)\b" % "|".join(
            re.escape(n) for n in sorted(rename_map, key=len, reverse=True)))
        out_dir = os.path.join(out_root, prefix)
        os.makedirs(out_dir, exist_ok=True)
        for fn in src_files:
            with open(fn, errors="ignore") as f:
                text = rx.sub(lambda m: rename_map[m.group(0)], f.read())
            with open(os.path.join(out_dir, os.path.basename(fn)), "w") as f:
                f.write(text)
        print("[%s] %d modules renamed across %d files: %s -> %s" % (
            prefix, len(rename_map), len(src_files), src_dir, out_dir))


if __name__ == "__main__":
    main()

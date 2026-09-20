"""Checks the REAL step_combine_partitions output (all_verilog_srcs.txt) for
the tiny 2-partition real-pipeline test: collects every `module NAME` decl
across all merged/renamed source files and reports any name that appears
more than once (excluding the known-shared/intentionally-duplicated
library files step_combine_partitions itself already tolerates)."""
import os
import re
import sys

OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/tiny_2part_real_pipeline_test"

_SKIP_RENAME_SUBSTRINGS = ("regslice_core", "swg_pkg", "axis_infrastructure_v1_1_vl_rfs")


def find_srcs_list():
    # step_combine_partitions writes into a make_build_dir("combined_stitch_proj_")
    # dir under FINN_BUILD_DIR; find the most recent one via the model's own
    # metadata, but simplest: scan build log for the path it printed.
    log_path = os.path.join(OUTPUT_DIR, "build_dataflow.log")
    with open(log_path, errors="ignore") as f:
        text = f.read()
    m = re.search(r"combined stitch ready: (\S+)", text)
    assert m, "Could not find 'combined stitch ready' line in build log"
    return os.path.join(m.group(1), "all_verilog_srcs.txt")


def collect_module_decls(path):
    with open(path, errors="ignore") as f:
        text = f.read()
    return set(re.findall(r"^\s*module\s+(\w+)", text, re.MULTILINE))


def main():
    srcs_list = find_srcs_list()
    print("srcs list:", srcs_list)
    with open(srcs_list) as f:
        files = [l.strip() for l in f if l.strip()]
    print(f"{len(files)} verilog source files merged")

    module_to_files = {}
    for fn in files:
        if any(s in fn for s in _SKIP_RENAME_SUBSTRINGS):
            continue
        if not (fn.endswith(".v") or fn.endswith(".sv")):
            continue
        if not os.path.isfile(fn):
            print("MISSING FILE:", fn)
            continue
        for mod in collect_module_decls(fn):
            module_to_files.setdefault(mod, []).append(fn)

    dupes = {m: fs for m, fs in module_to_files.items() if len(fs) > 1}
    print(f"\nTotal unique module names: {len(module_to_files)}")
    print(f"Modules declared in >1 file (potential collision): {len(dupes)}")
    for m, fs in sorted(dupes.items()):
        print(f"  {m}:")
        for fn in fs:
            print(f"    {fn}")

    if not dupes:
        print("\n=== NO CROSS-FILE MODULE NAME COLLISIONS FOUND ===")
    sys.exit(0)


if __name__ == "__main__":
    main()

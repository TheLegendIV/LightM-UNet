"""Orchestrates all 15 real FINN builds (stitched IP + automatic OOC synth)
for the MVAU-type/resType/noAct/PE-SIMD combination matrix -- see
finn_build_probe_lutmult_noact0.py's and finn_build_probe_s12_context_
noact1_int8.py's own docstrings for the exact 6+9=15 combination count and
rationale.

Runs INSIDE the FINN container (needs qonnx/finn importable, Vivado on
PATH). Deploy alongside finn_build_probe_lutmult_noact0.py,
finn_build_probe_s12_context_noact1_int8.py, finn_ooc_probe_s12_context_
synth.py, dump_node_attrs.py, probe_lutmult_noact0_int8.onnx and
probe_noact1_single_d16_int8.onnx (all flat in the same ENET_DIR).

Runs up to NUM_WORKERS combos IN PARALLEL. Each worker "slot" gets its own
FINN_BUILD_DIR (/tmp/finn_dev_thelegendiv_slotN) so its synth_out_of_context_*
dirs never collide with another slot's -- this is what makes the before/after
dir-diffing safe under parallelism (each slot only ever sees its own dirs).
Slots process their assigned combos (round-robin) sequentially; a shared
lock guards the master CSV writes.

For each combo:
  1. runs the build script (which builds + stitches + auto-runs OOC synth),
     capturing stdout to logs/<label>.log
  2. parses OUTPUT_DIR= from that log
  3. diffs this slot's own FINN_BUILD_DIR/synth_out_of_context_* before/after
     to find this combo's synth project dir (the hash is random per run)
  4. runs a real Vivado `report_utilization -hierarchical -hierarchical_depth 4`
     on the routed .dcp (same technique as hardware/build_alpha025_
     calibration_csv.py's per-partition pipeline, just for a single-
     partition probe instead)
  5. runs dump_node_attrs.py on the stitched-ip onnx to get landed nodeattrs
  6. joins both into rows appended to mvau_variant_matrix_dataset.csv (same
     schema as hardware/probes/build_probe_calibration_csv.py's output, plus
     combo_label/no_activation/impl_style_requested/fold_strategy columns to
     distinguish the 15 combos), written incrementally after every combo so
     partial progress is never lost if a later combo crashes.

Usage: python3 run_variant_matrix.py [num_workers]
Writes into ./mvau_variant_matrix_run/ (logs/, reports/, the master CSV).
"""
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ENET_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
OUT_DIR = os.path.join(ENET_DIR, "mvau_variant_matrix_run")
LOG_DIR = os.path.join(OUT_DIR, "logs")
RPT_DIR = os.path.join(OUT_DIR, "reports")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RPT_DIR, exist_ok=True)

os.environ["PATH"] = os.pathsep.join([
    "/tools/Xilinx/Vitis_HLS/2022.2/bin", "/tools/Xilinx/Vivado/2022.2/bin", os.environ.get("PATH", ""),
])
os.environ.setdefault("XILINX_VIVADO", "/tools/Xilinx/Vivado/2022.2")
os.environ.setdefault("XILINX_HLS", "/tools/Xilinx/Vitis_HLS/2022.2")
os.environ.setdefault("HOME", "/tmp/home_dir")

FOLDS = ["pemh_simd1", "balanced", "pe1_simdmw"]

COMBOS = []
for res_type in ["dsp", "lut"]:
    for fold in FOLDS:
        COMBOS.append(dict(
            label=f"noact0_hls_{res_type}_{fold}",
            script="finn_build_probe_lutmult_noact0.py",
            args=["--res-type", res_type, "--fold", fold],
            no_activation=0, impl_style_requested="hls", fold=fold,
        ))
for fold in FOLDS:
    COMBOS.append(dict(
        label=f"noact1_auto_{fold}",
        script="finn_build_probe_s12_context_noact1_int8.py",
        args=["--impl-style", "auto", "--fold", fold],
        no_activation=1, impl_style_requested="auto", fold=fold,
    ))
for res_type in ["dsp", "lut"]:
    for fold in FOLDS:
        COMBOS.append(dict(
            label=f"noact1_hls_{res_type}_{fold}",
            script="finn_build_probe_s12_context_noact1_int8.py",
            args=["--impl-style", "hls", "--res-type", res_type, "--fold", fold],
            no_activation=1, impl_style_requested="hls", fold=fold,
        ))
assert len(COMBOS) == 15, len(COMBOS)

DTYPE_BITS_RE = re.compile(r"(\d+)$")


def dtype_bits(s):
    if not s:
        return None
    m = DTYPE_BITS_RE.search(s)
    return int(m.group(1)) if m else None


def parse_hier_report(rpt_path):
    """Same indentation-calibration technique as build_probe_calibration_csv.py
    (each row already reports its full subtree total -- only sum rows at the
    partition-wrapper's direct-child depth)."""
    lines = open(rpt_path).read().splitlines()
    depth0_indent = None
    depth1_indent = None
    row_re = re.compile(
        r"^\|(?P<inst> +[^\|]+?) +\|[^\|]*\|"
        r" +(?P<lut>\d+) +\| +(?P<llut>\d+) +\| +(?P<lutram>\d+) +\| +(?P<srl>\d+) +\|"
        r" +(?P<ff>\d+) +\| +(?P<bram36>\d+) +\| +(?P<bram18>\d+) +\| +(?P<uram>\d+) +\| +(?P<dsp>\d+) +\|$"
    )
    parsed = []
    for line in lines:
        m = row_re.match(line)
        if not m:
            continue
        raw_inst = m.group("inst")
        indent = len(raw_inst) - len(raw_inst.lstrip(" "))
        name = raw_inst.strip()
        parsed.append((indent, name, m))
        if depth0_indent is None:
            depth0_indent = indent
        elif depth1_indent is None and indent > depth0_indent:
            depth1_indent = indent
    if depth0_indent is None or depth1_indent is None:
        raise RuntimeError(f"could not calibrate indentation in {rpt_path}")
    step = depth1_indent - depth0_indent
    target_indent = depth1_indent + step
    result = {}
    for indent, name, m in parsed:
        if indent != target_indent:
            continue
        if name.startswith("(") and name.endswith(")"):
            continue  # "logic directly in this instance" pseudo-rows
        result[name] = {
            "real_LUT": int(m.group("lut")), "real_LUTRAM": int(m.group("lutram")),
            "real_SRL": int(m.group("srl")), "real_FF": int(m.group("ff")),
            "real_BRAM36": int(m.group("bram36")), "real_BRAM18": int(m.group("bram18")),
            "real_URAM": int(m.group("uram")), "real_DSP": int(m.group("dsp")),
        }
    return result


HEADER = [
    "combo_label", "no_activation", "impl_style_requested", "fold_strategy",
    "partition", "node_name", "op_type", "MH", "MW", "PE", "SIMD",
    "PE_folding_json", "SIMD_folding_json", "pe_simd_landed_matches_json",
    "weightDataType", "inputDataType", "outputDataType", "weight_bits", "act_bits",
    "resType", "force_dsp", "ram_style", "mem_mode", "PE_times_SIMD", "log2_MW",
    "real_LUT", "real_LUTRAM", "real_SRL", "real_FF", "real_BRAM36", "real_BRAM18",
    "real_URAM", "real_DSP",
]

MASTER_CSV = os.path.join(OUT_DIR, "mvau_variant_matrix_dataset.csv")
all_rows = []
rows_lock = threading.Lock()


def write_csv():
    with open(MASTER_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)


def process_combo(combo, index, build_dir, env):
    label = combo["label"]
    print(f"\n=== [{index + 1}/15] {label} (slot dir={build_dir}) ===", flush=True)
    log_path = os.path.join(LOG_DIR, f"{label}.log")
    before = set(glob.glob(os.path.join(build_dir, "synth_out_of_context_*")))
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            ["python3", "-u", combo["script"]] + combo["args"],
            cwd=ENET_DIR, stdout=logf, stderr=subprocess.STDOUT, env=env,
        )
    dt = time.time() - t0
    print(f"  [{label}] exit_code={proc.returncode}  elapsed={dt:.0f}s  log={log_path}", flush=True)
    if proc.returncode != 0:
        print(f"  [{label}] FAILED, skipping row extraction", flush=True)
        return

    log_text = open(log_path).read()
    m = re.search(r"OUTPUT_DIR=\s*(\S+)", log_text)
    if not m:
        print(f"  [{label}] Could not find OUTPUT_DIR in log, skipping", flush=True)
        return
    output_dir = m.group(1)

    after = set(glob.glob(os.path.join(build_dir, "synth_out_of_context_*")))
    new_dirs = after - before
    if len(new_dirs) != 1:
        print(f"  [{label}] WARNING: expected exactly 1 new synth_out_of_context dir, got {len(new_dirs)}: {new_dirs}", flush=True)
        if not new_dirs:
            return
    synth_dir = sorted(new_dirs)[-1]

    dcp = os.path.join(synth_dir, "results_finn_design_wrapper", "vivadocompile",
                        "vivadocompile.runs", "impl_1", "finn_design_wrapper_routed.dcp")
    if not os.path.exists(dcp):
        print(f"  [{label}] WARNING: routed dcp not found at {dcp}, skipping row extraction", flush=True)
        return

    rpt_path = os.path.join(RPT_DIR, f"{label}.rpt")
    tcl_path = os.path.join(RPT_DIR, f"{label}.tcl")
    with open(tcl_path, "w") as f:
        f.write(f"open_checkpoint {dcp}\n")
        f.write(f"report_utilization -hierarchical -hierarchical_depth 4 -file {rpt_path}\n")
    vivado_cmd = (
        "source /tools/Xilinx/Vivado/2022.2/settings64.sh && "
        f"vivado -mode batch -source {tcl_path} -nolog -nojournal"
    )
    vivado_log = os.path.join(RPT_DIR, f"{label}_vivado.log")
    with open(vivado_log, "w") as vf:
        subprocess.run(["bash", "-c", vivado_cmd], stdout=vf, stderr=subprocess.STDOUT, env=env)

    if not os.path.exists(rpt_path):
        print(f"  [{label}] WARNING: hierarchical report not generated", flush=True)
        return

    stitched_onnx = os.path.join(output_dir, "intermediate_models", "step_create_stitched_ip.onnx")
    attrs_path = os.path.join(RPT_DIR, f"{label}_attrs.json")
    subprocess.run(["python3", os.path.join(ENET_DIR, "dump_node_attrs.py"), stitched_onnx, attrs_path], check=False, env=env)
    if not os.path.exists(attrs_path):
        print(f"  [{label}] WARNING: attrs json not generated", flush=True)
        return

    hier = parse_hier_report(rpt_path)
    attrs = json.loads(open(attrs_path).read())

    new_rows = []
    for node in attrs:
        name = node["node_name"]
        a = node["attrs"]
        res = hier.get(name)
        if res is None:
            print(f"  [{label}] WARNING: no hier row for node {name}", flush=True)
            continue
        if node["op_type"] in ("VVAU_hls", "VVAU_rtl"):
            mh = a.get("Channels")
            kernel = a.get("Kernel") or [1, 1]
            mw = kernel[0] * kernel[1]
        else:
            mh = a.get("MH")
            mw = a.get("MW")
        row = {
            "combo_label": label, "no_activation": combo["no_activation"],
            "impl_style_requested": combo["impl_style_requested"], "fold_strategy": combo["fold"],
            "partition": 0, "node_name": name, "op_type": node["op_type"],
            "MH": mh, "MW": mw, "PE": a.get("PE"), "SIMD": a.get("SIMD"),
            "PE_folding_json": a.get("PE"), "SIMD_folding_json": a.get("SIMD"),
            "pe_simd_landed_matches_json": True,
            "weightDataType": a.get("weightDataType"), "inputDataType": a.get("inputDataType"),
            "outputDataType": a.get("outputDataType"),
            "weight_bits": dtype_bits(a.get("weightDataType")), "act_bits": dtype_bits(a.get("inputDataType")),
            "resType": a.get("resType"), "force_dsp": a.get("resType") == "dsp",
            "ram_style": a.get("ram_style"), "mem_mode": a.get("mem_mode"),
            "PE_times_SIMD": (a.get("PE") or 0) * (a.get("SIMD") or 0),
            "log2_MW": round(math.log2(mw), 2) if mw else None,
        }
        row.update(res)
        new_rows.append(row)

    with rows_lock:
        all_rows.extend(new_rows)
        write_csv()
        print(f"  [{label}] wrote {len(new_rows)} rows ({len(all_rows)} total so far) to {MASTER_CSV}", flush=True)


def worker_slot(worker_id, combos_subset):
    build_dir = f"/tmp/finn_dev_thelegendiv_slot{worker_id}"
    os.makedirs(build_dir, exist_ok=True)
    env = os.environ.copy()
    env["FINN_BUILD_DIR"] = build_dir
    for index, combo in combos_subset:
        process_combo(combo, index, build_dir, env)


print(f"Running {len(COMBOS)} combos with {NUM_WORKERS} parallel worker slots", flush=True)
slots = [[] for _ in range(NUM_WORKERS)]
for i, combo in enumerate(COMBOS):
    slots[i % NUM_WORKERS].append((i, combo))

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(worker_slot, wid, subset) for wid, subset in enumerate(slots) if subset]
    for f in futures:
        f.result()

print("\n=== ALL 15 COMBOS DONE ===", flush=True)
print(f"Master CSV: {MASTER_CSV}")


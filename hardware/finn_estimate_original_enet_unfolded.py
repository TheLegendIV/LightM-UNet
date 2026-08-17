"""FINN fully-unfolded analytical estimate + per-node breakdown for the
'original' (full-scale) 8-bit ENet (no PReLU, no MaxUnpool) -- side job, run
2026-08-17, independent of the running S19 partitioned IP-build job.

'Fully unfolded': no SetFolding / no folding config file is applied (the
build config's target_fps=None makes step_target_fps_parallelization a
no-op, per finn/builder/build_dataflow_config.py's _resolve_cycles_per_frame
returning None when target_fps is None) -- every fpgadataflow node keeps its
FINN-default PE=SIMD=1 (maximum folding / minimum parallelism / smallest
hardware footprint, one MAC unit doing all the work serially). No FIFO depth
sizing, no HLS/Vitis HLS synthesis, no Vivado -- this only runs FINN's own
purely-analytical estimate step (step_generate_estimate_reports), matching
hardware/finn_resource_probe.py's estimate_only_dataflow_steps pattern but
using ENet's own custom tidy/streamline/convert_to_hw steps (residual-Add +
ConvTranspose handling), imported from hardware/finn_enet_build.py.

Usage (inside the FINN container):
    python3 finn_estimate_original_enet_unfolded.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# finn_enet_build.py sets up FINN src paths + Xilinx PATH env vars at import
# time and defines the reusable step_enet_tidy/streamline/convert_to_hw
# functions -- only its `if __name__ == "__main__":` guard is skipped here.
import finn_enet_build as enet_build  # noqa: E402

import finn.builder.build_dataflow as build  # noqa: E402
import finn.builder.build_dataflow_config as build_cfg  # noqa: E402
from finn.builder.build_dataflow_config import DataflowBuildConfig  # noqa: E402
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames  # noqa: E402
import qonnx.custom_op.registry as registry  # noqa: E402
from finn.util.fpgadataflow import is_hls_node, is_rtl_node  # noqa: E402


def step_reapply_unique_names(model: ModelWrapper, cfg: DataflowBuildConfig):
    """SpecializeLayers (in step_specialize_layers) creates new HLS/RTL-variant
    nodes WITHOUT calling GiveUniqueNodeNames afterward, leaving every node's
    .name == "" -- which silently collapses all per-node dicts in FINN's own
    op_and_param_counts/exp_cycles_per_layer/res_estimation analyses down to a
    single overwritten entry (confirmed empirically: report JSONs from a first
    run of this script had exactly 2 keys total, "" and "total"). Re-apply
    GiveUniqueNodeNames right after specialize_layers to fix this before any
    of the report-generating/breakdown steps run."""
    return model.transform(GiveUniqueNodeNames())

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
# 2026-08-17: corrected-width (16,64,128,64,16) re-export at 512x512 input
# (finn_export_original_enet_int8.py), confirmed via node/op-histogram
# inspection -- Conv/BatchNormalization/ConvTranspose/MaxPool/Quant/Relu only,
# no Resize/Upsample/MaxUnpool/PRelu -- FINN-compatible as-is.
MODEL_NAME = "quantEnet_original_int8"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")

from datetime import datetime  # noqa: E402
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"estimates_unfolded_{MODEL_NAME}_{timestamp}")

FPGA_PART = "xczu7ev-ffvc1156-2-e"
SYNTH_CLK_NS = 5.0  # 200 MHz, matches hardware/finn_resource_probe.py's default

# Real xczu7ev-ffvc1156-2-e (Zynq UltraScale+ ZU7EV) numbers per Xilinx DS891:
# 504,000 LUTs (230,400 usable as reported by FINN's own LUT estimator
# convention elsewhere in this repo), 1,728 DSP slices, 312 BRAM_36K tiles
# (=624 in FINN's BRAM_18K reporting granularity).
XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624, "DSP": 1728}

enet_estimate_unfolded_steps = [
    "step_qonnx_to_finn",
    enet_build.step_enet_tidy,
    enet_build.step_enet_streamline,
    enet_build.step_enet_convert_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",       # HW -> HLS/RTL backend variants; required before estimate reports
    step_reapply_unique_names,      # fix: SpecializeLayers resets node names to "" (see docstring above)
    # NOTE: NO step_target_fps_parallelization, NO step_apply_folding_config --
    # fully unfolded (PE=SIMD=1 default), per this script's docstring.
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
]

cfg_estimates = DataflowBuildConfig(
    output_dir=OUTPUT_DIR,
    target_fps=None,  # <- fully unfolded: makes step_target_fps_parallelization a no-op
    synth_clk_period_ns=SYNTH_CLK_NS,
    fpga_part=FPGA_PART,
    steps=enet_estimate_unfolded_steps,
    generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
    save_intermediate_models=True,
)


# ---------------------------------------------------------------------------
# Per-node breakdown (MAC/BOP, memory bits, SWU buffer size, PE/SIMD/parallel_window)
# ---------------------------------------------------------------------------
def _bits_from_param_key(key: str) -> int | None:
    """FINN's get_op_and_param_counts() key convention is literally
    'param_<kind>_<N>b' (e.g. 'param_weight_8b', 'param_threshold_16b') --
    NOT a DataType name -- see finn/custom_op/fpgadataflow/matrixvectoractivation.py's
    get_op_and_param_counts(). Parse the trailing '<N>b' suffix directly."""
    suffix = key.rsplit("_", 1)[-1]
    if suffix.endswith("b") and suffix[:-1].isdigit():
        return int(suffix[:-1])
    return None


def build_node_breakdown(model: ModelWrapper, opcounts: dict, cycles: dict, res: dict) -> list[dict]:
    """One row per fpgadataflow (HLS/RTL) node: PE, SIMD, parallel_window (FINN's
    closest attribute to 'output-pixel replication'; FINN has no separate 'M'/MMV
    factor at the Python level in this version -- see report), MAC count, weight/
    threshold memory in bits, SWU (ConvolutionInputGenerator) buffer size in bits
    where applicable, and BRAM_18K/LUT/DSP estimates.

    opcounts/cycles/res are the dicts loaded DIRECTLY from FINN's own
    op_and_param_counts.json / estimate_layer_cycles.json / estimate_layer_resources.json
    report files (same authoritative source step_generate_estimate_reports itself
    writes) -- NOT recomputed here, to avoid any risk of node-name-collision bugs
    (see step_reapply_unique_names docstring: SpecializeLayers resets node.name to
    '' and, if GiveUniqueNodeNames isn't re-applied before analysis, per-node dicts
    silently collapse into a single overwritten '' entry)."""
    rows = []
    for node in model.graph.node:
        if not (is_hls_node(node) or is_rtl_node(node)):
            continue
        inst = registry.getCustomOp(node)
        name = node.name

        def _attr(key):
            try:
                return inst.get_nodeattr(key)
            except Exception:
                return None

        pe = _attr("PE")
        simd = _attr("SIMD")
        parallel_window = _attr("parallel_window")

        node_counts = opcounts.get(name, {})
        mac_count = 0
        param_bits = 0
        for k, v in node_counts.items():
            if k.startswith("op_mac"):
                mac_count += v
            elif k.startswith("param_"):
                bits = _bits_from_param_key(k)
                if bits is not None:
                    param_bits += v * bits

        swu_buffer_bits = None
        if node.op_type.startswith("ConvolutionInputGenerator"):
            try:
                depth_elems = inst.get_buffer_depth()
                idt_bits = inst.get_input_datatype().bitwidth()
                swu_buffer_bits = depth_elems * (simd or 1) * idt_bits
            except Exception:
                swu_buffer_bits = None

        node_res = res.get(name, {})
        rows.append({
            "node_name": name,
            "op_type": node.op_type,
            "PE": pe,
            "SIMD": simd,
            "parallel_window": parallel_window,
            "mac_count": mac_count,
            "bop_count": mac_count * 8 * 8 if mac_count else 0,  # BOPs = MACs x weight_bits x act_bits (8x8 uniform)
            "param_memory_bits": param_bits,
            "param_memory_Mbits": param_bits / 1e6,
            "swu_buffer_bits": swu_buffer_bits,
            "swu_buffer_Mbits": (swu_buffer_bits / 1e6) if swu_buffer_bits else None,
            "exp_cycles": cycles.get(name),
            "BRAM_18K": node_res.get("BRAM_18K"),
            "LUT": node_res.get("LUT"),
            "DSP": node_res.get("DSP"),
        })
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    import csv
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    print(f"Model : {MODEL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Steps : {[s if isinstance(s, str) else s.__name__ for s in enet_estimate_unfolded_steps]}")
    print(f"Fully unfolded: target_fps=None -> PE=SIMD=1 everywhere (no SetFolding)")
    print()

    build.build_dataflow_cfg(MODEL_FILE, cfg_estimates)

    # ---- FINN's own network-level reports ----
    report_dir = os.path.join(OUTPUT_DIR, "report")
    perf_json = os.path.join(report_dir, "estimate_network_performance.json")
    res_json = os.path.join(report_dir, "estimate_layer_resources.json")

    print("\n" + "=" * 70)
    print("NETWORK PERFORMANCE ESTIMATES (fully unfolded, PE=SIMD=1)")
    print("=" * 70)
    if os.path.exists(perf_json):
        perf = json.loads(Path(perf_json).read_text())
        for k, v in perf.items():
            print(f"  {k}: {v}")
    else:
        print(f"  [WARN] not found: {perf_json}")

    total_lut = total_bram = total_dsp = 0
    if os.path.exists(res_json):
        res = json.loads(Path(res_json).read_text())
        # NOTE: res already contains a "total" key (aggregate_dict_keys' output,
        # written by step_generate_estimate_reports) -- use it directly rather
        # than summing all values (which would double-count "total" itself).
        totals = res.get("total", {})
        total_lut = totals.get("LUT", 0)
        total_bram = totals.get("BRAM_18K", 0)
        total_dsp = totals.get("DSP", 0)

    print("\n" + "=" * 70)
    print(f"TOTAL RESOURCE ESTIMATE vs {FPGA_PART}")
    print("=" * 70)
    print(f"  LUT      : {total_lut} / {XCZU7EV['LUT']} ({100*total_lut/XCZU7EV['LUT']:.1f}%)")
    print(f"  BRAM_18K : {total_bram} / {XCZU7EV['BRAM_18K']} ({100*total_bram/XCZU7EV['BRAM_18K']:.1f}%)")
    print(f"  DSP      : {total_dsp} / {XCZU7EV['DSP']} ({100*total_dsp/XCZU7EV['DSP']:.1f}%)")
    fits = total_lut <= XCZU7EV["LUT"] and total_bram <= XCZU7EV["BRAM_18K"] and total_dsp <= XCZU7EV["DSP"]
    print(f"  fits_bool: {fits}")

    # ---- Per-node breakdown ----
    # Use FINN's own report JSONs (authoritative, already correctly per-node-keyed
    # at this point since step_reapply_unique_names ran before generate_estimate_reports)
    # for MAC/param/cycle/resource data, and load the final pre-report checkpoint
    # only to read PE/SIMD/parallel_window nodeattrs + compute SWU buffer sizes.
    opcounts_json = os.path.join(report_dir, "op_and_param_counts.json")
    cycles_json = os.path.join(report_dir, "estimate_layer_cycles.json")
    opcounts = json.loads(Path(opcounts_json).read_text()) if os.path.exists(opcounts_json) else {}
    cycles = json.loads(Path(cycles_json).read_text()) if os.path.exists(cycles_json) else {}
    res_by_node = json.loads(Path(res_json).read_text()) if os.path.exists(res_json) else {}

    intermediate_dir = os.path.join(OUTPUT_DIR, "intermediate_models")
    minimize_ckpt = os.path.join(intermediate_dir, "step_minimize_bit_width.onnx")
    if os.path.exists(minimize_ckpt) and len(opcounts) > 1:
        model = ModelWrapper(minimize_ckpt)
        rows = build_node_breakdown(model, opcounts, cycles, res_by_node)
        csv_path = os.path.join(OUTPUT_DIR, "per_node_breakdown.csv")
        write_csv(rows, csv_path)
        print("\n" + "=" * 70)
        print(f"PER-NODE BREAKDOWN ({len(rows)} HW nodes) -- written to {csv_path}")
        print("=" * 70)
        for r in rows:
            print(
                f"  {r['node_name']:35s} {r['op_type']:28s} "
                f"PE={r['PE']} SIMD={r['SIMD']} par_win={r['parallel_window']} "
                f"MACs={r['mac_count']:>10} BOPs={r['bop_count']:>12} "
                f"param={r['param_memory_Mbits']:.4f}Mb "
                f"swu={r['swu_buffer_Mbits']}Mb "
                f"cyc={r['exp_cycles']} "
                f"BRAM18K={r['BRAM_18K']} LUT={r['LUT']} DSP={r['DSP']}"
            )
        total_macs = sum(r["mac_count"] for r in rows)
        total_bops = sum(r["bop_count"] for r in rows)
        total_param_mbits = sum(r["param_memory_Mbits"] for r in rows)
        total_swu_mbits = sum(r["swu_buffer_Mbits"] or 0 for r in rows)
        print(f"\n  TOTAL MACs={total_macs}  BOPs={total_bops}  "
              f"param_mem={total_param_mbits:.3f}Mb  swu_mem={total_swu_mbits:.3f}Mb")
    else:
        print(f"\n[WARN] checkpoint not found ({minimize_ckpt}) or op_and_param_counts.json "
              f"still shows a node-name collision (len={len(opcounts)}) -- per-node breakdown skipped.")

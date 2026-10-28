"""Pre-build dry run for one 8-way partition (container side, no HLS/Vivado): runs the same bridge -> specialize ->
fold -> weight bit-width steps as finn_s12_build.py and reports, per weight node, the logical name, landed PE/SIMD vs
the MILP, landed weight dtype vs the MILP weight_bits, and the weight value range. Exits nonzero on any failure.

    docker exec -e HOME=/tmp/home_dir <c> bash -c 'cd /home/thelegendiv/finn/notebooks/enet && \\
        python3 check_partition_bridge.py <preamble_dir> --conv-order C --folding-json F --partition 1'
"""
import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
_p = argparse.ArgumentParser()
_p.add_argument("preamble_dir")
_p.add_argument("--conv-order", required=True)
_p.add_argument("--folding-json", required=True)
_p.add_argument("--partition", type=int, default=1)
_p.add_argument("--out", default="/tmp/check_partition_bridge_out")
_args = _p.parse_args()

_real_argv, sys.argv = sys.argv, sys.argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers, step_target_fps_parallelization, step_apply_folding_config)
from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402
import flow_gates  # noqa: E402
from finn_s12_build_steps import (  # noqa: E402
    WEIGHT_OP_TYPES, _resolve_folding_entry, build_partition_folding_config, install_relaxed_stage_boundaries,
    load_partition_logical_names, step_fix_weight_dtype_bipolar_bug, step_minimize_bit_width_standalone_thresh_aware)

install_relaxed_stage_boundaries()
idx = _args.partition
os.makedirs(os.path.join(_args.out, "report"), exist_ok=True)
cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=_args.out)
folding = json.load(open(_args.folding_json))
per_layer, extra_nodes = folding["per_layer"], folding.get("extra_nodes")

parent = step_create_dataflow_partition_multi(ModelWrapper(os.path.join(
    _args.preamble_dir, "intermediate_models", "assign_stage_partition_ids_8way.onnx")), cfg)
sdp = parent.get_nodes_by_op_type("StreamingDataflowPartition")[idx]
fn = getCustomOp(sdp).get_nodeattr("model")
logical = load_partition_logical_names(_args.preamble_dir, _args.conv_order)[idx][0]
fc, _ = build_partition_folding_config(fn, sdp.name, logical, per_layer, cfg, tag=f"p{idx}", extra_nodes=extra_nodes,
                                       conv_order_file=_args.conv_order)
fc_file = os.path.join(_args.out, f"hawq_folding_config_partition{idx}.json")
json.dump(fc, open(fc_file, "w"), indent=2)
cfg = dataclasses.replace(cfg, folding_config_file=fc_file)

# same step order as finn_s12_build._build_one_partition up to the weight bit-width gate
m = ModelWrapper(fn)
m = step_specialize_layers(m, cfg)
m = m.transform(GiveUniqueNodeNames())
m = m.transform(GiveReadableTensorNames())
m = step_target_fps_parallelization(m, cfg)
m = step_apply_folding_config(m, cfg)
label = f"[check p{idx}]"
fails = []


def _gate(fn_, *a):
    try:
        fn_(*a)
    except AssertionError as e:
        fails.append(str(e))
        print(e)


_gate(flow_gates.gate_folding_landed, m, fc_file, label)
m = step_minimize_bit_width_standalone_thresh_aware(m, cfg)
m = step_fix_weight_dtype_bipolar_bug(m, cfg)
_gate(flow_gates.gate_rtl_weights_signed, m, label)

nodes = [n for n in m.graph.node if n.op_type in WEIGHT_OP_TYPES]
assert len(nodes) == len(logical), (len(nodes), len(logical))
print(f"\n{'node':16s} {'logical':26s} {'PE/SIMD landed':>14s} {'MILP':>9s} {'wdt':>6s} {'MILP wb':>7s}  weight range / producer")
for node, ln in zip(nodes, logical):
    inst = getCustomOp(node)
    pe, simd = inst.get_nodeattr("PE"), inst.get_nodeattr("SIMD")
    entry, key = _resolve_folding_entry(ln, per_layer, extra_nodes)
    wdt = inst.get_nodeattr("weightDataType")
    w = m.get_initializer(node.input[1])
    if entry is None:
        fails.append(f"{node.name} ({ln}): no MILP entry")
        print(f"{node.name:16s} {ln:26s} {pe:>6d}/{simd:<7d} {'-':>9s} {wdt:>6s} {'-':>7s}  NO MILP ENTRY")
        continue
    ce = entry["vvau"] if entry.get("node_type") == "depthwise_vvau_slot" else entry
    want_pe, want_simd, wb = ce["pe"], ce["simd"], ce.get("weight_bits")
    note = ""
    if (pe, simd) != (want_pe, want_simd):
        mh, mw = inst.get_nodeattr("MH"), inst.get_nodeattr("MW")
        if mh % want_pe == 0 and mw % want_simd == 0:
            fails.append(f"{node.name} ({ln}): landed PE/SIMD {pe}/{simd} != MILP {want_pe}/{want_simd}")
            note += " FOLDING MISMATCH"
        else:
            note += f" (MILP clamped, MH={mh} MW={mw})"
    bits = int(wdt[3:]) if wdt.startswith("INT") else None
    if bits is None:
        fails.append(f"{node.name} ({ln}): unsigned/non-INT weight dtype {wdt}")
        note += " NOT SIGNED"
    elif wb is not None and bits > wb + (1 if entry.get("kind") == "pad_mvau" else 0):
        fails.append(f"{node.name} ({ln}): landed {wdt} wider than MILP weight_bits {wb}")
        note += " BITWIDTH"
    elif entry.get("kind") == "pad_mvau" and bits != wb:
        fails.append(f"{node.name} ({ln}): pad weights {wdt}, MILP expects INT{wb}")
        note += " PAD BITWIDTH"
    prod = ""
    if entry.get("kind") == "pad_mvau":
        p = m.find_producer(node.input[0])
        while p is not None and p.op_type.startswith("StreamingDataWidthConverter"):
            p = m.find_producer(p.input[0])
        prod = f" producer={p.op_type if p else None}"
        if p is None or "MaxPool" not in p.op_type:
            fails.append(f"{node.name} ({ln}): pad MVAU not fed by MaxPool")
        if sorted(set(w.flatten().tolist())) != [0, 127]:
            fails.append(f"{node.name} ({ln}): pad weights not {{0,127}}: {sorted(set(w.flatten().tolist()))}")
    print(f"{node.name:16s} {ln:26s} {pe:>6d}/{simd:<7d} {want_pe:>4d}/{want_simd:<4d} {wdt:>6s} {str(wb):>7s}  "
          f"[{w.min():g},{w.max():g}]{prod}{note}")

print(f"\n{label} {len(nodes)} weight node(s), {len(fails)} failure(s)")
for f in fails:
    print("  FAIL:", f)
sys.exit(1 if fails else 0)

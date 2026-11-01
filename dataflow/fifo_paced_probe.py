"""FIFO-sizing probe: FINN's rtlsim FIFO autosize with a PACED source instead of the saturated one.

Runs the unmodified finn_s12_build.py per-partition flow (same specialize / auto-fold / ipgen / InsertAndSetFIFODepths
steps) but (a) swaps the Verilator FIFO-sizing driver for a paced one and (b) stops right after the
partition_<i>_postfifo_autosize.onnx checkpoint is saved (no SplitLargeFIFOs / stitch / OOC synth).

Why not force_python_rtlsim=True: that path (pyverilator + the PATH Verilator 4.224) cannot build the alpha MVU RTL.
Its source model is reproduced in the C++ driver instead: the input word k is offered no earlier than cycle
floor(k * PACE_SLACK * max_cycles / words_per_image), where max_cycles is the partition's dataflow_performance
max_cycles (the same quantity FINN's python path uses to derive ncycles_per_input). tvalid is held until accepted
(AXI-stream), and the number of cycles the source was held off by tready is reported as src_blocked_cycles.

Usage (container, same env as _launch_finn_auto.sh):
    python3 -u fifo_paced_probe.py <preamble_dir> --tag <new tag> --target-fps 150 --partitions 1 2 6 \
        --probe --fifo-autosize rtlsim
Env: FIFOSIM_SOURCE=paced|saturated (default paced; saturated = stock driver, as a control), PACE_SLACK (default 1.0),
FIFOSIM_N_INPUTS (default: FINN's rule, 2 for conv nets).
"""
import copy
import os
import sys
import tempfile

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

import finn_s12_build as fsb  # noqa: E402  (parses sys.argv at import)
import finn_partition_build_steps as fpbs  # noqa: E402
import finn.transformation.fpgadataflow.set_fifo_depths as sfd  # noqa: E402
import finn.builder.build_dataflow_steps as bds  # noqa: E402
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance  # noqa: E402
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles  # noqa: E402

SOURCE = os.environ.get("FIFOSIM_SOURCE", "paced")
PACE_SLACK = float(os.environ.get("PACE_SLACK", "1.0"))
N_INPUTS_OVERRIDE = os.environ.get("FIFOSIM_N_INPUTS")


class StopAfterCheckpoint(Exception):
    pass


def _patched_template(max_cycles, words_per_image):
    """Copy of FINN's verilator_fifosim.cpp with a paced source; returns a FINN_ROOT stand-in dir."""
    real_root = os.environ["FINN_ROOT"]
    with open(real_root + "/src/finn/qnn-data/cpp/verilator_fifosim.cpp") as f:
        t = f.read()

    def sub(old, new):
        nonlocal t
        assert t.count(old) == 1, f"template anchor not found exactly once: {old!r}"
        t = t.replace(old, new)

    period_num = int(round(max_cycles * PACE_SLACK))
    sub("    top->s_axis_0_tvalid = 1;\n",
        "    top->s_axis_0_tvalid = 1;\n"
        "    unsigned long long src_blocked = 0;\n"
        f"    const unsigned long long pace_num = {period_num}ULL, pace_den = {int(words_per_image)}ULL;\n")
    # after every accepted word, drop tvalid and re-offer the next word once its release cycle has passed
    sub("            n_in_txns++;\n",
        "            n_in_txns++;\n            top->s_axis_0_tvalid = 0;\n")
    sub("        if(top->m_axis_0_tvalid == 1) {\n            n_out_txns++;",
        "        else if(top->s_axis_0_tvalid == 1) { src_blocked++; }\n"
        "        if(top->s_axis_0_tvalid == 0 && n_in_txns < n_iters_per_input * n_inputs "
        "&& (unsigned long long)iters >= (n_in_txns * pace_num) / pace_den) { top->s_axis_0_tvalid = 1; }\n"
        "        if(top->m_axis_0_tvalid == 1) {\n            n_out_txns++;")
    sub('    results_file << "N" << "\\t" << n_inputs << endl;\n',
        '    results_file << "N" << "\\t" << n_inputs << endl;\n'
        '    results_file << "src_blocked_cycles" << "\\t" << src_blocked << endl;\n')

    root = tempfile.mkdtemp(prefix="finn_root_paced_")
    cpp_dir = os.path.join(root, "src/finn/qnn-data/cpp")
    os.makedirs(cpp_dir)
    with open(os.path.join(cpp_dir, "verilator_fifosim.cpp"), "w") as f:
        f.write(t)
    os.symlink(real_root + "/finn-rtllib", os.path.join(root, "finn-rtllib"))
    return root


def paced_fifosim(model, n_inputs, max_iters=100000000):
    """Drop-in for verilator_fifosim_v5 (InsertAndSetFIFODepths 'largefifo_rtlsim')."""
    if N_INPUTS_OVERRIDE:
        n_inputs = int(N_INPUTS_OVERRIDE)
    if SOURCE == "saturated":
        return fpbs.verilator_fifosim_v5(model, n_inputs, max_iters)

    from qonnx.custom_op.registry import getCustomOp
    import numpy as np

    perf_model = copy.deepcopy(model).transform(AnnotateCycles())
    max_cycles = int(perf_model.analysis(dataflow_performance)["max_cycles"])
    proj = fpbs.prepare_stitched_ip_for_verilator(model)
    first = model.find_consumer(model.graph.input[0].name)
    last = model.find_producer(model.graph.output[0].name)
    ishape = getCustomOp(first).get_folded_input_shape()
    oshape = getCustomOp(last).get_folded_output_shape()
    words = int(np.prod(ishape[:-1]))
    print(f"[paced source] max_cycles={max_cycles} words/image={words} slack={PACE_SLACK} "
          f"-> {max_cycles * PACE_SLACK / words:.3f} cycles/word, n_inputs={n_inputs}", flush=True)

    templ = '    results_file << "maxcount%s" << "\\t" << to_string(top->maxcount%s) << endl;'
    log_lines, ind = [], 0
    for fifo_node in model.get_nodes_by_op_type("StreamingFIFO_rtl"):
        if getCustomOp(fifo_node).get_nodeattr("depth_monitor") == 1:
            suffix = "" if ind == 0 else "_%d" % ind
            log_lines.append(templ % (suffix, suffix))
            ind += 1

    real_root = os.environ["FINN_ROOT"]
    os.environ["FINN_ROOT"] = _patched_template(max_cycles, words)
    try:
        # the swg_pkg path is taken from FINN_ROOT too -> finn-rtllib is symlinked into the stand-in root
        res = fpbs.run_verilator_fifosim(
            proj, "finn_design_wrapper", n_inputs, ishape, oshape, max_iters,
            fifo_depth_logging="\n".join(log_lines),
        )
    finally:
        os.environ["FINN_ROOT"] = real_root
    print(f"[paced source] done: cycles={res.get('cycles')} src_blocked_cycles={res.get('src_blocked_cycles')} "
          f"latency={res.get('latency_cycles')} out={res.get('N_OUT_TXNS')}/{res.get('expected_out_txns')}", flush=True)
    return res


sfd.verilator_fifosim = paced_fifosim
bds.verilator_fifosim = paced_fifosim
fsb._sfd.verilator_fifosim = paced_fifosim

# stop each partition right after its FIFO-autosize checkpoint has been written
_orig_split = fsb.SplitLargeFIFOs


def _stop(*a, **k):
    raise StopAfterCheckpoint()


fsb.SplitLargeFIFOs = _stop
_orig_build_one = fsb._build_one_partition


def _build_one_then_stop(*args):
    try:
        return _orig_build_one(*args)
    except StopAfterCheckpoint:
        print(f"[partition {args[4]}] FIFO checkpoint saved, stopping (probe)", flush=True)
        return None


fsb._build_one_partition = _build_one_then_stop

if __name__ == "__main__":
    print(f"FIFO-sim source: {SOURCE}", flush=True)
    fsb.main()

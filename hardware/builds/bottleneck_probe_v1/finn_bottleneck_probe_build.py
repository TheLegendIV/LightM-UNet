"""FINN build of ONE bottleneck probe case, run INSIDE the FINN container (UNTESTED: written without access to a FINN
container, see FINN_AGENT_HANDOFF.md -- expect to fix FINN-API details on the first run).

Goal of the job: prove that the tightly coupled bottleneck (our folding + our small FIFOs, forced) does NOT deadlock in
hardware and reaches the required throughput. So the primary result is the stitched-IP rtlsim (throughput, latency, no
deadlock); OOC synthesis is an optional second stage (--ooc).

Flow (one block = one partition, no 8-way machinery):
  qonnx_to_finn -> live enet tidy / streamline / forked-dequant fixes -> compose consecutive thresholds (unless --no-merge)
  -> live RTL-MVAU convert_to_hw (standalone Thresholding, noActivation=1) -> dataflow partition -> specialize layers
  -> apply role-keyed folding from <case>_folding.json -> min-bitwidth (threshold aware) -> force resType=dsp
  -> codegen + ipgen -> stock set_fifo_depths (largefifo_rtlsim) -> force OUR FIFO depths -> SplitLargeFIFOs + re-ipgen
  -> stitched IP -> rtlsim (several frames, stable throughput) -> optional OOC synth.

Run (see run_probes.sh):
    docker exec -e HOME=/tmp/home_dir <container> bash -c 'source /tools/Xilinx/Vivado/2022.2/settings64.sh && \\
        cd /home/thelegendiv/finn/notebooks/enet && python3 finn_bottleneck_probe_build.py bottleneck_cin32_d8_int4'
Inputs (docker cp'd flat into the enet dir): <case>.onnx, <case>_probe.json, <case>_folding.json.
"""
import argparse
import dataclasses
import json
import os
import sys
import time

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
sys.path.insert(0, ENET_DIR)

_XILINX_BIN_DIRS = ["/tools/Xilinx/Vitis_HLS/2022.2/bin", "/tools/Xilinx/Vivado/2022.2/bin"]
os.environ["PATH"] = os.pathsep.join(_XILINX_BIN_DIRS + [os.environ.get("PATH", "")])
os.environ.setdefault("XILINX_VIVADO", "/tools/Xilinx/Vivado/2022.2")
os.environ.setdefault("XILINX_HLS", "/tools/Xilinx/Vitis_HLS/2022.2")

# importing finn_enet_build FIRST puts finn / qonnx / brevitas source trees on sys.path (finn is a source install)
from finn_enet_build import step_enet_streamline, step_enet_tidy  # noqa: E402
from finn_enet_build_fixups import (  # noqa: E402
    _fixup_degenerate_signed_bias,
    step_absorb_leftover_scale_before_matmul,
    step_dedup_forked_matmul_before_threshold,
    step_fuse_forked_dequant_into_duplicate_threshold,
    step_fuse_leaky_relu_to_threshold,
)
from finn_enet_convert_to_hw_rtl_mvau import step_enet_convert_to_hw_rtl_mvau  # noqa: E402
from finn_compose_thresholds import step_compose_consecutive_thresholds  # noqa: E402
from finn_s12_build_steps import (  # noqa: E402
    step_fix_weight_dtype_bipolar_bug,
    step_force_dsp,
    step_minimize_bit_width_standalone_thresh_aware,
)

import finn.builder.build_dataflow_config as build_cfg  # noqa: E402
from finn.builder.build_dataflow_config import DataflowBuildConfig  # noqa: E402
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_create_dataflow_partition,
    step_hw_codegen,
    step_hw_ipgen,
    step_measure_rtlsim_performance,
    step_qonnx_to_finn,
    step_set_fifo_depths,
    step_specialize_layers,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP  # noqa: E402
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP  # noqa: E402
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP  # noqa: E402
from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs  # noqa: E402
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext  # noqa: E402
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames  # noqa: E402

FPGA_PART = "xczu7ev-ffvc1156-2-e"
CLK_NS = 10.0
SKIP_OPS = ("StreamingFIFO", "StreamingDataWidthConverter")   # transparent when walking the graph


# ------------------------------------------------------------------ role identification (structural, no node names)
def _is_transparent(node) -> bool:
    return node.op_type.startswith(SKIP_OPS)


def _producer(model, tensor):
    return model.find_producer(tensor)


def _real_producer(model, node):
    """Producer of node's first input, skipping FIFO/DWC nodes."""
    p = _producer(model, node.input[0])
    while p is not None and _is_transparent(p):
        p = _producer(model, p.input[0])
    return p


def _real_consumer(model, node):
    c = model.find_consumer(node.output[0])
    while c is not None and _is_transparent(c):
        c = model.find_consumer(c.output[0])
    return c


def identify_roles_down(model, probe: dict) -> dict:
    """Downsampling block (dn_bottleneck.py): role -> node for dup, swg_r, mvau_r, thr_r, fmpad, swg_m, mvau_m, thr_m, mvau_e,
    thr_e, maxpool, [mvau_s], thr_s, [fmpad_c], add, thr_out, thr_in. The channel-pad FMPadding (fmpad_c, padding on the regrouped
    axis [0,0,0,p]) is told apart from the 3x3 spatial one (padding all ones)."""
    cin, cmid, cout = probe["cin"], probe["cmid"], probe["cout"]
    roles = {}
    for n in model.graph.node:
        inst = getCustomOp(n)
        if "MVAU" in n.op_type:
            mw, mh = inst.get_nodeattr("MW"), inst.get_nodeattr("MH")
            key = {(4 * cin, cmid): "mvau_r", (9 * cmid, cmid): "mvau_m", (cmid, cout): "mvau_e", (cin, cout): "mvau_s"}.get((mw, mh))
            if key is None:
                raise RuntimeError(f"unexpected MVAU shape MW={mw} MH={mh} ({n.name})")
            roles[key] = n
        elif n.op_type.startswith("DuplicateStreams"):
            roles["dup"] = n
        elif n.op_type.startswith("AddStreams"):
            roles["add"] = n
        elif n.op_type.startswith("StreamingMaxPool"):
            roles["maxpool"] = n
        elif n.op_type.startswith("ConvolutionInputGenerator"):
            ks = list(inst.get_nodeattr("ConvKernelSize"))
            roles["swg_r" if ks[0] == 2 else "swg_m"] = n
        elif n.op_type.startswith("FMPadding"):
            pad = list(inst.get_nodeattr("Padding"))
            roles["fmpad" if all(int(x) == 1 for x in pad) else "fmpad_c"] = n
    need = ["mvau_r", "mvau_m", "mvau_e", "dup", "add", "maxpool", "swg_r", "swg_m", "fmpad"]
    need.append("mvau_s" if probe["skip_pad"] == "mvau" else "fmpad_c")
    for r in need:
        if r not in roles:
            raise RuntimeError(f"role {r} not found in graph (found {sorted(roles)})")
    cons = lambda role: _real_consumer(model, roles[role])
    roles["thr_r"], roles["thr_m"], roles["thr_e"], roles["thr_out"] = cons("mvau_r"), cons("mvau_m"), cons("mvau_e"), cons("add")
    roles["thr_s"] = cons("mvau_s" if probe["skip_pad"] == "mvau" else "fmpad_c")   # pad_thr order: threshold after the (channel) pad
    roles["thr_in"] = _real_producer(model, roles["dup"])
    for r, n in roles.items():
        if n is None:
            raise RuntimeError(f"role {r} unresolved")
    return roles


def identify_roles(model, probe: dict) -> dict:
    """role -> node for dup, mvau_r/m/e, thr_r/m/e/s/out/in, fmpad, swg, add (regular block) or the downsampling roles."""
    if probe.get("block") == "down":
        return identify_roles_down(model, probe)
    cin, cmid, cout, k = probe["cin"], probe["cmid"], probe["cout"], probe["k"]
    roles = {}
    for n in model.graph.node:
        inst = getCustomOp(n)
        if "MVAU" in n.op_type:
            mw, mh = inst.get_nodeattr("MW"), inst.get_nodeattr("MH")
            key = {(cin, cmid): "mvau_r", (k * k * cmid, cmid): "mvau_m", (cmid, cout): "mvau_e"}.get((mw, mh))
            if key is None:
                raise RuntimeError(f"unexpected MVAU shape MW={mw} MH={mh} ({n.name})")
            roles[key] = n
        elif n.op_type.startswith("DuplicateStreams"):
            roles["dup"] = n
        elif n.op_type.startswith("AddStreams"):
            roles["add"] = n
        elif n.op_type.startswith("ConvolutionInputGenerator"):
            roles["swg"] = n
        elif n.op_type.startswith("FMPadding"):
            roles["fmpad"] = n
    for r in ("mvau_r", "mvau_m", "mvau_e", "dup", "add", "swg", "fmpad"):
        if r not in roles:
            raise RuntimeError(f"role {r} not found in graph")
    thr_after = lambda role, name: roles.__setitem__(name, _real_consumer(model, roles[role]))
    thr_after("mvau_r", "thr_r")
    thr_after("mvau_m", "thr_m")
    thr_after("mvau_e", "thr_e")
    thr_after("add", "thr_out")
    roles["thr_in"] = _real_producer(model, roles["dup"])
    # skip threshold = the Dup output that does not lead to mvau_r
    for out in roles["dup"].output:
        c = model.find_consumer(out)
        while c is not None and _is_transparent(c):
            c = model.find_consumer(c.output[0])
        if c is not None and c.op_type.startswith("Thresholding") and c is not roles["thr_in"]:
            roles["thr_s"] = c
    for r, n in roles.items():
        if n is None:
            raise RuntimeError(f"role {r} unresolved")
    if "thr_s" not in roles:
        raise RuntimeError("skip threshold (Dup output -> Thresholding -> AddStreams) not found; merged skip_quant may differ")
    return roles


def role_name_of(node, roles_by_node: dict):
    if node is None:
        return None
    if node.op_type.startswith("StreamingDataWidthConverter"):
        return "dwc"
    return roles_by_node.get(node.name)


# ------------------------------------------------------------------ build steps specific to the probe
def step_apply_probe_folding(model, folding: dict, probe: dict):
    roles = identify_roles(model, probe)
    landed = {}
    for role, attrs in folding.items():
        if attrs.get("custom"):      # applied by the pass that creates the node (finn_channel_pad.py)
            continue
        node = roles[role]
        inst = getCustomOp(node)
        for key, val in attrs.items():
            inst.set_nodeattr(key, int(val))
        landed[role] = dict(node=node.name, op=node.op_type, **{k: inst.get_nodeattr(k) for k in attrs})
    # input stand-in threshold: match the Dup so no converter sits between them
    inst = getCustomOp(roles["thr_in"])
    inst.set_nodeattr("PE", int(folding["dup"]["PE"]))
    landed["thr_in"] = dict(node=roles["thr_in"].name, op=roles["thr_in"].op_type, PE=inst.get_nodeattr("PE"))
    # divisibility sanity (FINN would only complain at codegen)
    for role in ("mvau_r", "mvau_m", "mvau_e"):
        inst = getCustomOp(roles[role])
        assert inst.get_nodeattr("MH") % inst.get_nodeattr("PE") == 0, role
        assert inst.get_nodeattr("MW") % inst.get_nodeattr("SIMD") == 0, role
    print("[probe folding] applied:", json.dumps(landed))
    return model, roles, landed


def step_force_fifo_depths(model, roles: dict, predicted: list, policy: str, skip_scale: float, min_depth: int = 2):
    """After FINN's own FIFO sizing, overwrite depths of StreamingFIFO nodes with OUR depths, matched by the roles of the
    nodes at both ends. Returns (model, report). policy 'stock' keeps FINN's depths (control run)."""
    by_name = {n.name: r for r, n in roles.items()}
    wanted = {(f["producer"], f["consumer"]): f for f in predicted}
    report = []
    for n in model.graph.node:
        if not n.op_type.startswith("StreamingFIFO"):
            continue
        inst = getCustomOp(n)
        prod, cons = _real_producer(model, n), _real_consumer(model, n)
        key = (role_name_of(prod, by_name), role_name_of(cons, by_name))
        stock = inst.get_nodeattr("depth")
        entry = dict(fifo=n.name, edge=list(key), stock_depth=stock, forced_depth=None)
        f = wanted.get(key)
        if policy == "ours" and f is not None:
            depth = max(min_depth, int(round(f["depth"] * (skip_scale if f["is_skip"] else 1.0))))
            inst.set_nodeattr("depth", depth)
            entry["forced_depth"] = depth
            entry["is_skip"] = f["is_skip"]
        elif policy == "ours":
            entry["note"] = "edge not in prediction; stock depth kept"
        report.append(entry)
    print("[force fifo depths]", json.dumps(report))
    return model, report


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", help="e.g. bottleneck_cin32_d8_int4 (expects <case>.onnx, _probe.json, _folding.json in the enet dir)")
    ap.add_argument("--inputs-dir", default=ENET_DIR)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--no-merge", action="store_true", help="skip the residual_add/out_act threshold merge pass")
    ap.add_argument("--fifo-policy", choices=["ours", "stock"], default="ours")
    ap.add_argument("--skip-scale", type=float, default=1.0, help="scale the forced skip FIFO depth (0.5 = negative control)")
    ap.add_argument("--rtlsim-frames", type=int, default=3)
    ap.add_argument("--ooc", action="store_true", help="also run out-of-context synthesis after rtlsim")
    ap.add_argument("--stop-after", choices=["convert", "folding", "fifo", "stitch", "rtlsim"], default="rtlsim")
    a = ap.parse_args()

    model_file = os.path.join(a.inputs_dir, f"{a.case}.onnx")
    probe = json.load(open(os.path.join(a.inputs_dir, f"{a.case}_probe.json")))
    cfgj = json.load(open(os.path.join(a.inputs_dir, f"{a.case}_folding.json")))
    tag = ("merged" if not a.no_merge else "unmerged") + ("" if a.fifo_policy == "ours" else "_stockfifo") + (
        "" if a.skip_scale == 1.0 else f"_skip{a.skip_scale:g}")
    out_dir = a.output_dir or os.path.join(ENET_DIR, "finn_deployment_outputs", f"{a.case}_{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.join(out_dir, "report"), exist_ok=True)
    os.environ.setdefault("FINN_BUILD_DIR", os.path.join(out_dir, "finn_build_tmp"))
    os.makedirs(os.environ["FINN_BUILD_DIR"], exist_ok=True)
    print(f"case {a.case}  tag {tag}\nout  {out_dir}", flush=True)

    cfg = DataflowBuildConfig(
        output_dir=out_dir, mvau_wwidth_max=80, target_fps=None, synth_clk_period_ns=CLK_NS, fpga_part=FPGA_PART,
        auto_fifo_strategy=build_cfg.AutoFIFOSizingMethod("largefifo_rtlsim"), rtlsim_batch_size=a.rtlsim_frames,
        generate_outputs=[build_cfg.DataflowOutputType.STITCHED_IP, build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE],
        save_intermediate_models=True, steps=[],
    )
    result = dict(case=a.case, tag=tag, probe=probe, predicted=cfgj["predicted"], stages={})

    def save_result():
        with open(os.path.join(out_dir, "probe_result.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)

    m = ModelWrapper(model_file)
    for step in (
        step_qonnx_to_finn, step_enet_tidy, step_fuse_leaky_relu_to_threshold, step_enet_streamline,
        step_absorb_leftover_scale_before_matmul, step_fuse_forked_dequant_into_duplicate_threshold,
        step_dedup_forked_matmul_before_threshold, _fixup_degenerate_signed_bias,
    ):
        m = step(m, cfg)
    if not a.no_merge:
        m = step_compose_consecutive_thresholds(m, cfg)
    m = step_enet_convert_to_hw_rtl_mvau(m, cfg)
    if probe.get("block") == "down" and probe.get("skip_pad") == "fmpad":
        # channel zero-pad of the skip -> FMPadding on the regrouped stream (new pass, see finn_channel_pad.py)
        from finn_channel_pad import step_channel_pad_to_fmpadding
        m = step_channel_pad_to_fmpadding(m, cfgj["folding"]["fmpad_c"])
    m.save(os.path.join(out_dir, "after_convert_to_hw.onnx"))
    m = step_create_dataflow_partition(m, cfg)           # returns the child dataflow model
    m = step_specialize_layers(m, cfg)
    m = m.transform(GiveUniqueNodeNames())
    m = m.transform(GiveReadableTensorNames())
    result["stages"]["convert"] = dict(ops=sorted({n.op_type for n in m.graph.node}), n_nodes=len(m.graph.node))
    save_result()
    if a.stop_after == "convert":
        return

    m, roles, landed = step_apply_probe_folding(m, cfgj["folding"], probe)
    result["stages"]["folding"] = landed
    m = step_minimize_bit_width_standalone_thresh_aware(m, cfg)
    m = step_fix_weight_dtype_bipolar_bug(m, cfg)
    m = step_force_dsp(m, cfg)
    save_result()
    if a.stop_after == "folding":
        return

    m = step_hw_codegen(m, cfg)
    m = step_hw_ipgen(m, cfg)
    m = step_set_fifo_depths(m, cfg)                      # FINN's own sizing (largefifo_rtlsim)
    m.save(os.path.join(out_dir, "after_stock_fifo_sizing.onnx"))
    m, fifo_report = step_force_fifo_depths(m, roles, cfgj["fifos"], a.fifo_policy, a.skip_scale)
    result["stages"]["fifo"] = fifo_report
    m = m.transform(SplitLargeFIFOs())
    m = m.transform(GiveUniqueNodeNames())
    m = m.transform(PrepareIP(FPGA_PART, CLK_NS))
    m = m.transform(HLSSynthIP())
    m.save(os.path.join(out_dir, "after_forced_fifo.onnx"))
    save_result()
    if a.stop_after == "fifo":
        return

    m = m.transform(CreateStitchedIP(FPGA_PART, CLK_NS))
    final = os.path.join(out_dir, f"{a.case}_{tag}_stitched.onnx")
    m.save(final)
    save_result()
    if a.stop_after == "stitch":
        return

    # primary check: multi-frame rtlsim. A deadlock shows up as a timeout / no output; run_probes.sh wraps this in `timeout`.
    m = step_measure_rtlsim_performance(m, cfg)
    m.save(final)
    perf_file = os.path.join(out_dir, "report", "rtlsim_performance.json")
    if os.path.exists(perf_file):
        perf = json.load(open(perf_file))
        result["stages"]["rtlsim"] = perf
        px = probe["height"] * probe["width"]
        stable = perf.get("stable_throughput[images/s]") or perf.get("throughput[images/s]")
        fclk = perf.get("fclk[mhz]", 1000.0 / CLK_NS) * 1e6
        if stable:
            result["stages"]["rtlsim"]["steady_cyc_per_pixel"] = fclk / stable / px
            result["stages"]["rtlsim"]["target_cyc_per_pixel"] = cfgj["predicted"]["T"]
    save_result()

    if a.ooc:
        m = m.transform(SynthOutOfContext(part=FPGA_PART, clk_period_ns=CLK_NS))
        res = eval(m.get_metadata_prop("res_total_ooc_synth"))
        result["stages"]["ooc"] = res
        with open(os.path.join(out_dir, "report", "ooc_synth.json"), "w") as f:
            json.dump(res, f, indent=2)
        m.save(final)
        save_result()
    print("DONE", out_dir)


if __name__ == "__main__":
    main()

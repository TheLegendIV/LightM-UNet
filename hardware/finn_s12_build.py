"""Unified S12-dense FINN build: MILP folding bridge -> per-partition
specialize/fold/codegen/ipgen/FIFO-autosize/SplitLargeFIFOs/stitch -> real
Vivado OOC synth (+ rtlsim). URAM is NOT forced by default -- pass
--allocate-uram to opt in (see finn_s12_build_steps.step_allocate_uram_fifos).

Modes:
  --partitions all   full 8-way build; per-partition stitched IPs named after
                     their SDP node, then combine + estimate + rtlsim (multi).
  --partitions 2 [5 ...]  standalone partition build(s), FINN's stock
                     step_out_of_context_synthesis + step_measure_rtlsim_performance
                     per partition (hwsweep/ablation flow).
Without --folding-json FINN's own target_fps auto-fold is kept.

Run inside the FINN container:
    docker exec -e HOME=/tmp/home_dir <c> bash -c 'cd /home/thelegendiv/finn/notebooks/enet && \\
        nohup python3 finn_s12_build.py <preamble_dir> --tag <tag> --conv-order <conv_order.json> \\
            [--folding-json layer_bits_folding_<tag>.json] [--partitions all|N ...] \\
            [--target-fps F] [--mvau-wwidth-max N] [--allocate-uram [--uram-budget-blocks N]] \\
            > /tmp/ooc_<tag>.log 2>&1 &'
"""
import argparse
import concurrent.futures
import dataclasses
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("preamble_dir")
_parser.add_argument("--tag", required=True)
_parser.add_argument("--conv-order", help="<model>_conv_order.json (required with --folding-json)")
_parser.add_argument("--folding-json", help="MILP layer_bits_folding_*.json; omit for FINN auto-fold")
_parser.add_argument("--partitions", nargs="+", default=["all"])
_parser.add_argument("--target-fps", type=float, default=None)
_parser.add_argument("--mvau-wwidth-max", type=int, default=None)
_parser.add_argument("--allocate-uram", action="store_true",
                      help="greedily set ram_style=ultra on the deepest vivado-impl StreamingFIFO_rtl nodes "
                           "per partition, up to an 8-way-shared board budget (default OFF -- see "
                           "finn_s12_build_steps.step_allocate_uram_fifos)")
_parser.add_argument("--uram-budget-blocks", type=int, default=None,
                      help="whole-board URAM288 budget shared across all 8 partitions (default: "
                           "finn_s12_build_steps.URAM_BUDGET_BLOCKS, currently 88; ZCU7EV has 96 total)")
_parser.add_argument("--max-workers", type=int, default=4)
_parser.add_argument("--build-dir", default=None, help="FINN_BUILD_DIR base (default for 'all': finn_build_tmp/<tag>)")
_parser.add_argument("--output-dir", default=None, help="reuse an existing output dir instead of a new timestamped one")
_parser.add_argument("--bridge-only", action="store_true", help="write the bridged folding configs and stop")
_parser.add_argument("--fifo-autosize", choices=["fixed2", "rtlsim"], default="fixed2",
                      help="'fixed2' (default): skip FINN's own rtlsim-based FIFO autosizing entirely -- every "
                           "edge gets a real FIFO at depth 2, then step_force_fifo_depths_from_milp overwrites "
                           "matched ones (unmatched edges stay at 2, reported explicitly, no silent huge "
                           "autosized depths). 'rtlsim': old behaviour, FINN's own largefifo_rtlsim autosizer "
                           "runs first (slow, real Verilator cosim) and MILP forcing only overwrites matches.")
_args = _parser.parse_args()
if _args.folding_json and not _args.conv_order:
    _parser.error("--folding-json requires --conv-order")

# finn_enet_ip_build_partitioned_8way reads sys.argv[1]/[2] at import time.
_real_argv, sys.argv = sys.argv, sys.argv[:1]
import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
sys.argv = _real_argv

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402

import finn.builder.build_dataflow as build  # noqa: E402
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
    step_measure_rtlsim_performance,
    step_out_of_context_synthesis,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP  # noqa: E402
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP  # noqa: E402
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP  # noqa: E402
from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs  # noqa: E402
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext  # noqa: E402
from finn_partition_build_steps import (  # noqa: E402
    step_create_dataflow_partition_multi,
    step_combine_partitions,
    step_generate_estimate_reports_multi,
    step_measure_rtlsim_performance_multi,
)
from finn_stage_partition import validate_partition_single_output  # noqa: E402
from finn_s12_build_steps import (  # noqa: E402
    URAM_BUDGET_BLOCKS,
    URAM_PARTITIONS,
    build_partition_folding_config,
    install_relaxed_stage_boundaries,
    load_partition_logical_names,
    step_allocate_uram_fifos,
    step_fix_weight_dtype_bipolar_bug,
    step_force_dsp,
    step_force_fifo_depths_from_milp,
    step_minimize_bit_width_standalone_thresh_aware,
    step_set_fifo_depths_fixed2,
)

install_relaxed_stage_boundaries()


def _build_one_partition(fn, cfg, prefix, folding_file, idx, full, fold_suffix, tag, build_dir,
                          allocate_uram=False, uram_budget_blocks=None, fifo_plan=None, fifo_autosize="fixed2"):
    """full=True: 8-way flow (IP named <prefix>, SynthOutOfContext -> report/ooc_synth_partition_<i>.json).
    full=False: standalone flow (default "finn_design" IP name, required by stock rtlsim's hardcoded wrapper name).
    fifo_plan (this partition's bridged MILP inter_block_fifos/intra_block_fifos, see
    finn_s12_build_steps.build_partition_folding_config) is applied AFTER the FIFO-depth-assignment step
    (fixed2 default or rtlsim, see fifo_autosize) -- matched edges get the MILP/analytical-sim depth,
    unmatched edges keep whatever the depth-assignment step gave them (2 for fixed2, autosized for rtlsim)."""
    if build_dir:
        part_build_dir = os.path.join(build_dir, prefix.rstrip("_"))
        os.makedirs(part_build_dir, exist_ok=True)
        os.environ["FINN_BUILD_DIR"] = part_build_dir
    log = f"[partition {idx}]"
    if folding_file:
        cfg = dataclasses.replace(cfg, folding_config_file=folding_file)

    m = ModelWrapper(fn)
    m = step_specialize_layers(m, cfg)
    m = m.transform(GiveUniqueNodeNames() if full else GiveUniqueNodeNames(prefix))
    m = m.transform(GiveReadableTensorNames())
    m = step_target_fps_parallelization(m, cfg)
    if folding_file:
        m = step_apply_folding_config(m, cfg)
    else:
        print(f"{log} auto-fold: skipping step_apply_folding_config", flush=True)
    m = step_minimize_bit_width_standalone_thresh_aware(m, cfg)
    m = step_fix_weight_dtype_bipolar_bug(m, cfg)
    m = step_force_dsp(m, cfg)
    m = step_hw_codegen(m, cfg)
    m = step_hw_ipgen(m, cfg)
    m = step_set_fifo_depths(m, cfg) if fifo_autosize == "rtlsim" else step_set_fifo_depths_fixed2(m, cfg)
    if fifo_plan is not None:
        m, fifo_report = step_force_fifo_depths_from_milp(m, fifo_plan)
        report_dir = os.path.dirname(fn) if full else cfg.output_dir
        with open(os.path.join(report_dir, f"fifo_force_report_partition_{idx}.json"), "w") as f:
            json.dump(fifo_report, f, indent=2)
    if allocate_uram:
        # MUST run before SplitLargeFIFOs -- see step_allocate_uram_fifos's own docstring.
        budget = (uram_budget_blocks or URAM_BUDGET_BLOCKS) // URAM_PARTITIONS
        m = step_allocate_uram_fifos(m, idx, budget=budget)

    ckpt_dir = os.path.dirname(fn) if full else cfg.output_dir
    ckpt_name = f"partition_{idx}_prefifo_autosize.onnx" if full else f"partition{idx}_{tag}_{fold_suffix}_prefifo_autosize.onnx"
    m.save(os.path.join(ckpt_dir, ckpt_name))
    print(f"{log} saved FIFO-autosize checkpoint: {os.path.join(ckpt_dir, ckpt_name)}", flush=True)

    # Vivado axis_data_fifo caps at depth 32768; split after the checkpoint, not via cfg.split_large_fifos.
    m = m.transform(SplitLargeFIFOs())
    m = m.transform(GiveUniqueNodeNames(prefix))
    fpga_part, clk = cfg._resolve_fpga_part(), cfg.synth_clk_period_ns
    m = m.transform(PrepareIP(fpga_part, clk))
    m = m.transform(HLSSynthIP())

    if full:
        m = m.transform(CreateStitchedIP(fpga_part, clk, prefix.rstrip("_"), False))
        m.save(fn)
        print(f"{log} build done, starting OOC synth...", flush=True)
        m = m.transform(SynthOutOfContext(part=fpga_part, clk_period_ns=clk))
        res = eval(m.get_metadata_prop("res_total_ooc_synth"))
        m.save(fn)
        with open(os.path.join(cfg.output_dir, "report", f"ooc_synth_partition_{idx}.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"{log} OOC synth done: {res}", flush=True)
        return res

    m = m.transform(CreateStitchedIP(fpga_part, clk))
    final_fn = os.path.join(cfg.output_dir, f"partition{idx}_{tag}_{fold_suffix}_stitched.onnx")
    m.save(final_fn)
    m = step_out_of_context_synthesis(m, cfg)
    m.save(final_fn)
    try:
        m = step_measure_rtlsim_performance(m, cfg)
        m.save(final_fn)
    except Exception as e:
        print(f"{log} step_measure_rtlsim_performance FAILED, continuing without it: {e}")
    return None


def _aggregate_full(ooc_results, report_dir):
    all_results = {f"partition_{i}": ooc_results[i] for i in sorted(ooc_results)}
    numeric_keys = set()
    for res in all_results.values():
        for k, v in res.items():
            try:
                float(v)
                numeric_keys.add(k)
            except (TypeError, ValueError):
                pass
    aggregate = {}
    for k in numeric_keys:
        vals = [float(r[k]) for r in all_results.values() if k in r]
        if k.lower().startswith("fmax") or "period" in k.lower():
            aggregate[k + "_min_across_partitions"] = min(vals)
        else:
            aggregate[k + "_sum"] = sum(vals)
    all_results["aggregate"] = aggregate
    out = os.path.join(report_dir, "ooc_synth_and_timing_per_partition.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Combined OOC report:", out)


def main():
    full = _args.partitions == ["all"]
    part_ids = list(range(8)) if full else [int(p) for p in _args.partitions]
    fold_suffix = "milpfold" if _args.folding_json else "autofold"
    part_tag = "8way" if full else "partition" + "_".join(map(str, part_ids))

    output_dir = _args.output_dir or os.path.join(
        base.ENET_DIR, "finn_deployment_outputs",
        f"{_args.tag}_{fold_suffix}_{part_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    build_dir = _args.build_dir or (os.path.join(base.ENET_DIR, "finn_build_tmp", _args.tag) if full else None)
    if build_dir:
        os.makedirs(build_dir, exist_ok=True)
        os.environ["FINN_BUILD_DIR"] = build_dir
    os.makedirs(os.path.join(output_dir, "report"), exist_ok=True)

    overrides = {"output_dir": output_dir}
    if _args.target_fps is not None:
        overrides["target_fps"] = _args.target_fps
    if _args.mvau_wwidth_max is not None:
        overrides["mvau_wwidth_max"] = _args.mvau_wwidth_max
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, **overrides)

    print(f"Tag: {_args.tag}  partitions: {part_ids}  fold: {fold_suffix}")
    print(f"Preamble: {_args.preamble_dir}\nFolding json: {_args.folding_json}\nConv order: {_args.conv_order}")
    print(f"target_fps={cfg.target_fps} mvau_wwidth_max={cfg.mvau_wwidth_max}\nOUTPUT_DIR={output_dir}", flush=True)

    flat_ckpt = os.path.join(_args.preamble_dir, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
    parent = step_create_dataflow_partition_multi(ModelWrapper(flat_ckpt), cfg)
    sdp_nodes = parent.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 8, f"expected 8 partitions, got {len(sdp_nodes)}"
    if full:
        validate_partition_single_output(parent)

    folding_files = {i: None for i in part_ids}
    if _args.folding_json:
        with open(_args.folding_json) as f:
            folding_block = json.load(f)
        per_layer = folding_block["per_layer"]
        extra_nodes = folding_block.get("extra_nodes")
        if extra_nodes is None:
            print("WARNING: folding json has no extra_nodes -- join thresholds left on Vivado auto placement")
        inter_block_fifos = folding_block.get("inter_block_fifos")
        intra_block_fifos = folding_block.get("intra_block_fifos")
        if not inter_block_fifos and not intra_block_fifos:
            print("WARNING: folding json has no inter_block_fifos/intra_block_fifos -- all FIFOs left on FINN's autosized depth")
        logical = load_partition_logical_names(_args.preamble_dir, _args.conv_order)
        fifo_plans = {i: None for i in part_ids}
        for i in part_ids:
            fc, fifo_plan = build_partition_folding_config(
                getCustomOp(sdp_nodes[i]).get_nodeattr("model"), sdp_nodes[i].name, logical[i][0], per_layer,
                cfg, tag=f"p{i}", extra_nodes=extra_nodes, conv_order_file=_args.conv_order,
                inter_block_fifos=inter_block_fifos, intra_block_fifos=intra_block_fifos,
            )
            folding_files[i] = os.path.join(output_dir, f"hawq_folding_config_partition{i}.json")
            with open(folding_files[i], "w") as f:
                json.dump(fc, f, indent=2)
            print(f"[partition {i}] saved bridged folding config ({len(fc) - 1} entries): {folding_files[i]}")
            if fifo_plan is not None:
                fifo_plans[i] = fifo_plan
                fifo_plan_file = os.path.join(output_dir, f"fifo_plan_partition{i}.json")
                with open(fifo_plan_file, "w") as f:
                    json.dump(fifo_plan, f, indent=2)
                print(f"[partition {i}] saved FIFO plan ({len(fifo_plan['wanted'])} matched): {fifo_plan_file}")
    if _args.bridge_only:
        return

    jobs = []
    for i in part_ids:
        pcfg = cfg
        if not full and len(part_ids) > 1:
            pcfg = dataclasses.replace(cfg, output_dir=os.path.join(output_dir, f"partition_{i}"))
            os.makedirs(pcfg.output_dir, exist_ok=True)
        jobs.append((getCustomOp(sdp_nodes[i]).get_nodeattr("model"), pcfg, sdp_nodes[i].name + "_",
                     folding_files[i], i, full, fold_suffix, _args.tag, build_dir,
                     _args.allocate_uram, _args.uram_budget_blocks, fifo_plans.get(i) if _args.folding_json else None,
                     _args.fifo_autosize))

    results = {}
    if len(jobs) == 1:
        results[jobs[0][4]] = _build_one_partition(*jobs[0])
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=_args.max_workers) as ex:
            futs = {ex.submit(_build_one_partition, *j): j[4] for j in jobs}
            for fut in concurrent.futures.as_completed(futs):
                results[futs[fut]] = fut.result()
                print(f"partition {futs[fut]} done", flush=True)

    if not full:
        print(f"Done. Reports in {output_dir}")
        return

    _aggregate_full(results, os.path.join(output_dir, "report"))
    parent_ckpt = os.path.join(output_dir, "intermediate_models", "dataflow_parent_built.onnx")
    os.makedirs(os.path.dirname(parent_ckpt), exist_ok=True)
    parent.save(parent_ckpt)
    cfg = dataclasses.replace(cfg, steps=[
        step_combine_partitions, step_generate_estimate_reports_multi, step_measure_rtlsim_performance_multi,
    ])
    build.build_dataflow_cfg(parent_ckpt, cfg)


if __name__ == "__main__":
    main()

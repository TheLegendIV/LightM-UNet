"""Rebuild partition 1 AGAIN (picking up the dn-block role-mapping fix: 'reduce.0' is actually the
MILP's skip-branch 'mvau_s'/'thr_s', not 'mvau_r'/'thr_r' -- the real main-chain entry 'shortcut_proj'
has no per_layer entry and is now found structurally) through CreateStitchedIP only (no OOC synth),
then run it through the STANDARD (non-VCD) FINN zero-backpressure rtlsim testbench to confirm the
down1/down2 skip-branch FIFOs (now correctly forced to depth 1202/1258) no longer deadlock the
partition. Distinct BUILD_TAG/OUTPUT_DIR from refix1 -- runs alongside the original 3 live jobs
without touching their working directories.

Run:
    docker exec -e HOME=/tmp/home_dir -e FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp \\
        finn_persistent bash -c 'cd /home/thelegendiv/finn/notebooks/enet && \\
        nohup python3 -u rebuild_partition1_refix2_and_rtlsim.py > /tmp/partition1_refix2.log 2>&1 &'
"""
import dataclasses
import json
import os
import subprocess
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
os.environ.setdefault("FINN_ROOT", "/home/thelegendiv/finn")

BUILD_TAG = "S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_refix2"
ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
os.environ["FINN_BUILD_DIR"] = os.path.join(ENET_DIR, "finn_build_tmp", BUILD_TAG)
FINN_ROOT = "/home/thelegendiv/finn"
VERILATOR_ROOT = "/tmp/home_dir/.local/lib/python3.10/site-packages/verilator"
VERILATOR_BIN = VERILATOR_ROOT + "/bin/verilator"
VIVADO_PATH = "/tools/Xilinx/Vivado/2022.2"

PREAMBLE_DIR = os.path.join(
    ENET_DIR, "finn_deployment_outputs", "S12_dense_256_u4_analytical_v1_ft15ep_preamble_20261005_220558"
)
CONV_ORDER = os.path.join(ENET_DIR, "quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_conv_order.json")
FOLDING_JSON = os.path.join(
    ENET_DIR, "layer_bits_folding_S12_dense_256_u4_analytical_v1_int6_fps250_lat200_20261006.json"
)
PART_IDX = 1

OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"partition1_refix2_{BUILD_TAG}")
os.makedirs(os.path.join(OUTPUT_DIR, "report"), exist_ok=True)

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames  # noqa: E402
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_specialize_layers, step_target_fps_parallelization, step_apply_folding_config,
    step_hw_codegen, step_hw_ipgen,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP  # noqa: E402
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP  # noqa: E402
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP  # noqa: E402
from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs  # noqa: E402
from finn.util.pyverilator import prepare_stitched_ip_for_verilator  # noqa: E402
import numpy as np  # noqa: E402

import finn_enet_ip_build_partitioned_8way as base  # noqa: E402
from finn_partition_build_steps import step_create_dataflow_partition_multi  # noqa: E402
from finn_s12_build_steps import (  # noqa: E402
    build_partition_folding_config, install_relaxed_stage_boundaries, load_partition_logical_names,
    step_fix_weight_dtype_bipolar_bug, step_force_dsp, step_force_fifo_depths_from_milp,
    step_minimize_bit_width_standalone_thresh_aware, step_set_fifo_depths_fixed2,
)

install_relaxed_stage_boundaries()


def main():
    cfg = dataclasses.replace(base.cfg_stitched_ip_partitioned_8way, output_dir=OUTPUT_DIR)

    flat_ckpt = os.path.join(PREAMBLE_DIR, "intermediate_models", "assign_stage_partition_ids_8way.onnx")
    parent = step_create_dataflow_partition_multi(ModelWrapper(flat_ckpt), cfg)
    sdp_nodes = parent.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 8
    sdp = [n for n in sdp_nodes if n.name == f"GenericPartition_{PART_IDX}"][0]

    with open(FOLDING_JSON) as f:
        folding_block = json.load(f)
    per_layer = folding_block["per_layer"]
    extra_nodes = folding_block.get("extra_nodes")
    inter_block_fifos = folding_block.get("inter_block_fifos")
    intra_block_fifos = folding_block.get("intra_block_fifos")
    logical = load_partition_logical_names(PREAMBLE_DIR, CONV_ORDER)

    fc, fifo_plan = build_partition_folding_config(
        getCustomOp(sdp).get_nodeattr("model"), sdp.name, logical[PART_IDX][0], per_layer, cfg,
        tag=f"p{PART_IDX}", extra_nodes=extra_nodes, conv_order_file=CONV_ORDER,
        inter_block_fifos=inter_block_fifos, intra_block_fifos=intra_block_fifos,
    )
    folding_file = os.path.join(OUTPUT_DIR, f"hawq_folding_config_partition{PART_IDX}.json")
    with open(folding_file, "w") as f:
        json.dump(fc, f, indent=2)
    fifo_plan_file = os.path.join(OUTPUT_DIR, f"fifo_plan_partition{PART_IDX}.json")
    with open(fifo_plan_file, "w") as f:
        json.dump(fifo_plan, f, indent=2)
    print(f"[partition {PART_IDX}] matched {len(fifo_plan['wanted'])} FIFO roles", flush=True)

    cfg2 = dataclasses.replace(cfg, folding_config_file=folding_file)
    fn = getCustomOp(sdp).get_nodeattr("model")
    m = ModelWrapper(fn)
    m = step_specialize_layers(m, cfg2)
    m = m.transform(GiveUniqueNodeNames())
    m = m.transform(GiveReadableTensorNames())
    m = step_target_fps_parallelization(m, cfg2)
    m = step_apply_folding_config(m, cfg2)
    m = step_minimize_bit_width_standalone_thresh_aware(m, cfg2)
    m = step_fix_weight_dtype_bipolar_bug(m, cfg2)
    m = step_force_dsp(m, cfg2)
    m = m.transform(GiveUniqueNodeNames())
    m = step_hw_codegen(m, cfg2)
    m = step_hw_ipgen(m, cfg2)
    m = step_set_fifo_depths_fixed2(m, cfg2)
    m, fifo_report = step_force_fifo_depths_from_milp(m, fifo_plan)
    with open(os.path.join(OUTPUT_DIR, f"fifo_force_report_partition_{PART_IDX}.json"), "w") as f:
        json.dump(fifo_report, f, indent=2)
    for e in fifo_report:
        if e["fifo"] in ("StreamingFIFO_rtl_13", "StreamingFIFO_rtl_103"):
            print("CHECK:", json.dumps(e), flush=True)

    m = m.transform(SplitLargeFIFOs())
    m = m.transform(GiveUniqueNodeNames())
    fpga_part, clk = cfg2._resolve_fpga_part(), cfg2.synth_clk_period_ns
    m = m.transform(PrepareIP(fpga_part, clk))
    m = m.transform(HLSSynthIP())
    m = m.transform(CreateStitchedIP(fpga_part, clk))
    stitched_fn = os.path.join(OUTPUT_DIR, f"partition{PART_IDX}_refix2_stitched.onnx")
    m.save(stitched_fn)
    print(f"[partition {PART_IDX}] stitched IP saved: {stitched_fn}", flush=True)

    # ---- standard (non-debug) rtlsim, same recipe as run_single_partition_rtlsim.py ----
    kernel_model = m
    vivado_stitch_proj_dir = prepare_stitched_ip_for_verilator(kernel_model)
    wrapper_filename = kernel_model.get_metadata_prop("wrapper_filename")
    top_module = os.path.splitext(os.path.basename(wrapper_filename))[0]
    merged_src = os.path.join(vivado_stitch_proj_dir, top_module + ".v")
    assert os.path.isfile(merged_src), "merged source not found: %s" % merged_src

    first_node = kernel_model.find_consumer(kernel_model.graph.input[0].name)
    last_node = kernel_model.find_producer(kernel_model.graph.output[0].name)
    ishape_folded = getCustomOp(first_node).get_folded_input_shape()
    oshape_folded = getCustomOp(last_node).get_folded_output_shape()
    n_iters_in = int(np.prod(ishape_folded[:-1]))
    n_iters_out = int(np.prod(oshape_folded[:-1]))

    build_dir = os.path.realpath(os.path.join(vivado_stitch_proj_dir, "..", "rtlsim_single_refix2"))
    os.makedirs(build_dir, exist_ok=True)

    template_path = os.path.join(FINN_ROOT, "src/finn/qnn-data/cpp/verilator_fifosim.cpp")
    with open(template_path) as f:
        cpp = f.read()
    cpp = cpp.replace("@ITERS_PER_INPUT@", str(n_iters_in))
    cpp = cpp.replace("@ITERS_PER_OUTPUT@", str(n_iters_out))
    cpp = cpp.replace("@N_INPUTS@", "1")
    cpp = cpp.replace("@MAX_ITERS@", str(5000000))
    cpp = cpp.replace("@FIFO_DEPTH_LOGGING@", "")
    cpp = cpp.replace("finn_design_wrapper", top_module)

    cpp_fname = "verilator_fifosim_%s.cpp" % top_module
    with open(os.path.join(build_dir, cpp_fname), "w") as f:
        f.write(cpp)

    verilog_header_dir = os.path.join(vivado_stitch_proj_dir, "pyverilator_vh")
    xpm_memory = os.path.join(VIVADO_PATH, "data/ip/xpm/xpm_memory/hdl/xpm_memory.sv")
    xpm_cdc = os.path.join(VIVADO_PATH, "data/ip/xpm/xpm_cdc/hdl/xpm_cdc.sv")
    xpm_fifo = os.path.join(VIVADO_PATH, "data/ip/xpm/xpm_fifo/hdl/xpm_fifo.sv")
    swg_pkg = os.path.join(FINN_ROOT, "finn-rtllib/swg/swg_pkg.sv")

    verilator_args = [
        "perl", VERILATOR_BIN,
        "-Wno-fatal", "-Mdir", build_dir,
        "-y", vivado_stitch_proj_dir,
        "-y", verilog_header_dir,
        "--CFLAGS", "--std=c++17",
        "-O3", "--x-assign", "fast", "--x-initial", "fast", "--noassert",
        "--cc",
        swg_pkg, merged_src, xpm_memory, xpm_cdc, xpm_fifo,
        "--top-module", top_module,
        "--exe", cpp_fname,
        "--no-timing",
        "-DDISABLE_XPM_ASSERTIONS", "-DOBSOLETE", "-DONESPIN", "--bbox-unsup",
    ]
    log_path = os.path.join(build_dir, "verilate.log")
    with open(log_path, "w") as logf:
        ret = subprocess.run(verilator_args, cwd=build_dir, stdout=logf, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        print("VERILATE FAILED, see", log_path, flush=True)
        sys.exit(1)

    mk_file = "V%s.mk" % top_module
    make_log = os.path.join(build_dir, "make.log")
    env = os.environ.copy()
    env["CFG_CXXFLAGS_PCH_I"] = "-include"
    with open(make_log, "w") as logf:
        ret = subprocess.run(
            ["make", "-j4", "-f", mk_file, "V%s" % top_module, "CFG_CXXFLAGS_PCH_I=-include"],
            cwd=build_dir, stdout=logf, stderr=subprocess.STDOUT, env=env,
        )
    if ret.returncode != 0:
        print("MAKE FAILED, see", make_log, flush=True)
        sys.exit(1)

    binary = os.path.join(build_dir, "V%s" % top_module)
    run_log = os.path.join(build_dir, "run.log")
    try:
        with open(run_log, "w") as logf:
            ret = subprocess.run([binary], cwd=build_dir, stdout=logf, stderr=subprocess.STDOUT, timeout=1800)
        run_returncode = ret.returncode
    except subprocess.TimeoutExpired:
        run_returncode = "TIMEOUT"

    results_path = os.path.join(build_dir, "results.txt")
    results = {}
    if os.path.isfile(results_path):
        with open(results_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    results[parts[0]] = parts[1]
    print(
        f"PART {PART_IDX} REFIX2 RESULT: top_module={top_module} run_returncode={run_returncode} results={results}",
        flush=True,
    )


if __name__ == "__main__":
    main()

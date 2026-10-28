"""Reuse the ALREADY-STITCHED partition 6 "refix1" IP (the one that deadlocked in rtlsim:
N_IN_TXNS=3163/32768, N_OUT_TXNS=200/32768, hit the 5,000,000-cycle no-output timeout) and
re-run it through the stock verilator_fifosim.cpp testbench, but compiled with VCD tracing
enabled (-DDEBUG + --trace) and a MUCH shorter no-output timeout (so the sim exits and
flushes trace.vcd soon after it stalls, instead of running 5M more cycles pointlessly).

Does NOT re-run CreateStitchedIP / Vivado -- loads the existing
partition6_refix_stitched.onnx checkpoint directly, so this is fast (just
verilator compile + a short sim run).

Run:
    docker exec -e HOME=/tmp/home_dir finn_persistent bash -c 'cd /home/thelegendiv/finn/notebooks/enet && \\
        nohup python3 -u vcd_trace_partition6_refix1.py > /tmp/partition6_vcd.log 2>&1 &'
"""
import os
import subprocess
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
os.environ.setdefault("FINN_ROOT", "/home/thelegendiv/finn")

BUILD_TAG = "S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_refix1"
ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
os.environ["FINN_BUILD_DIR"] = os.path.join(ENET_DIR, "finn_build_tmp", BUILD_TAG)
FINN_ROOT = "/home/thelegendiv/finn"
VERILATOR_ROOT = "/tmp/home_dir/.local/lib/python3.10/site-packages/verilator"
VERILATOR_BIN = VERILATOR_ROOT + "/bin/verilator"
VIVADO_PATH = "/tools/Xilinx/Vivado/2022.2"
PART_IDX = 6

STITCHED_ONNX = os.path.join(
    ENET_DIR, "finn_deployment_outputs", f"partition6_refix_{BUILD_TAG}", "partition6_refix_stitched.onnx"
)
# how many cycles with NO new output before giving up and dumping the VCD -- short on purpose
NO_OUTPUT_TIMEOUT = 20000

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from finn.util.pyverilator import prepare_stitched_ip_for_verilator  # noqa: E402
import numpy as np  # noqa: E402


def main():
    assert os.path.isfile(STITCHED_ONNX), f"missing checkpoint: {STITCHED_ONNX}"
    kernel_model = ModelWrapper(STITCHED_ONNX)

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
    print(f"[partition {PART_IDX}] n_iters_in={n_iters_in} n_iters_out={n_iters_out}", flush=True)

    build_dir = os.path.realpath(os.path.join(vivado_stitch_proj_dir, "..", "rtlsim_single_refix_vcd"))
    os.makedirs(build_dir, exist_ok=True)

    template_path = os.path.join(FINN_ROOT, "src/finn/qnn-data/cpp/verilator_fifosim.cpp")
    with open(template_path) as f:
        cpp = f.read()
    cpp = cpp.replace("@ITERS_PER_INPUT@", str(n_iters_in))
    cpp = cpp.replace("@ITERS_PER_OUTPUT@", str(n_iters_out))
    cpp = cpp.replace("@N_INPUTS@", "1")
    cpp = cpp.replace("@MAX_ITERS@", str(NO_OUTPUT_TIMEOUT))
    cpp = cpp.replace("@FIFO_DEPTH_LOGGING@", "")
    cpp = cpp.replace("finn_design_wrapper", top_module)

    cpp_fname = "verilator_fifosim_vcd_%s.cpp" % top_module
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
        "--CFLAGS", "--std=c++17 -DDEBUG",
        "--trace",
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

    vcd_path = os.path.join(build_dir, "trace.vcd")
    vcd_size = os.path.getsize(vcd_path) if os.path.isfile(vcd_path) else -1
    print(
        f"PART {PART_IDX} VCD RESULT: top_module={top_module} run_returncode={run_returncode} "
        f"results={results} vcd_path={vcd_path} vcd_size_bytes={vcd_size}",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Finish the paced-source FIFO probe builds: SplitLargeFIFOs -> stitch -> OOC synth -> per-partition rtlsim, from the
partition_<i>_postfifo_autosize.onnx checkpoints written by fifo_paced_probe.py (the probe stops right after that checkpoint).

Mirrors the tail of finn_s12_build._build_one_partition (full=True), so the numbers are comparable to the saturated build's
ooc_synth_partition_<i>.json. Partition 7 can fail in placement (UTLZ-1, too many RAMB36): then the post-synthesis utilization is
dumped from the opt checkpoint instead (synth_only=true), like finish_p7.py did for the saturated build.

Usage (container): python3 -u fifo_paced_finish.py <preamble_dir> <probe tag> <out dir> <ids...> [--max-workers N]
    <out dir> receives report/ooc_synth_partition_<i>.json, report/rtlsim_partition_<i>.json, intermediate_models/partition_<i>.onnx
"""
import concurrent.futures
import dataclasses
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
ARGS = sys.argv[1:]
MAXW = int(ARGS.pop(ARGS.index("--max-workers") + 1)) if "--max-workers" in ARGS else 4
if "--max-workers" in ARGS:
    ARGS.remove("--max-workers")
PREAMBLE, TAG, OUT, IDS = ARGS[0], ARGS[1], ARGS[2], [int(x) for x in ARGS[3:]]
sys.argv = [sys.argv[0], PREAMBLE, "--tag", TAG]  # finn_s12_build parses argv at import

ENET = "/home/thelegendiv/finn/notebooks/enet"


def finish(i):
    import finn_s12_build as fsb
    from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
    from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
    from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
    from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs
    from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext
    from finn_partition_build_steps import measure_partition_rtlsim
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.transformation.general import GiveUniqueNodeNames

    prefix = f"GenericPartition_{i}_"
    part_dir = f"{ENET}/finn_build_tmp/{TAG}/GenericPartition_{i}"
    os.environ["FINN_BUILD_DIR"] = part_dir
    cfg = dataclasses.replace(fsb.base.cfg_stitched_ip_partitioned_8way, output_dir=OUT, target_fps=150.0)
    ckpt = sorted(glob.glob(f"{ENET}/finn_deployment_outputs/{TAG}_autofold_probe*/intermediate_models/supported_op_partitions/"
                            f"partition_{i}_postfifo_autosize.onnx"))[-1]
    os.makedirs(f"{OUT}/report", exist_ok=True)
    os.makedirs(f"{OUT}/intermediate_models", exist_ok=True)
    fn = f"{OUT}/intermediate_models/partition_{i}.onnx"
    log = f"[partition {i}]"
    print(f"{log} resuming from {ckpt}", flush=True)

    m = ModelWrapper(ckpt)
    m = m.transform(SplitLargeFIFOs())
    m = m.transform(GiveUniqueNodeNames(prefix))
    fpga_part, clk = cfg._resolve_fpga_part(), cfg.synth_clk_period_ns
    m = m.transform(PrepareIP(fpga_part, clk))
    m = m.transform(HLSSynthIP())
    m = m.transform(CreateStitchedIP(fpga_part, clk, prefix.rstrip("_"), False))
    m.save(fn)
    print(f"{log} stitched, starting OOC synth...", flush=True)

    try:
        res = eval((m.transform(SynthOutOfContext(part=fpga_part, clk_period_ns=clk))).get_metadata_prop("res_total_ooc_synth"))
    except Exception as e:  # placement failure (UTLZ-1): fall back to post-synthesis utilization
        print(f"{log} OOC synth failed ({str(e)[-300:]}), dumping post-synthesis utilization", flush=True)
        dcp = sorted(glob.glob(f"{part_dir}/synth_out_of_context_*/results_*/vivadocompile/vivadocompile.runs/impl_1/*_opt.dcp"))[-1]
        res_txt = f"{OUT}/report/p{i}_synth_only.txt"
        tcl = f"{ENET}/util_from_dcp.tcl"
        subprocess.run(["bash", "-c", f"source /tools/Xilinx/Vivado/2022.2/settings64.sh >/dev/null 2>&1; "
                        f"vivado -mode batch -source {tcl} -tclargs {dcp} {res_txt}"], check=True)
        res = {}
        for line in open(res_txt):
            k, v = line.strip().split("=", 1)
            res[k] = float(v)
        res.update({"synth_only": True, "vivado_proj_folder": os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(dcp)))),
                    "note": "post-synthesis/opt utilization; placement failed: no WNS / Fmax"})
    with open(f"{OUT}/report/ooc_synth_partition_{i}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"{log} OOC synth done: {res}", flush=True)

    try:
        rtl = measure_partition_rtlsim(m)
        print(f"{log} rtlsim {'DEADLOCK' if rtl['deadlock'] else 'PASS'}: {rtl}", flush=True)
    except Exception as e:
        rtl = {"error": str(e)[-2000:], "deadlock": None}
        print(f"{log} rtlsim FAILED to run, continuing: {e}", flush=True)
    with open(f"{OUT}/report/rtlsim_partition_{i}.json", "w") as f:
        json.dump(rtl, f, indent=2)
    return i


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAXW) as ex:
        for fut in concurrent.futures.as_completed([ex.submit(finish, i) for i in IDS]):
            print(f"partition {fut.result()} finished", flush=True)

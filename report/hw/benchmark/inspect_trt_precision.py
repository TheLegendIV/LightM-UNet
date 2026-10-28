"""Which layers of the TensorRT engine of benchmark_int8_cpu_gpu.py are NOT int8?

Builds (or loads the cached) engine through TrtGpu, reads the per-layer description with the engine inspector and prints, per layer type, how many layers have
int8 / half / float inputs and outputs, then lists every layer that has no int8 tensor on its input or output side. Needs the engine built with
ProfilingVerbosity.DETAILED (set in TrtGpu); delete int8/*.plan if it was built before that.

Usage (lightmunet_dev):  bash report/MILP/benchmark/run_trt_int8.sh is the benchmark; this one:
    PYTHONPATH=/opt/trt_pkgs LD_LIBRARY_PATH=<as in run_trt_int8.sh> python3 report/MILP/benchmark/inspect_trt_precision.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "deployment"))
from benchmark_int8_cpu_gpu import TrtGpu  # noqa: E402


def dtype_of(t: dict) -> str:
    fmt = t.get("Format/Datatype", "")
    for tag, name in (("Int8", "int8"), ("Half", "half"), ("Float", "float"), ("Int32", "int32"), ("Int64", "int64"), ("Bool", "bool")):
        if tag.lower() in fmt.lower():
            return name
    return fmt or "?"


def main() -> None:
    be = TrtGpu(threads=1)
    insp = be.engine.create_engine_inspector()
    info = json.loads(insp.get_engine_information(be.trt.LayerInformationFormat.JSON))
    layers = info["Layers"]
    print(f"{len(layers)} layers in the engine")
    by_type = defaultdict(Counter)
    off = []
    for ly in layers:
        if isinstance(ly, str):          # a layer folded away has only its name
            by_type["(name only)"]["?"] += 1
            continue
        ins, outs = ly.get("Inputs", []), ly.get("Outputs", [])
        din, dout = {dtype_of(t) for t in ins}, {dtype_of(t) for t in outs}
        kind = ly.get("LayerType", "?")
        by_type[kind][f"in={'/'.join(sorted(din)) or '-'} out={'/'.join(sorted(dout)) or '-'}"] += 1
        if "int8" not in din | dout:
            off.append((ly.get("Name", "?"), kind, sorted(din), sorted(dout), ly.get("TacticName", "")))
    print("\nlayer type -> dtype pattern : count")
    for kind in sorted(by_type):
        for pat, n in by_type[kind].most_common():
            print(f"  {kind:<22} {pat:<36} {n}")
    print(f"\n{len(off)} layers with NO int8 input or output:")
    for name, kind, din, dout, tac in off:
        print(f"  {kind:<20} in={din} out={dout}  {name}  [{tac}]")
    # a stricter view: layers that touch a non-int8 tensor of a convolution-sized type
    print("\nConvolution-type layers that are not int8 in AND out:")
    for ly in layers:
        if isinstance(ly, dict) and ly.get("LayerType") in ("Convolution", "CaskConvolution", "Deconvolution", "CaskDeconvolution"):
            d = {dtype_of(t) for t in ly.get("Inputs", [])} | {dtype_of(t) for t in ly.get("Outputs", [])}
            if d != {"int8"}:
                print(f"  {ly.get('Name')}  {sorted(d)}  [{ly.get('TacticName', '')}]")


if __name__ == "__main__":
    main()

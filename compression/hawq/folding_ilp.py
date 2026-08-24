"""Per-LAYER folding ILP for nnUNetTrainerENet_23_1_s19_warmstart_4c at
FIXED W8A8 (uniform 8-bit weights and activations everywhere -- folding is
a completely separate design axis from the per-stage bit-width search in
ilp_search.py, and this script answers a different question: "given a
fixed bit-width, what PE/SIMD (folding) per layer minimizes latency while
fitting the ZCU7's real LUT/BRAM budget").

Folding recap (see finn_cost_model.py's own docstring for the full
derivation): PE = output channels computed in parallel per cycle, SIMD =
reduction elements (C_in*K_h*K_w) processed in parallel per PE per cycle.
FOLDING_UNFOLDED (PE=C_out, SIMD=C_in*K_h*K_w) = one cycle per output
pixel, maximum resource, minimum latency. FOLDING_SERIAL (PE=SIMD=1) = one
MAC per cycle, minimum resource, maximum latency. This ILP searches the
FULL space between those two extremes, per layer independently (each
layer's own C_out/C_in*K_h*K_w have their own valid divisors -- FINN's real
folding constraint: PE must evenly divide C_out, SIMD must evenly divide
C_in*K_h*K_w, no ragged folding).

Variables: one binary x[layer, pe, simd] per (layer, valid PE, valid SIMD)
triple, exactly one chosen per layer (sum == 1).
Objective: minimize sum_layer cycles(layer, pe, simd) -- i.e. total
per-image LATENCY (summing every layer's own cycle count, as if the
image passes through the pipeline once, no cross-layer overlap credited).
This is NOT the same as maximizing STEADY-STATE THROUGHPUT (which is
governed by max_layer cycles(layer, pe, simd), the single slowest/
bottleneck layer in a pipelined dataflow design, once many images are
in flight) -- sum is used here because it's directly linear/MIP-friendly
and answers "how long does one image take", the more relevant question
for single-image medical segmentation inference. A throughput-oriented
(minimize max) variant would need one extra continuous variable T with
T >= cycles(layer) for every layer -- straightforward to add later if
wanted, not built here.
Constraints (HARD, unlike ilp_search.py's BRAM-as-penalty choice -- here
BOTH resources get a real shot at being satisfiable since folding, unlike
bit-width alone, has enough range to plausibly reach budget, per the
compression/hawq/finn_stage_costs_serial.json finding that serial+2bit
alone gets BRAM down to 728/624 with W2A2 -- W8A8 will need more folding
headroom to compensate, this ILP finds out whether that's actually
achievable):
    sum_layer LUT(layer, pe, simd) <= 230400  (XCZU7EV LUT budget)
    sum_layer BRAM18K(layer, pe, simd) <= 624  (XCZU7EV BRAM_18K budget)

Usage:
    python compression/hawq/folding_ilp.py --config config_23_1 --out-file compression/hawq/folding_23_1_w8a8.json
    python compression/hawq/folding_ilp.py --config config_21_2 --cost-backend native
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pulp
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finn_cost_model import LayerGeometry, divisors, layer_cost_pe_simd, max_pe, max_simd  # noqa: E402
import finn_stage_costs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402

XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624, "URAM": 96}
WEIGHT_BITS = 8
ACT_BITS = 8
RAM_STYLES = ("block", "ultra")  # BRAM vs URAM weight-memory placement, see finn_cost_model.py's RamStyle

# Container-side paths for the native cost backend (see
# hardware/finn_native_cost_estimator.py's own docstring for the full
# design rationale/scope notes).
NATIVE_ESTIMATOR_LOCAL = REPO_ROOT / "hardware" / "finn_native_cost_estimator.py"
NATIVE_COST_MODEL_LOCAL = Path(__file__).resolve().parent / "finn_cost_model.py"
NATIVE_REMOTE_DIR = "/home/thelegendiv/finn/notebooks/enet"


def candidate_folds(layer: LayerGeometry) -> list[tuple[int, int, str]]:
    """Every valid (PE, SIMD, ram_style) triple for this layer -- MaxPool2d
    has neither PE/SIMD nor a weight memory at all (no MVAU, no weights),
    so it gets the single sentinel (1, 1, "block"), which
    layer_cost_pe_simd ignores for that op_type anyway (its cost/cycles
    don't depend on pe/simd/ram_style at all -- see maxpool_cost). Conv2d/
    ConvTranspose2d get BOTH ram_style options per (PE, SIMD) -- the ILP
    picks whichever (block=BRAM vs ultra=URAM) fits/minimizes cycles per
    layer, since real FINN's LUT/cycle cost is identical either way (only
    the BRAM_18K vs URAM resource ledger differs, see finn_cost_model.py's
    conv_cost_pe_simd docstring)."""
    if layer.op_type == "MaxPool2d":
        return [(1, 1, "block")]
    return [
        (pe, simd, ram_style)
        for pe in divisors(max_pe(layer)) for simd in divisors(max_simd(layer)) for ram_style in RAM_STYLES
    ]


def build_cost_table_analytical(geometries: list[LayerGeometry]) -> dict[str, dict[tuple[int, int, str], dict]]:
    """Default/original cost backend -- finn_cost_model.py's closed-form
    formulae, no Docker/FINN needed. Preserves the exact behavior this
    script always had (backward-compat default), now keyed by (pe, simd,
    ram_style) triples instead of (pe, simd) pairs."""
    layer_costs: dict[str, dict[tuple[int, int, str], dict]] = {}
    for layer in geometries:
        folds = candidate_folds(layer)
        layer_costs[layer.name] = {
            (pe, simd, ram_style): layer_cost_pe_simd(layer, WEIGHT_BITS, ACT_BITS, pe, simd, ram_style)
            for pe, simd, ram_style in folds
        }
    return layer_costs


def build_cost_table_native(
    geometries: list[LayerGeometry], container: str = "brave_lewin", remote_dir: str = NATIVE_REMOTE_DIR,
) -> dict[str, dict[tuple[int, int, str], dict]]:
    """Real-FINN cost backend -- serializes every (layer, PE, SIMD,
    ram_style) candidate to a request JSON, docker cp's it plus
    hardware/finn_native_cost_estimator.py + finn_cost_model.py into the
    FINN container, docker execs the estimator there (calls FINN's own
    getCustomOp(MVAU_hls) resource-estimation methods -- see that file's
    docstring), and docker cp's the response back. Output shape is
    IDENTICAL to build_cost_table_analytical's (same {layer_name: {(pe,
    simd, ram_style): cost_dict}} nesting, same required keys) so
    solve_folding() doesn't need to know or care which backend produced it."""
    candidates = {layer.name: [list(c) for c in candidate_folds(layer)] for layer in geometries}
    request = {
        "weight_bits": WEIGHT_BITS, "act_bits": ACT_BITS, "fpga_part": "xczu7ev-ffvc1156-2-e",
        "layers": [dataclasses.asdict(layer) for layer in geometries],
        "candidates": candidates,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        request_path = Path(tmpdir) / "native_request.json"
        response_path = Path(tmpdir) / "native_response.json"
        with open(request_path, "w") as f:
            json.dump(request, f)

        def docker_cp(src: Path, dst: str) -> None:
            subprocess.run(["docker", "cp", str(src), dst], check=True)

        docker_cp(NATIVE_ESTIMATOR_LOCAL, f"{container}:{remote_dir}/finn_native_cost_estimator.py")
        docker_cp(NATIVE_COST_MODEL_LOCAL, f"{container}:{remote_dir}/finn_cost_model.py")
        docker_cp(request_path, f"{container}:{remote_dir}/native_request.json")

        subprocess.run(
            [
                "docker", "exec", container, "bash", "-c",
                f"export HOME=/tmp/home_dir; cd {remote_dir}; "
                f"python3 finn_native_cost_estimator.py native_request.json native_response.json",
            ],
            check=True,
        )
        subprocess.run(["docker", "cp", f"{container}:{remote_dir}/native_response.json", str(response_path)], check=True)

        with open(response_path) as f:
            raw_response = json.load(f)

    layer_costs: dict[str, dict[tuple[int, int, str], dict]] = {}
    for layer_name, pe_simd_costs in raw_response.items():
        layer_costs[layer_name] = {}
        for key, cost in pe_simd_costs.items():
            pe_str, simd_str, ram_style = key.split(",")
            layer_costs[layer_name][(int(pe_str), int(simd_str), ram_style)] = cost
    return layer_costs


def solve_folding(geometries: list[LayerGeometry], layer_costs: dict[str, dict[tuple[int, int, str], dict]]) -> dict:
    prob = pulp.LpProblem("HAWQ_folding_w8a8", pulp.LpMinimize)
    x = {}
    for layer in geometries:
        folds = candidate_folds(layer)
        for pe, simd, ram_style in folds:
            x[(layer.name, pe, simd, ram_style)] = pulp.LpVariable(f"x_{layer.name}_{pe}_{simd}_{ram_style}", cat=pulp.LpBinary)
        prob += pulp.lpSum(x[(layer.name, pe, simd, ram_style)] for pe, simd, ram_style in folds) == 1, f"one_fold_{layer.name}"

    prob += pulp.lpSum(
        x[(layer.name, pe, simd, ram_style)] * cost["cycles"]
        for layer in geometries for (pe, simd, ram_style), cost in layer_costs[layer.name].items()
    )
    prob += pulp.lpSum(
        x[(layer.name, pe, simd, ram_style)] * cost["total_lut"]
        for layer in geometries for (pe, simd, ram_style), cost in layer_costs[layer.name].items()
    ) <= XCZU7EV["LUT"], "lut_budget"
    prob += pulp.lpSum(
        x[(layer.name, pe, simd, ram_style)] * (cost["swu_bram18"] + cost["wm_bram18"])
        for layer in geometries for (pe, simd, ram_style), cost in layer_costs[layer.name].items()
    ) <= XCZU7EV["BRAM_18K"], "bram_budget"
    prob += pulp.lpSum(
        x[(layer.name, pe, simd, ram_style)] * cost.get("wm_uram18", 0)
        for layer in geometries for (pe, simd, ram_style), cost in layer_costs[layer.name].items()
    ) <= XCZU7EV["URAM"], "uram_budget"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status_name = pulp.LpStatus[status]
    result_per_layer = {}
    if status_name == "Optimal":
        for layer in geometries:
            for pe, simd, ram_style in candidate_folds(layer):
                if pulp.value(x[(layer.name, pe, simd, ram_style)]) > 0.5:
                    result_per_layer[layer.name] = {
                        "stage": layer.stage, "pe": pe, "simd": simd, "ram_style": ram_style,
                        **layer_costs[layer.name][(pe, simd, ram_style)],
                    }
                    break
    return {"status": status_name, "per_layer": result_per_layer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_23_1", help="Architecture config module under compression/hawq/ (e.g. config_23_1, config_21_2).")
    parser.add_argument("--out-file", type=Path, default=None)
    parser.add_argument(
        "--cost-backend", choices=["analytical", "native"], default="analytical",
        help="'analytical' (default, unchanged behavior) = finn_cost_model.py's closed-form formulae, "
             "no Docker needed. 'native' = real FINN getCustomOp(MVAU_hls) resource estimation for the "
             "MVU term (see hardware/finn_native_cost_estimator.py), SWU/MaxPool2d still analytical -- "
             "requires a running FINN container (see --container).",
    )
    parser.add_argument("--container", default="brave_lewin", help="FINN Docker container name (--cost-backend native only).")
    args = parser.parse_args()
    if args.out_file is None:
        suffix = args.config.removeprefix("config_")
        args.out_file = Path(f"compression/hawq/folding_{suffix}_w8a8.json")

    if args.config != "config_23_1":
        finn_stage_costs.load_config(args.config)
    cfg = importlib.import_module(args.config)
    model = ENet(
        in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS, channels=cfg.CHANNELS,
        bottlenecks_per_stage=cfg.BOTTLENECKS_PER_STAGE, decoder_type=cfg.DECODER_TYPE,
        use_asymmetric=cfg.USE_ASYMMETRIC, context_pattern=cfg.CONTEXT_PATTERN,
        separable_dilated=cfg.SEPARABLE_DILATED, use_prelu=True, prelu_variant=cfg.PRELU_VARIANT,
    )
    geometries = finn_stage_costs.dump_layer_geometry(model, finn_stage_costs.INPUT_HW)
    n_candidates = sum(len(candidate_folds(g)) for g in geometries)
    print(f"Traced {len(geometries)} layers, {n_candidates} total (layer, PE, SIMD) candidates. "
          f"Building cost table ({args.cost_backend} backend)...")

    if args.cost_backend == "native":
        layer_costs = build_cost_table_native(geometries, container=args.container)
    else:
        layer_costs = build_cost_table_analytical(geometries)

    print("Solving ILP...")
    result = solve_folding(geometries, layer_costs)
    print(f"ILP status: {result['status']}")
    if result["status"] != "Optimal":
        print("INFEASIBLE at W8A8 within LUT+BRAM+URAM budget -- no folding configuration fits. "
              "See compression/hawq/finn_stage_costs_serial.json for the theoretical floor at W2A2 "
              "(728/624 BRAM, still over) -- W8A8 needs even more headroom, may simply be infeasible.")
    else:
        total_lut = sum(v["total_lut"] for v in result["per_layer"].values())
        total_bram = sum(v["swu_bram18"] + v["wm_bram18"] for v in result["per_layer"].values())
        total_uram = sum(v.get("wm_uram18", 0) for v in result["per_layer"].values())
        total_cycles = sum(v["cycles"] for v in result["per_layer"].values())
        print(f"Total LUT: {total_lut:.0f} ({100*total_lut/XCZU7EV['LUT']:.1f}% of budget)")
        print(f"Total BRAM_18K: {total_bram:.0f} ({100*total_bram/XCZU7EV['BRAM_18K']:.1f}% of budget)")
        print(f"Total URAM: {total_uram:.0f} ({100*total_uram/XCZU7EV['URAM']:.1f}% of budget)")
        print(f"Total cycles (sum, ~= per-image latency): {total_cycles:.0f}")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out_file}")


if __name__ == "__main__":
    main()

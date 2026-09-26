"""Computes the REAL end-to-end latency of a SOLVED finn_milp.py folding
result: the longest path through the actual dataflow DAG (critical path),
not the naive flat sum finn_milp.py's own `total_cycles`/`latency_ms`
diagnostics report.

WHY THIS EXISTS: finn_milp.py's `total_cycles = sum(v["cycles"] for v in
per_layer.values())` adds up EVERY layer's own cycles, with zero regard for
topology. That's the right answer for a purely serial chain, but real FINN
dataflow hardware runs PARALLEL branches (any fork/join -- InitialBlock's
own conv/pool fork, every DownsamplingBottleneck's pool-vs-reduce/expand
pair) CONCURRENTLY as independent streaming engines. A flat sum charges
BOTH branches' full cycle cost as if they ran one after another; the real
contribution of a join to end-to-end latency is max(branch cycles), not
their sum -- the faster branch just idles/buffers waiting for the slower
one (see MILP/scan_fork_join_mismatch.py's own `predicted_depth`, which
estimates exactly that buffering need). So finn_milp.py's own reported
latency (and anything gated on --max-latency-ms) systematically
OVERESTIMATES true single-image latency, growing with how much fork/join
structure the network has.

ALGORITHM: standard longest-path-in-a-DAG via dynamic programming, walked
in topological order. layer_topology.compute_predecessor_map already gives
the real dataflow edges (backward: consumer -> [real predecessors]); the
`geometries` list from finn_milp.trace_layer_geometry is ALREADY in a valid
topological order (forward hooks fire in true execution order), so no
separate topological sort is needed.

    critical_path_cycles[layer] = own_cycles + max(
        critical_path_cycles[p] for p in real_predecessors(layer), default=0
    )
    total_latency_cycles = max(critical_path_cycles.values())  # the network's own sink

This is a pure post-hoc REPORTING tool -- it does not change what
finn_milp.py solves for (see finn_milp.py's own --max-join-imbalance-ratio
for the constraint-side fix to the SAME underlying fork/join blind spot;
this script fixes how the resulting plan's latency gets REPORTED, not what
gets chosen). No FINN/HLS/rtlsim needed -- reuses the exact same per-layer
"cycles" number finn_milp.py's own cost model already computed and wrote
into the folding file's `per_layer` dict.

Usage (host Python, same env finn_milp.py itself runs in):
    python MILP/critical_path_latency.py \\
        --config config_12_dense_relu_warmstart150ep \\
        --folding-file MILP/artifacts/.../layer_bits_folding_....json \\
        --clock-mhz 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch  # noqa: F401 -- side-effect parity with finn_milp.py's own import order

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finn_milp import INPUT_HW, load_config, trace_layer_geometry  # noqa: E402
from layer_topology import compute_predecessor_map  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402


def compute_critical_path(
    geometries: list, per_layer: dict, predecessor_map: dict[str, list[str]],
) -> tuple[dict[str, float], dict[str, str | None]]:
    """Returns (critical_path_cycles, best_predecessor) -- the second dict
    records, for each layer, WHICH real predecessor was on its own critical
    path (the one with the largest critical_path_cycles value, ties broken
    by predecessor_map's own discovery order), so the actual bottleneck
    chain can be reconstructed by walking it backward from the sink. None
    for a layer with no real predecessor (the network's own first layer(s))."""
    critical_path_cycles: dict[str, float] = {}
    best_predecessor: dict[str, str | None] = {}
    for g in geometries:  # geometries is already in topological (execution) order
        own_cycles = per_layer[g.name]["cycles"]
        preds = [p for p in predecessor_map.get(g.name, []) if p in critical_path_cycles]
        if not preds:
            critical_path_cycles[g.name] = own_cycles
            best_predecessor[g.name] = None
        else:
            best = max(preds, key=lambda p: critical_path_cycles[p])
            critical_path_cycles[g.name] = own_cycles + critical_path_cycles[best]
            best_predecessor[g.name] = best
    return critical_path_cycles, best_predecessor


def reconstruct_path(sink: str, best_predecessor: dict[str, str | None]) -> list[str]:
    """Walks best_predecessor backward from `sink` to the network's own
    first layer, returning the chain in FORWARD (execution) order."""
    chain = [sink]
    node = sink
    while best_predecessor.get(node) is not None:
        node = best_predecessor[node]
        chain.append(node)
    return list(reversed(chain))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="MILP/configs/config_*.py module name (no .py).")
    parser.add_argument("--folding-file", type=Path, required=True,
                         help="finn_milp.py --out-file JSON (has a top-level 'per_layer' dict with a "
                              "'cycles' entry per layer name).")
    parser.add_argument("--clock-mhz", type=float, default=100.0,
                         help="Clock frequency to convert cycles -> ms (default 100.0, matches finn_milp.py's "
                              "own --clock-mhz default).")
    args = parser.parse_args()

    load_config(args.config)
    from finn_milp import IN_CHANNELS, CHANNELS, BOTTLENECKS_PER_STAGE, DECODER_TYPE, CONTEXT_PATTERN  # noqa: E402
    import finn_milp as fm

    model = ENet(
        in_channels=IN_CHANNELS, out_channels=fm.OUT_CHANNELS, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
        use_asymmetric=fm.USE_ASYMMETRIC, context_pattern=CONTEXT_PATTERN,
        separable_dilated=fm.SEPARABLE_DILATED, use_prelu=fm.__dict__.get("USE_PRELU", True),
        prelu_variant=fm.PRELU_VARIANT, use_dsc=fm.__dict__.get("USE_DSC", False),
        dsc_no_projection=fm.__dict__.get("DSC_NO_PROJECTION", False),
        dsc_no_projection_context_only=fm.__dict__.get("DSC_NO_PROJECTION_CONTEXT_ONLY", False),
        reg_bookend_dsc=fm.__dict__.get("REG_BOOKEND_DSC", False),
        dsc_separable=fm.__dict__.get("DSC_SEPARABLE", False),
    )
    geometries, _ = trace_layer_geometry(model, INPUT_HW, IN_CHANNELS)

    predecessor_map = compute_predecessor_map(model)

    with open(args.folding_file) as f:
        folding = json.load(f)
    per_layer = folding["per_layer"]

    missing = [g.name for g in geometries if g.name not in per_layer]
    if missing:
        raise ValueError(f"--folding-file is missing per_layer entries for: {missing} -- must be a "
                          f"finn_milp.py --out-file JSON traced against the SAME --config.")

    naive_sum_cycles = sum(v["cycles"] for v in per_layer.values())
    critical_path_cycles, best_predecessor = compute_critical_path(geometries, per_layer, predecessor_map)

    sink = max(critical_path_cycles, key=lambda name: critical_path_cycles[name])
    real_latency_cycles = critical_path_cycles[sink]
    chain = reconstruct_path(sink, best_predecessor)

    naive_sum_ms = naive_sum_cycles / (args.clock_mhz * 1000)
    real_latency_ms = real_latency_cycles / (args.clock_mhz * 1000)
    overestimate_pct = 100 * (naive_sum_cycles / real_latency_cycles - 1) if real_latency_cycles > 0 else float("nan")

    print(f"Folding file: {args.folding_file}")
    print(f"Layers: {len(geometries)}, clock: {args.clock_mhz}MHz")
    print()
    print(f"Naive flat sum   (finn_milp.py's own total_cycles): {naive_sum_cycles:>12.0f} cycles  ({naive_sum_ms:>8.3f} ms)")
    print(f"Real critical path (longest DAG path, this script): {real_latency_cycles:>12.0f} cycles  ({real_latency_ms:>8.3f} ms)")
    print(f"Naive sum overestimates real latency by: {overestimate_pct:.1f}%")
    print(f"Sink (network's own bottleneck-path endpoint): {sink}")
    print()
    print(f"Critical path chain ({len(chain)} layers, execution order):")
    for name in chain:
        print(f"  {name:<32} own_cycles={per_layer[name]['cycles']:>10.0f}  "
              f"cumulative={critical_path_cycles[name]:>12.0f}")


if __name__ == "__main__":
    main()

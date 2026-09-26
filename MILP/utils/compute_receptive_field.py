"""Computes the network's TRUE receptive field (in original 512x512 input
pixels) at every conv/pool layer, via the standard jump/RF accumulation
recursion, walking the real dataflow graph from layer_topology.py's
compute_predecessor_map (not naive sequential order, so it's correct across
skip-connection joins and DownsamplingBottleneck's pool/main split).

Answers: "downsampled feature maps cover more of the image per pixel, so
isn't global context baked in anyway?" -- growing RF via downsampling is
real (that's exactly what this script measures), but it's a bounded,
computable number, not automatically "the whole image". This tells us
whether it actually reaches close to 512 or tops out much smaller.

At a join (2+ predecessors, e.g. a DownsamplingBottleneck's pool+main
elementwise add, or an UpsamplingBottleneck's skip add), the combined RF is
max(RF of each predecessor) -- an elementwise add's output at a spatial
location depends on the UNION of whatever each operand's RF covers at that
same location -- and jump must match across predecessors (asserted).

Usage:
    python MILP/compute_receptive_field.py --config <config_module_name>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finn_milp import INPUT_HW, load_config, trace_layer_geometry  # noqa: E402
from layer_topology import compute_predecessor_map  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="MILP config module name, e.g. config_12_dense_relu")
    args = parser.parse_args()

    load_config(args.config)
    from finn_milp import IN_CHANNELS, CHANNELS, BOTTLENECKS_PER_STAGE, DECODER_TYPE, CONTEXT_PATTERN
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
    geom_by_name = {g.name: g for g in geometries}
    predecessor_map = compute_predecessor_map(model)

    # jump/RF per axis, in input-pixel units. jump = cumulative stride so
    # far (how many input pixels one step in this layer's own input grid
    # corresponds to). RF = receptive field size so far, in input pixels.
    jump_h: dict[str, float] = {}
    jump_w: dict[str, float] = {}
    rf_h: dict[str, float] = {}
    rf_w: dict[str, float] = {}

    rows = []
    for g in geometries:  # trace order == execution order == topological order
        preds = predecessor_map.get(g.name, [])
        if not preds:
            jh_in, jw_in, rh_in, rw_in = 1.0, 1.0, 1.0, 1.0
        else:
            known = [p for p in preds if p in jump_h]
            jh_in = jump_h[known[0]]
            jw_in = jump_w[known[0]]
            for p in known[1:]:
                assert abs(jump_h[p] - jh_in) < 1e-6, f"{g.name}: jump_h mismatch across predecessors {preds}"
                assert abs(jump_w[p] - jw_in) < 1e-6, f"{g.name}: jump_w mismatch across predecessors {preds}"
            rh_in = max(rf_h[p] for p in known)
            rw_in = max(rf_w[p] for p in known)

        k_eff_h = g.dh * (g.kh - 1) + 1
        k_eff_w = g.dw * (g.kw - 1) + 1
        rh_out = rh_in + (k_eff_h - 1) * jh_in
        rw_out = rw_in + (k_eff_w - 1) * jw_in

        if g.op_type == "ConvTranspose2d":
            jh_out = jh_in / g.sh
            jw_out = jw_in / g.sw
        else:  # Conv2d, MaxPool2d
            jh_out = jh_in * g.sh
            jw_out = jw_in * g.sw

        jump_h[g.name] = jh_out
        jump_w[g.name] = jw_out
        rf_h[g.name] = rh_out
        rf_w[g.name] = rw_out
        rows.append((g.name, g.stage, g.op_type, rh_out, rw_out, jh_out, jw_out))

    print(f"{'layer':<38} {'stage':<16} {'op':<15} {'RF_h':>8} {'RF_w':>8} {'jump_h':>8} {'jump_w':>8}")
    for name, stage, op, rh, rw, jh, jw in rows:
        print(f"{name:<38} {stage:<16} {op:<15} {rh:>8.0f} {rw:>8.0f} {jh:>8.1f} {jw:>8.1f}")

    final_rh = max(rf_h.values())
    final_rw = max(rf_w.values())
    input_h, input_w = INPUT_HW
    print()
    print(f"Max receptive field reached anywhere in the network: {final_rh:.0f} x {final_rw:.0f} "
          f"(input image is {input_h} x {input_w})")
    print(f"As a fraction of the full image: {final_rh / input_h:.1%} x {final_rw / input_w:.1%}")


if __name__ == "__main__":
    main()

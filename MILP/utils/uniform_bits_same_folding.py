"""Builds a "naive uniform quantization" baseline for comparison against a
real per-layer joint bits+folding ILP result: takes an EXISTING solved
layer_bits_folding_*.json (e.g. the alpha=0.25 solve for 12_dense_relu_
warmstart150ep), keeps every layer's own real (pe, simd, ram_style) folding
choice UNCHANGED, and overrides weight_bits/act_bits to a single uniform
value at every layer -- then recomputes every derived cost field (total_lut,
swu_bram18, wm_bram18, thr_bram18, total_dsp, cycles, ...) at that new
bit-width via this repo's own finn_cost_model.layer_cost_pe_simd, so the
output is a fully self-consistent layer_bits_folding_*.json (not just
bit-overridden with stale cost fields), usable directly by
expand_layer_bits.py exactly like a real ILP solve would be.

WHY hold folding fixed rather than re-solving it per bit-width: the whole
point of this baseline is to isolate "does per-layer BIT ALLOCATION matter",
holding the hardware STRUCTURE (which layer gets how much PE/SIMD
parallelism) fixed at whatever the real alpha=0.25 solve already chose --
re-solving folding fresh for each uniform bit-width would let the folding
search compensate for the bit change, which would answer a different
question (do overall best achievable-under-70%-LUT designs compare
favorably) rather than the one asked (at the SAME hardware structure, does
smarter bit allocation beat uniform).

2026-09-29: also recomputes every EXTRA node (residual-join/Initial
thresholds, AddStreams/DuplicateStreams/StreamingConcat/Upsample, the
downsampling pad-MVAU) at the same fixed (pe, simd, ram_style) with bits
updated the same "hold structure, override bits" way -- previously this
script only touched `per_layer` (real conv/pool layers), so its totals
silently under-counted relative to a real `finn_milp.py` solve's
`_diagnostics.total_lut_calibrated`/`total_bram18k_calibrated` (which
include both), making any direct comparison apples-to-oranges. Reuses
`finn_milp.build_model_and_graph()`/`extra_node_options` rather than
re-deriving each extra-node kind's cost formula here, so the two can never
silently diverge.

Usage:
    python MILP/uniform_bits_same_folding.py \\
        --config config_12_dense_relu_warmstart150ep \\
        --reference-ilp-result MILP/artifacts/archive/12_dense_relu_warmstart150ep_ILP_outputs_perlayer_forcedsp_lut70/layer_bits_folding_12_dense_relu_warmstart150ep_joint_alpha0.25_candidatebits468_forcedsp_lut70.json \\
        --uniform-bits 4,6,8 \\
        --out-dir MILP/artifacts/archive/12_dense_relu_warmstart150ep_uniform_samefolding_alpha0.25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finn_cost_model import calibrated_bram18k, calibrated_lut, layer_cost_pe_simd  # noqa: E402
import finn_milp  # noqa: E402 -- module object, not just names: extra_node_options reads finn_milp's own
                   # CANDIDATE_BITS global, which this script must set before calling it (see main()); its
                   # own load_config() also populates finn_milp's globals, which build_model_and_graph() needs.
from finn_milp import (  # noqa: E402
    RAM_STYLE_AUTO, STREAM_NODE_KINDS, VARIANT_HLS_LUT_NOACT0, VARIANT_RTL_DSP_NOACT1, _calibration_force_dsp,
    _variant_cost_kwargs, build_model_and_graph, extra_node_options,
)

XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624, "DSP": 1_728}


def _recompute_extra_node(node, ref_entry: dict, uniform_bits: int, force_dsp: bool) -> dict:
    """Same idea as the per_layer loop below, for one ExtraNode: keep its real
    (pe, simd, ram_style) fixed at whatever the reference solve chose, pick its
    new bits per the SAME rule the ILP itself used (fixed_bits nodes never
    depend on layer bits at all; bit_sources nodes become the uniform value,
    since every source is now uniform; stream/pad-MVAU nodes keep their
    original bits -- pad-MVAU's are a fixed constant unrelated to per-layer
    bits, stream nodes' cost doesn't depend on bits at all), then look up that
    EXACT (fold, bits) option from `extra_node_options` (reused, not
    re-derived) and apply the same calibrated-vs-raw split `finn_milp.py`'s
    own solve does for STREAM_NODE_KINDS vs. everything else."""
    pe, simd, ram_style = ref_entry["pe"], ref_entry["simd"], ref_entry["ram_style"]
    if node.fixed_bits is not None:
        target_bits = node.fixed_bits
    elif node.bit_sources:
        target_bits = uniform_bits
    else:
        target_bits = ref_entry["act_bits"] or 0  # stream kinds: 0 (cost-irrelevant); pad_mvau: its real fixed bits
    options = extra_node_options(node, force_dsp)
    matches = [
        (key, cost) for key, cost in options
        if key[1] == pe and key[2] == simd and key[3] == ram_style and key[6] == target_bits
    ]
    if not matches:
        raise ValueError(
            f"{node.geom.name}: no extra_node_options match for pe={pe} simd={simd} ram_style={ram_style} "
            f"bits={target_bits} (reference fold no longer legal for this node under the current config/code?)"
        )
    key, cost = matches[0]
    _, _, _, _, variant, w, bits = key
    calib_w = w or bits
    if node.kind in STREAM_NODE_KINDS:
        lut_calibrated = cost["total_lut"]
        bram_calibrated = cost["swu_bram18"] + cost["wm_bram18"] + cost["thr_bram18"]
    else:
        rtl_kwargs = _variant_cost_kwargs(VARIANT_RTL_DSP_NOACT1, force_dsp)
        lut_calibrated = calibrated_lut(cost["total_lut"], calib_w, bits, force_dsp=_calibration_force_dsp(rtl_kwargs))
        bram_calibrated = calibrated_bram18k(
            cost["swu_bram18"] + cost["wm_bram18"] + cost["thr_bram18"], calib_w, bits, force_dsp=rtl_kwargs["force_dsp"],
        )
    return {
        "kind": node.kind, "stage": node.geom.stage, "pe": pe, "simd": simd, "ram_style": ram_style,
        "weight_bits": w or None, "act_bits": bits or None, "bit_sources": list(node.bit_sources),
        "channels": node.geom.cout, "cycles": cost["cycles"], "lut_calibrated": lut_calibrated,
        "bram18k_calibrated": bram_calibrated, "dsp": cost["total_dsp"],
        "uram18": cost["wm_uram18"] + cost.get("swu_uram18", 0) + cost["thr_uram18"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-ilp-result", type=Path, required=True,
                         help="A real, already-solved layer_bits_folding_*.json (per_layer dict with real "
                              "pe/simd/ram_style/weight_bits/act_bits) whose FOLDING this baseline keeps fixed.")
    parser.add_argument("--uniform-bits", type=str, required=True,
                         help="Comma-separated uniform (weight_bits==act_bits) targets, e.g. '4,6,8' -- one "
                              "output file per value.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    # finn_milp.load_config injects the config into FINN_MILP's OWN module globals
    # (not this script's) -- build_model_and_graph() (defined there) reads them from
    # there, since it's shared with finn_milp.py's own main().
    finn_milp.load_config(args.config)

    with open(args.reference_ilp_result) as f:
        reference = json.load(f)
    ref_per_layer = reference["per_layer"]
    ref_extra_nodes = reference.get("extra_nodes", {})

    uniform_bits_list = [int(b) for b in args.uniform_bits.split(",")]
    # extra_node_options reads finn_milp's own CANDIDATE_BITS global for THRESHOLD_KINDS
    # nodes whose bits derive from bit_sources -- must cover every requested uniform
    # value or the lookup below raises "no extra_node_options match".
    finn_milp.CANDIDATE_BITS = tuple(sorted(set(finn_milp.CANDIDATE_BITS) | set(uniform_bits_list)))

    model, geometries, extra_nodes, _predecessor_map, _dataflow_map, _node_kinds = build_model_and_graph()
    geom_by_name = {g.name: g for g in geometries}
    extra_by_name = {n.geom.name: n for n in extra_nodes}

    missing = set(ref_per_layer) - set(geom_by_name)
    if missing:
        raise ValueError(f"Reference ILP result has layer name(s) not found in this config's own traced "
                          f"geometry: {sorted(missing)} -- wrong --config for this reference file?")
    missing_extra = set(ref_extra_nodes) - set(extra_by_name)
    if missing_extra:
        raise ValueError(f"Reference ILP result has extra node name(s) not found in this config's own traced "
                          f"dataflow graph: {sorted(missing_extra)} -- wrong --config for this reference file?")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for bits in uniform_bits_list:
        new_per_layer = {}
        for name, ref_entry in ref_per_layer.items():
            geom = geom_by_name[name]
            # Real per_layer schema (finn_milp.py:679-683) keys the threshold ram_style as
            # "thr_ram_style" -- MVAU/VVAU weight memory's own ram_style is NOT part of this
            # key at all, it's hard-fixed to RAM_STYLE_AUTO at solve time regardless of fold.
            pe, simd, thr_ram_style, variant = (
                ref_entry["pe"], ref_entry["simd"], ref_entry["thr_ram_style"], ref_entry["variant"],
            )
            variant_kwargs = _variant_cost_kwargs(variant, force_dsp=True)
            cost = layer_cost_pe_simd(
                geom, bits, bits, pe, simd, RAM_STYLE_AUTO,
                swu_ram_style="distributed", thr_ram_style=thr_ram_style, **variant_kwargs,
            )
            new_per_layer[name] = {
                "stage": ref_entry["stage"], "pe": pe, "simd": simd, "thr_ram_style": thr_ram_style,
                "variant": variant, "force_dsp": variant_kwargs["force_dsp"],
                "mvau_noAct": variant_kwargs["no_activation"], "weight_bits": bits, "act_bits": bits, **cost,
            }
        new_extra_nodes = {
            name: _recompute_extra_node(extra_by_name[name], ref_entry, bits, force_dsp=True)
            for name, ref_entry in ref_extra_nodes.items()
        }

        # Post-hoc calibration mirrors finn_milp.py:685-694 exactly -- per-layer variant
        # (not a blanket force_dsp=True) decides the calibration branch, since
        # rtl_dsp_noact1 always takes the RTL derate regardless of --force-dsp (see
        # finn_milp.md's `_calibration_force_dsp` note) while hls_lut_noact0 does not.
        total_lut = sum(
            calibrated_lut(
                v["total_lut"], v["weight_bits"], v["act_bits"],
                force_dsp=_calibration_force_dsp(_variant_cost_kwargs(v["variant"], True)),
                lut_mult=(v["variant"] == VARIANT_HLS_LUT_NOACT0),
            )
            for v in new_per_layer.values()
        )
        total_bram = sum(
            calibrated_bram18k(
                v["swu_bram18"] + v["wm_bram18"] + v.get("thr_bram18", 0), v["weight_bits"], v["act_bits"],
                force_dsp=_variant_cost_kwargs(v["variant"], True)["force_dsp"],
            )
            for v in new_per_layer.values()
        )
        total_uram = sum(v.get("wm_uram18", 0) for v in new_per_layer.values())
        total_dsp = sum(v.get("total_dsp", 0) for v in new_per_layer.values())
        total_cycles = sum(v["cycles"] for v in new_per_layer.values())
        extra_lut = sum(v["lut_calibrated"] for v in new_extra_nodes.values())
        extra_bram = sum(v["bram18k_calibrated"] for v in new_extra_nodes.values())
        extra_dsp = sum(v["dsp"] for v in new_extra_nodes.values())
        extra_uram = sum(v["uram18"] for v in new_extra_nodes.values())
        extra_cycles = sum(v["cycles"] for v in new_extra_nodes.values())
        total_lut += extra_lut
        total_bram += extra_bram
        total_dsp += extra_dsp
        total_uram += extra_uram
        total_cycles += extra_cycles

        result = {
            "status": "Optimal",  # not solved -- a fixed, valid assignment; "Optimal" only in the sense
                                   # expand_layer_bits.py's own status check expects to see this literal value
            "alpha": None,
            # expand_layer_bits.py reads THESE top-level dicts (not per_layer) -- must match
            # joint_bits_folding_ilp_perlayer.py's own real output schema exactly.
            "layer_weight_bits": {name: bits for name in new_per_layer},
            "layer_act_bits": {name: bits for name in new_per_layer},
            "per_layer": new_per_layer,
            "extra_nodes": new_extra_nodes,
            "_diagnostics": {
                "total_lut_calibrated": total_lut, "xczu7ev_lut_budget": XCZU7EV["LUT"],
                "lut_pct_of_budget": 100 * total_lut / XCZU7EV["LUT"],
                "total_bram18k_calibrated": total_bram, "xczu7ev_bram18k_budget": XCZU7EV["BRAM_18K"],
                "bram_pct_of_budget": 100 * total_bram / XCZU7EV["BRAM_18K"],
                "total_uram18": total_uram, "total_cycles": total_cycles,
                "total_dsp": total_dsp, "xczu7ev_dsp_budget": XCZU7EV["DSP"],
                "dsp_pct_of_budget": 100 * total_dsp / XCZU7EV["DSP"],
                "force_dsp": True,
                "extra_lut_calibrated": extra_lut, "extra_bram18k_calibrated": extra_bram,
                "extra_dsp": extra_dsp, "extra_cycles": extra_cycles,
                "note": f"NAIVE UNIFORM INT{bits} baseline -- NOT ILP-solved. Every layer's/extra-node's real "
                        f"(pe, simd, ram_style) folding choice is copied UNCHANGED from "
                        f"{args.reference_ilp_result.name}; only weight_bits/act_bits are overridden to a "
                        f"uniform {bits} (extra nodes: fixed-bit and stream/pad-MVAU nodes keep their original "
                        f"bits, since those don't derive from per-layer bit choice) and every derived cost field "
                        f"recomputed at that bit-width via this repo's own finn_cost_model.layer_cost_pe_simd / "
                        f"finn_milp.extra_node_options, forced-DSP calibration. Totals include extra nodes "
                        f"(thresholds/streams/pad-MVAU), matching finn_milp.py's own real-solve totals -- fixed "
                        f"2026-09-29, previously per_layer-only and under-counted relative to a real solve. "
                        f"Purpose: isolate whether smarter per-layer bit allocation beats naive uniform "
                        f"quantization AT THE SAME hardware folding structure (not re-optimized per bit-width).",
            },
        }
        out_file = args.out_dir / f"layer_bits_folding_uniform_int{bits}_samefolding.json"
        out_file.write_text(json.dumps(result, indent=2))
        print(f"INT{bits}: LUT={total_lut:.0f} ({100*total_lut/XCZU7EV['LUT']:.1f}%)  "
              f"BRAM_18K={total_bram:.0f} ({100*total_bram/XCZU7EV['BRAM_18K']:.1f}%)  "
              f"cycles={total_cycles:.0f} ({total_cycles/100000:.2f}ms@100MHz)  -- wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

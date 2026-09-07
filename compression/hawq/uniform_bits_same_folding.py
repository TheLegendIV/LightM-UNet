"""Builds a "naive uniform quantization" baseline for comparison against a
real per-layer joint bits+folding ILP result: takes an EXISTING solved
layer_bits_folding_*.json (e.g. the alpha=0.25 solve for 12_dense_relu_
warmstart150ep), keeps every layer's own real (pe, simd, ram_style) folding
choice UNCHANGED, and overrides weight_bits/act_bits to a single uniform
value at every layer -- then recomputes every derived cost field (total_lut,
swu_bram18, wm_bram18, cycles, ...) at that new bit-width via this repo's
own finn_cost_model.layer_cost_pe_simd, so the output is a fully self-
consistent layer_bits_folding_*.json (not just bit-overridden with stale
cost fields), usable directly by expand_layer_bits.py exactly like a real
ILP solve would be.

WHY hold folding fixed rather than re-solving it per bit-width: the whole
point of this baseline is to isolate "does per-layer BIT ALLOCATION matter",
holding the hardware STRUCTURE (which layer gets how much PE/SIMD
parallelism) fixed at whatever the real alpha=0.25 solve already chose --
re-solving folding fresh for each uniform bit-width would let the folding
search compensate for the bit change, which would answer a different
question (do overall best achievable-under-70%-LUT designs compare
favorably) rather than the one asked (at the SAME hardware structure, does
smarter bit allocation beat uniform).

Usage:
    python compression/hawq/uniform_bits_same_folding.py \\
        --config config_12_dense_relu_warmstart150ep \\
        --reference-ilp-result compression/hawq/artifacts/12_dense_relu_warmstart150ep_ILP_outputs_perlayer_forcedsp_lut70/layer_bits_folding_12_dense_relu_warmstart150ep_joint_alpha0.25_candidatebits468_forcedsp_lut70.json \\
        --uniform-bits 4,6,8 \\
        --out-dir compression/hawq/artifacts/12_dense_relu_warmstart150ep_uniform_samefolding_alpha0.25
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from finn_block_costs import dump_block_layer_geometry  # noqa: E402
from finn_cost_model import calibrated_bram18k, calibrated_lut, layer_cost_pe_simd  # noqa: E402
from finn_stage_costs import INPUT_HW  # noqa: E402

XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624}


def load_config(config_module: str) -> None:
    cfg = importlib.import_module(config_module)
    globals().update({k: v for k, v in vars(cfg).items() if not k.startswith("_")})


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

    load_config(args.config)

    with open(args.reference_ilp_result) as f:
        reference = json.load(f)
    ref_per_layer = reference["per_layer"]

    model = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
        use_asymmetric=USE_ASYMMETRIC, context_pattern=CONTEXT_PATTERN,
        separable_dilated=SEPARABLE_DILATED, use_prelu=globals().get("USE_PRELU", True), prelu_variant=PRELU_VARIANT,
        use_dsc=globals().get("USE_DSC", False), dsc_no_projection=globals().get("DSC_NO_PROJECTION", False),
    )
    geometries, _block_names = dump_block_layer_geometry(model, INPUT_HW)
    geom_by_name = {g.name: g for g in geometries}

    missing = set(ref_per_layer) - set(geom_by_name)
    if missing:
        raise ValueError(f"Reference ILP result has layer name(s) not found in this config's own traced "
                          f"geometry: {sorted(missing)} -- wrong --config for this reference file?")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for bits_str in args.uniform_bits.split(","):
        bits = int(bits_str)
        new_per_layer = {}
        for name, ref_entry in ref_per_layer.items():
            geom = geom_by_name[name]
            pe, simd, ram_style = ref_entry["pe"], ref_entry["simd"], ref_entry["ram_style"]
            cost = layer_cost_pe_simd(geom, bits, bits, pe, simd, ram_style)
            new_per_layer[name] = {
                "stage": ref_entry["stage"], "pe": pe, "simd": simd, "ram_style": ram_style,
                "weight_bits": bits, "act_bits": bits, **cost,
            }

        total_lut = sum(calibrated_lut(v["total_lut"], v["weight_bits"], v["act_bits"], force_dsp=True) for v in new_per_layer.values())
        total_bram = sum(
            calibrated_bram18k(v["swu_bram18"] + v["wm_bram18"], v["weight_bits"], v["act_bits"], force_dsp=True)
            for v in new_per_layer.values()
        )
        total_uram = sum(v.get("wm_uram18", 0) for v in new_per_layer.values())
        total_cycles = sum(v["cycles"] for v in new_per_layer.values())

        result = {
            "status": "Optimal",  # not solved -- a fixed, valid assignment; "Optimal" only in the sense
                                   # expand_layer_bits.py's own status check expects to see this literal value
            "alpha": None,
            # expand_layer_bits.py reads THESE top-level dicts (not per_layer) -- must match
            # joint_bits_folding_ilp_perlayer.py's own real output schema exactly.
            "layer_weight_bits": {name: bits for name in new_per_layer},
            "layer_act_bits": {name: bits for name in new_per_layer},
            "per_layer": new_per_layer,
            "_diagnostics": {
                "total_lut_calibrated": total_lut, "xczu7ev_lut_budget": XCZU7EV["LUT"],
                "lut_pct_of_budget": 100 * total_lut / XCZU7EV["LUT"],
                "total_bram18k_calibrated": total_bram, "xczu7ev_bram18k_budget": XCZU7EV["BRAM_18K"],
                "bram_pct_of_budget": 100 * total_bram / XCZU7EV["BRAM_18K"],
                "total_uram18": total_uram, "total_cycles": total_cycles,
                "force_dsp": True,
                "note": f"NAIVE UNIFORM INT{bits} baseline -- NOT ILP-solved. Every layer's real (pe, simd, "
                        f"ram_style) folding choice is copied UNCHANGED from {args.reference_ilp_result.name}; "
                        f"only weight_bits/act_bits are overridden to a uniform {bits} and every derived cost "
                        f"field recomputed at that bit-width via this repo's own finn_cost_model.layer_cost_"
                        f"pe_simd, forced-DSP calibration. Purpose: isolate whether smarter per-layer bit "
                        f"allocation beats naive uniform quantization AT THE SAME hardware folding structure "
                        f"(not re-optimized per bit-width).",
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

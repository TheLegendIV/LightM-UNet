"""Expands joint_bits_folding_ilp_perlayer.py's own output (one (weight_bits,
act_bits) pair per real Conv2d/ConvTranspose2d/MaxPool2d LAYER, ~104 for S12)
into nnunetv2.nets.LayerQuantENet's full per-QUANTIZER-SITE schema (~101
weight sites + ~126 act sites for S12) -- the missing bridge explicitly
flagged as future work in that ILP's own module docstring ("SCOPE BOUNDARY").
Without this, the per-layer ILP's output cannot be fed to LayerQuantENet at
all: LayerQuantENet demands exactly one entry per quantizer site it actually
builds (reduce/conv_bn_act/residual_add/out_act each independently, not one
shared act_bits per whole conv "layer"), and strictly rejects both missing
AND unrecognized extra keys.

WEIGHT sites need no real work: every LayerQuantENet weight site name IS
literally one of the ILP's own layer names (confirmed exact key-set match
this session, block_utils.block_weight_targets vs. finn_block_costs.
dump_block_layer_geometry) -- direct passthrough.

ACT sites are the real gap: LayerQuantENet independently parametrizes
several activation quantizers per block (e.g. RegularBottleneck's own
reduce.2/conv_bn_act.2/residual_add/out_act), but the per-layer ILP's cost
model (and hence its z/y variables) only ever assigns ONE act_bits per
whole conv "layer" -- its own INPUT stream's precision (see
finn_cost_model.py's BRAM_swu term / joint_bits_folding_ilp_perlayer.py's
own module docstring). This file resolves each such act site to the
nearest REAL conv/pool layer(s) whose act_bits value represents (or, for a
join, contributes to) that exact wire:

  - Sites strictly INSIDE one block, downstream of one specific local conv
    (reduce.2 <- reduce.0, conv.2 <- conv.0, conv_bn_act.2 <- whichever of
    conv/conv.1/conv.3 that block actually has, up.2 <- up.0,
    input_quant <- conv) -- a static, hand-known mapping (these local
    structures are fixed by LayerQuantENet.py's own construction code, not
    architecture-search-dependent, so no graph tracing is needed for them).

  - residual_add/out_act (no local conv immediately precedes either -- both
    sit downstream of a residual JOIN) -- resolved per block TYPE:
      * UpsamplingBottleneck: both real operands (main_proj, expand) are
        LOCAL to the same block -- no cross-block tracing needed.
      * DownsamplingBottleneck: both real operands (pool, expand) are ALSO
        local (the "main" branch is literally MaxPool2d(x), itself one of
        this block's own geometry entries) -- no cross-block tracing needed.
      * RegularBottleneck/DSCNoProjectionBottleneck: ONE operand is local
        (expand.0 / conv.3, this block's own last computation), the OTHER
        is the block's own raw INPUT -- i.e. whatever fed its OWN first
        conv (reduce.0 / conv.0) from OUTSIDE the block. That IS a real
        cross-block relationship, resolved via
        compression/hawq/layer_topology.compute_predecessor_map on a plain
        FP32 mirror (dataflow topology is identical between the FP32 and
        Brevitas-quantized architectures -- an already-established,
        verified property of this codebase).

  - InitialBlock's own trailing "act" (downstream of concatenating conv AND
    pool) -- both operands (conv, pool) are local, no tracing needed.

Every act site's final value is the MAX across however many real sources it
resolves to -- consistent with joint_bits_folding_ilp_perlayer.py's own
aggregation rule for the same reason: a wire is only as safe to compress as
its most sensitive/demanding real contributor.

Usage:
    python compression/hawq/expand_layer_bits.py \\
        --config config_12_separable_dense_relu \\
        --ilp-result compression/hawq/artifacts/S12_ILP_outputs_perlayer/layer_bits_folding_12_separable_dense_relu_joint_alpha0.5_candidatebits468_maxlat1000ms.json \\
        --out-file compression/hawq/artifacts/S12_ILP_outputs_perlayer/layer_bits_SITES_12_separable_dense_relu_joint_alpha0.5_candidatebits468_maxlat1000ms.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer_topology import compute_predecessor_map  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402

# Longest/most-specific tails checked first -- "out_act"/"residual_add" have
# no "." before "act"/"add" that would collide with the bare "act" entry
# (InitialBlock's own trailing activation), so order among those doesn't
# matter, but reduce.2/reduce.3 and conv.2/conv_bn_act.2 must each be
# matched as a whole two-segment tail, never truncated to just ".2".
ACT_SUFFIXES_ORDERED = (
    "reduce.2", "reduce.3", "conv_bn_act.2", "conv.2", "up.2",
    "input_quant", "residual_add", "out_act", "act",
)


def load_config(config_module: str) -> None:
    cfg = importlib.import_module(config_module)
    globals().update({k: v for k, v in vars(cfg).items() if not k.startswith("_")})


def _match_suffix(name: str) -> tuple[str, str]:
    for suf in ACT_SUFFIXES_ORDERED:
        tail = f".{suf}"
        if name.endswith(tail):
            return name[: -len(tail)], suf
    raise ValueError(f"Unrecognized act site name (no known suffix matched): {name!r}")


def _conv_bn_act_local_source(block_prefix: str, layer_weight_bits: dict[str, int]) -> list[str]:
    for candidate in ("conv", "conv.1", "conv.3"):
        key = f"{block_prefix}.{candidate}"
        if key in layer_weight_bits:
            return [key]
    raise ValueError(f"No conv found for {block_prefix}'s own conv_bn_act.2 (checked conv/conv.1/conv.3).")


def _block_type(block_prefix: str, layer_weight_bits: dict[str, int], layer_act_bits: dict[str, int]) -> str:
    if f"{block_prefix}.main_proj.0" in layer_weight_bits:
        return "upsampling"
    if f"{block_prefix}.pool" in layer_act_bits:
        return "downsampling"
    if f"{block_prefix}.reduce.0" in layer_weight_bits:
        return "regular"
    if f"{block_prefix}.conv.0" in layer_weight_bits:
        return "dsc_no_projection"
    raise ValueError(f"Cannot classify block type for {block_prefix!r} (no reduce.0/conv.0/main_proj.0/pool found).")


def _residual_join_sources(
    block_prefix: str, block_type: str, predecessor_map: dict[str, list[str]],
) -> list[str]:
    """residual_add/out_act's own real sources -- see module docstring for
    why each block type resolves differently (local-only for Downsampling/
    Upsampling, one local + one real cross-block predecessor for Regular/
    DSCNoProjection)."""
    if block_type == "upsampling":
        return [f"{block_prefix}.main_proj.0", f"{block_prefix}.expand.0"]
    if block_type == "downsampling":
        return [f"{block_prefix}.pool", f"{block_prefix}.expand.0"]
    if block_type == "regular":
        first, last = f"{block_prefix}.reduce.0", f"{block_prefix}.expand.0"
    elif block_type == "dsc_no_projection":
        first, last = f"{block_prefix}.conv.0", f"{block_prefix}.conv.3"
    else:
        raise ValueError(block_type)
    cross_block = predecessor_map.get(first, [])
    return list(dict.fromkeys([last, *cross_block]))  # dedupe, preserve order


def resolve_act_sources(
    act_site_name: str, layer_weight_bits: dict[str, int], layer_act_bits: dict[str, int],
    predecessor_map: dict[str, list[str]],
) -> list[str]:
    block_prefix, suffix = _match_suffix(act_site_name)
    if suffix == "reduce.2":
        return [f"{block_prefix}.reduce.0"]
    if suffix == "reduce.3":
        return [f"{block_prefix}.reduce.1"]
    if suffix == "conv.2":
        return [f"{block_prefix}.conv.0"]
    if suffix == "up.2":
        return [f"{block_prefix}.up.0"]
    if suffix == "conv_bn_act.2":
        return _conv_bn_act_local_source(block_prefix, layer_weight_bits)
    if suffix == "input_quant":
        return [f"{block_prefix}.conv"]
    if suffix == "act":
        # InitialBlock's own trailing act, downstream of torch.cat([conv(x), pool(x)]) -- both local.
        return [f"{block_prefix}.conv", f"{block_prefix}.pool"]
    if suffix in ("residual_add", "out_act"):
        block_type = _block_type(block_prefix, layer_weight_bits, layer_act_bits)
        return _residual_join_sources(block_prefix, block_type, predecessor_map)
    raise ValueError(f"Unhandled suffix {suffix!r} for act site {act_site_name!r}")


def expand_layer_bits_to_site_bits(
    ilp_result: dict, weight_site_names: tuple[str, ...], act_site_names: tuple[str, ...],
    predecessor_map: dict[str, list[str]],
) -> tuple[dict[str, int], dict[str, int]]:
    layer_weight_bits = ilp_result["layer_weight_bits"]
    layer_act_bits = ilp_result["layer_act_bits"]

    site_weight_bits = {name: layer_weight_bits[name] for name in weight_site_names}

    site_act_bits: dict[str, int] = {}
    for name in act_site_names:
        sources = resolve_act_sources(name, layer_weight_bits, layer_act_bits, predecessor_map)
        values = [layer_act_bits[s] for s in sources if s in layer_act_bits]
        if not values:
            raise ValueError(f"No act_bits value resolvable for site {name!r} (sources tried: {sources}).")
        site_act_bits[name] = max(values)
    return site_weight_bits, site_act_bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_12_separable_dense_relu",
                         help="Which compression/hawq/config_*.py to load -- defines the architecture shape "
                              "layer_names_for/compute_predecessor_map need (must match the ILP result's own).")
    parser.add_argument("--ilp-result", type=Path, required=True,
                         help="A layer_bits_folding_*.json from joint_bits_folding_ilp_perlayer.py.")
    parser.add_argument("--out-file", type=Path, required=True)
    args = parser.parse_args()

    load_config(args.config)

    model = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
        use_asymmetric=USE_ASYMMETRIC, context_pattern=CONTEXT_PATTERN,
        separable_dilated=SEPARABLE_DILATED, use_prelu=globals().get("USE_PRELU", True), prelu_variant=PRELU_VARIANT,
        use_dsc=globals().get("USE_DSC", False), dsc_no_projection=globals().get("DSC_NO_PROJECTION", False),
        dsc_no_projection_context_only=globals().get("DSC_NO_PROJECTION_CONTEXT_ONLY", False),
        reg_bookend_dsc=globals().get("REG_BOOKEND_DSC", False),
        dsc_separable=globals().get("DSC_SEPARABLE", False),
    )
    predecessor_map = compute_predecessor_map(model)

    weight_site_names, act_site_names = layer_names_for(
        out_channels=OUT_CHANNELS, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=USE_ASYMMETRIC, use_strided=True,
        use_dsc=globals().get("USE_DSC", False), dsc_no_projection=globals().get("DSC_NO_PROJECTION", False),
        dsc_no_projection_context_only=globals().get("DSC_NO_PROJECTION_CONTEXT_ONLY", False),
        separable_dilated=SEPARABLE_DILATED,
    )

    with open(args.ilp_result) as f:
        ilp_result = json.load(f)
    if ilp_result.get("status") != "Optimal":
        raise ValueError(f"{args.ilp_result} status={ilp_result.get('status')!r}, not Optimal -- nothing to expand.")

    site_weight_bits, site_act_bits = expand_layer_bits_to_site_bits(
        ilp_result, weight_site_names, act_site_names, predecessor_map,
    )

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump({"layer_weight_bits": site_weight_bits, "layer_act_bits": site_act_bits}, f, indent=2)
    print(f"Wrote {args.out_file} ({len(site_weight_bits)} weight sites, {len(site_act_bits)} act sites).")


if __name__ == "__main__":
    main()

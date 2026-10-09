"""One naming scheme for the analytical block models (MILP/analytical/*) and the MILP (finn_milp.py, per_layer / extra_nodes / dataflow_graph).

The MILP names are the canonical ones: a hardware node belongs to a MILP entry (a conv/pool layer or an extra node, e.g. `regular1.0.conv`, `regular1.0.skip_quant`,
`initial.act`, `final.argmax`) and, because one MILP layer entry bundles several FINN nodes (MVAU + its threshold + the window generator + the padding stage), a `part`.
The analytical blocks call the same nodes MVAU_m, Thr_m, SWG_m, FMPad, Thr_s, ... This module is the single table between the two vocabularies:

    milp_node(stage, kind, "SWG_m")  ->  ("regular1.0.conv", "swg")

Writers (net_fold intra/inter FIFO lists, net_onnx node names) add the MILP name next to the analytical role they already carry, so nothing that reads the old role
strings (the FINN bridge in hardware/finn_s12_build_steps.py) changes. Pure data, no imports of torch / pulp.

Block kinds follow net_fold.block_kind: reg (identity residual bottleneck), dn (downsampling), up (upsampling), init (initial block), final (last deconv + bias + argmax).
"""
from __future__ import annotations

# analytical node name -> (MILP leaf under the stage, part). leaf None = the node's MILP name is producer-qualified (dup) and built by the caller.
_COMMON = {
    "MVAU_r": ("reduce.0", "mvau"), "Thr_r": ("reduce.0", "thr"),
    "MVAU_e": ("expand.0", "mvau"), "Thr_e": ("expand.0", "thr"),
    "Thr_s": ("skip_quant", "thr"), "Add": ("add", "add"), "Thr_out": ("out_act", "thr"),
    "Dup": (None, "dup"),
}
_TABLE = {
    "reg": {**_COMMON, "MVAU_m": ("conv", "mvau"), "Thr_m": ("conv", "thr"), "FMPad": ("conv", "fmpad"), "SWG_m": ("conv", "swg")},
    "dn": {**_COMMON, "MVAU_m": ("conv.0", "mvau"), "Thr_m": ("conv.0", "thr"), "FMPad": ("conv.0", "fmpad"), "SWG_m": ("conv.0", "swg"),
           "SWG_r": ("reduce.0", "swg"), "MVAU_s": ("skip_pad", "mvau"), "MaxPool": ("pool", "pool"), "SWG_p": ("pool", "swg"), "Pool": ("pool", "pool"),
           "FMPad_c": ("skip_pad", "fmpad")},
    "up": {**_COMMON, "MVAU_p": ("main_proj.0", "mvau"), "Thr_p": ("main_proj.0", "thr"), "UpNN": ("upsample", "upsample"),
           "FMPadPix": ("up.0", "fmpadpix"), "FMPad_u": ("up.0", "fmpad"), "SWG_u": ("up.0", "swg"), "MVAU_u": ("up.0", "mvau"), "Thr_u": ("up.0", "thr"),
           "FMPad_k": ("skip_conv", "fmpad"), "SWG_k": ("skip_conv", "swg"), "MVAU_k": ("skip_conv", "mvau"), "Thr_k": ("skip_conv", "thr")},
    # bilinear decoder (LayerQuantEnetFINN `_nearest_depthwise_bilinear_kernel`, analytical up_bottleneck skip_dw=True): the main branch's windowed slot is the frozen depthwise 3x3 conv
    # `<stage>.main_up.1` (FMPadding -> depthwise SWG -> VVAU_hls); its threshold is the block's skip_quant
    "updw": {**_COMMON, "MVAU_p": ("main_proj.0", "mvau"), "Thr_p": ("main_proj.0", "thr"), "UpNN": ("upsample", "upsample"),
             "FMPadPix": ("up.0", "fmpadpix"), "FMPad_u": ("up.0", "fmpad"), "SWG_u": ("up.0", "swg"), "MVAU_u": ("up.0", "mvau"), "Thr_u": ("up.0", "thr"),
             "FMPad_k": ("main_up.1", "fmpad"), "SWG_k": ("main_up.1", "swg"), "MVAU_k": ("main_up.1", "mvau"), "Thr_k": ("skip_quant", "thr")},
    "init": {"Thr_in": ("input_quant", "thr"), "Dup": (None, "dup"), "FMPad": ("conv", "fmpad"), "SWG": ("conv", "swg"), "MVAU_c": ("conv", "mvau"),
             "Thr_c": ("conv", "thr"), "Thr_m": ("pool_quant", "thr"), "MaxPool": ("pool", "pool"), "SWG_p": ("pool", "swg"), "Pool": ("pool", "pool"),
             "Concat": ("concat", "concat"), "Thr_act": ("act", "thr")},
    "final": {"FMPadPix": (None, "fmpadpix"), "FMPad_u": (None, "fmpad"), "SWG_u": (None, "swg"), "MVAU_f": (None, "mvau"), "Bias": (None, "bias"),
              "LabelSelect": ("argmax", "argmax")},
}


def table_kind(kind: str, params: dict | None = None) -> str:
    """The name-table key of a block: an up block with the bilinear depthwise upsampler (params['skip_dw']) has its own table."""
    return "updw" if kind == "up" and params and params.get("skip_dw") else kind


def block_output_name(stage: str, kind: str) -> str:
    """MILP name of the node a block's output stream leaves from (the producer a following block's Dup is named after)."""
    return "initial.act" if kind == "init" else f"{stage}.out_act"


def milp_node(stage: str, kind: str, node: str, prev_output: str | None = None) -> tuple[str, str]:
    """(MILP name, part) of an analytical node. prev_output = block_output_name of the PREVIOUS block, needed for Dup (MILP names a dup after its producer)."""
    if node.startswith("DWC("):
        return f"{stage}.dwc", "dwc"
    if node not in _TABLE[kind]:
        return f"{stage}.{node}", "unmapped"                 # test_node_names asserts no real block node lands here
    leaf, part = _TABLE[kind][node]
    if node == "Dup":
        if kind == "init":                                    # the initial block forks its own input quantizer
            return f"{stage}.input_quant.dup", part
        return f"{prev_output}.dup" if prev_output else f"{stage}.dup", part
    if kind == "final" and leaf is None:
        return "final", part                                  # the final layer's MILP entry is `final`; argmax is its own extra node
    return f"{stage}.{leaf}", part


def milp_role(stage: str, kind: str, node: str, prev_output: str | None = None) -> str:
    """'<MILP name>#<part>', unique per hardware node: the name used in the ONNX picture and the FIFO lists' *_milp fields."""
    name, part = milp_node(stage, kind, node, prev_output)
    return f"{name}#{part}" if part != "dwc" else f"{name}#{node}"          # several DWCs per block: keep the edge so the role stays unique


def covered_nodes(kind: str) -> set[str]:
    return set(_TABLE[kind])

"""Native FINN resource-cost estimator -- calls FINN's OWN getCustomOp()
resource-estimation methods (MVAU_hls.lut_estimation/bram_estimation/
dsp_estimation/uram_estimation/get_exp_cycles) for the MVU/compute-engine
term of each Conv2d/ConvTranspose2d layer, instead of
compression/hawq/finn_cost_model.py's hand-derived closed-form
re-implementation of the same FINN-R 2018 paper formulae. Produces the
SAME output dict schema as finn_cost_model.layer_cost_pe_simd (plus
'dsp'/'wm_uram18' extra keys) so it's a drop-in backward-compatible
alternative cost backend for folding_ilp.py's --cost-backend native flag.

Confirmed via direct FINN source reads (this repo's earlier FINN-source
archaeology session):
- qonnx.custom_op.registry.getCustomOp() needs only a bare ONNX NodeProto
  (node.op_type/node.domain) -- no ModelWrapper, no graph, no initializer
  data -- so a synthetic MVAU_hls node can be built standalone, cheaply,
  in pure Python (no Vivado/HLS subprocess, no real model needed).
- MVAU_hls.lut_estimation/bram_estimation/uram_estimation/dsp_estimation/
  get_exp_cycles all read ONLY nodeattrs (PE, SIMD, MW, MH,
  inputDataType/weightDataType/outputDataType/accDataType, mem_mode,
  ram_style, resType, noActivation, numInputVectors) -- no dependency on
  actual tensor/weight values, ValueInfo, or graph connectivity.
- Real FINN's LUT formula is `c0 + c1*(P*(mult_luts + addertree_luts +
  acc_luts + thr_luts + comp_luts)) + c2`, MORE detailed than
  finn_cost_model.py's simplified `300 + 1.1*M*(P*Q)*(W*A)` (e.g. it
  separately accounts for the adder tree and accumulator width) -- this
  is the actual value of calling native FINN instead of the analytical
  approximation.

SCOPE / WHY THIS IS ONLY "PARTIALLY NATIVE":
- MVU/compute-engine (mvu_lut, wm_bram18, wm_uram18, dsp, cycles) IS the
  real getCustomOp(MVAU_hls) estimate -- the actual FINN resource-model
  code path, not a re-derived formula.
- SWU/line-buffer (swu_lut, swu_bram18) and MaxPool2d cost STILL come from
  finn_cost_model.py's analytical formulas, unchanged. Real FINN's
  ConvolutionInputGenerator_hls SIMD must (only) divide IFMChannels, a
  DIFFERENT constraint than MVAU's SIMD dividing cin*kh*kw -- these two
  SIMD values don't correspond 1:1 in general, so plugging real SWU
  estimation into the existing single-(PE,SIMD)-per-layer candidate grid
  isn't a clean drop-in. Deferred as future work (finn_cost_model.py's own
  docstring already notes SWU_BRAM doesn't depend on P/Q/folding at all --
  it's the least folding-search-relevant term anyway).
- No threshold/routing/interconnect resources modeled here either (same
  scope limit finn_cost_model.py documents) -- MVAU nodes are built with
  noActivation=1 (thresholds fused elsewhere / not modeled), matching the
  analytical model's own "no threshold memory" scope note.

Two ways to use this file:
  1. As a library (native_layer_cost_pe_simd / native_conv_cost_pe_simd) --
     needs `finn`/`qonnx` importable, i.e. this file must run INSIDE the
     FINN container (see folding_ilp.py's build_cost_table_native, which
     docker cp's this file + finn_cost_model.py in and docker execs it).
  2. As a batch CLI:
       python finn_native_cost_estimator.py request.json response.json
     Reads a {weight_bits, act_bits, fpga_part, layers, candidates}
     request, writes a {layer_name: {"pe,simd": cost_dict}} response.
     This is the actual invocation folding_ilp.py shells out to via
     docker exec.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finn_cost_model import (  # noqa: E402
    LayerGeometry,
    conv_cost_pe_simd as analytical_conv_cost_pe_simd,
    maxpool_cost,
)

FPGA_PART = "xczu7ev-ffvc1156-2-e"


def _mvau_node(
    mw: int, mh: int, pe: int, simd: int, weight_bits: int, act_bits: int, num_input_vectors: list[int],
    ram_style: str = "block",
):
    """Build a bare synthetic MVAU_hls node and wrap it with getCustomOp()
    -- no real graph/model needed, confirmed via direct FINN source read.
    Configured to match finn_cost_model.py's own scope/assumptions: M=1
    (numInputVectors=[hout*wout], no separate M axis -- FINN's own `mmv`
    is hardcoded to 1 everywhere anyway, confirmed this session), resType=
    "lut" (matches the analytical model, which has no DSP term -- note
    real FINN's lut/dsp estimation only branches on resType=="dsp", so
    "auto" would give byte-identical numbers here regardless), noActivation=1
    (no threshold memory modeled, matching finn_cost_model.py's documented
    scope note).

    ram_style="ultra" requests URAM for the weight tile instead of BRAM
    (real FINN's uram_estimation()/bram_estimation() are mutually exclusive
    on this nodeattr, see matrixvectoractivation.py) -- confirmed via source
    read that real FINN asserts runtime_writeable_weights=1 whenever
    ram_style=="ultra" ("Layer with URAM weights must have runtime_writeable_
    weights=1"), so that nodeattr is set alongside it here. Also confirmed
    neither lut_estimation() nor get_exp_cycles() reads ram_style or
    runtime_writeable_weights at all (only ram_style=="distributed" adds an
    extra LUT term, not "ultra") -- so this is a pure BRAM<->URAM resource
    swap, no LUT/cycle side effects in FINN's own cost model."""
    from onnx import helper
    from qonnx.custom_op.registry import getCustomOp

    node = helper.make_node(
        "MVAU_hls",
        inputs=["inp", "weights"],
        outputs=["outp"],
        domain="finn.custom_op.fpgadataflow.hls",
        PE=pe, SIMD=simd, MW=mw, MH=mh,
        resType="lut",
        inputDataType=f"INT{act_bits}",
        weightDataType=f"INT{weight_bits}",
        outputDataType="INT32",
        accDataType="INT32",
        binaryXnorMode=0,
        noActivation=1,
        numInputVectors=list(num_input_vectors),
        mem_mode="internal_decoupled",
        ram_style=ram_style,
        runtime_writeable_weights=1 if ram_style == "ultra" else 0,
    )
    return getCustomOp(node)


def native_conv_cost_pe_simd(
    layer: LayerGeometry, weight_bits: int, act_bits: int, pe: int, simd: int, fpga_part: str = FPGA_PART,
    ram_style: str = "block",
) -> dict:
    """Conv2d cost with the MVU term computed by REAL FINN
    (getCustomOp(MVAU_hls).lut_estimation/bram_estimation/...) and the SWU
    term from finn_cost_model.py's analytical formula (see module
    docstring for why SWU stays analytical)."""
    analytical = analytical_conv_cost_pe_simd(layer, weight_bits, act_bits, pe, simd)
    mw = layer.cin * layer.kh * layer.kw
    mh = layer.cout
    inst = _mvau_node(mw, mh, pe, simd, weight_bits, act_bits, [layer.hout * layer.wout], ram_style)
    mvu_lut = inst.lut_estimation()
    wm_bram18 = inst.bram_estimation()
    wm_uram18 = inst.uram_estimation()
    dsp = inst.dsp_estimation(fpga_part)
    cycles = inst.get_exp_cycles()
    swu_lut = analytical["swu_lut"]
    total_lut = swu_lut + mvu_lut
    return {
        "total_pe": analytical["total_pe"], "total_simd_lanes": analytical["total_simd_lanes"],
        "swu_bram18": analytical["swu_bram18"], "wm_bram18": wm_bram18, "wm_uram18": wm_uram18,
        "swu_lut": swu_lut, "mvu_lut": mvu_lut, "mp_lut": 0,
        "total_lut": total_lut, "cycles": cycles, "dsp": dsp,
    }


def native_layer_cost_pe_simd(
    layer: LayerGeometry, weight_bits: int, act_bits: int, pe: int, simd: int, fpga_part: str = FPGA_PART,
    ram_style: str = "block",
) -> dict:
    """Dispatcher mirroring finn_cost_model.layer_cost_pe_simd's exact
    op_type branching. MaxPool2d has no MVAU at all -- falls back entirely
    to the analytical model (nothing 'more native' exists to call for it;
    'dsp'/'wm_uram18' are added as 0 purely for schema consistency with
    the Conv2d/ConvTranspose2d branches)."""
    if layer.op_type == "Conv2d":
        return native_conv_cost_pe_simd(layer, weight_bits, act_bits, pe, simd, fpga_part, ram_style)
    if layer.op_type == "ConvTranspose2d":
        # Same zero-insertion equivalent-geometry trick finn_cost_model.py uses.
        n_eff_h = (layer.hin - 1) * layer.sh + 1 + 2 * (layer.kh - 1)
        n_eff_w = (layer.win - 1) * layer.sw + 1 + 2 * (layer.kw - 1)
        equivalent = LayerGeometry(
            op_type="Conv2d", name=layer.name, stage=layer.stage,
            cin=layer.cin, hin=n_eff_h, win=n_eff_w, cout=layer.cout, hout=layer.hout, wout=layer.wout,
            kh=layer.kh, kw=layer.kw, sh=1, sw=1, dh=1, dw=1,
        )
        return native_conv_cost_pe_simd(equivalent, weight_bits, act_bits, pe, simd, fpga_part, ram_style)
    if layer.op_type == "MaxPool2d":
        cost = maxpool_cost(layer, act_bits)
        return {**cost, "wm_uram18": 0, "dsp": 0}
    raise ValueError(f"Unknown op_type {layer.op_type!r} for layer {layer.name}")


def main(request_path: str, response_path: str) -> None:
    with open(request_path) as f:
        request = json.load(f)
    weight_bits = request["weight_bits"]
    act_bits = request["act_bits"]
    fpga_part = request.get("fpga_part", FPGA_PART)
    layers_by_name = {l["name"]: LayerGeometry(**l) for l in request["layers"]}

    response: dict[str, dict[str, dict]] = {}
    n_done = 0
    for name, candidates in request["candidates"].items():
        layer = layers_by_name[name]
        response[name] = {}
        for pe, simd, ram_style in candidates:
            cost = native_layer_cost_pe_simd(layer, weight_bits, act_bits, pe, simd, fpga_part, ram_style)
            response[name][f"{pe},{simd},{ram_style}"] = cost
            n_done += 1

    with open(response_path, "w") as f:
        json.dump(response, f)
    print(f"Wrote {n_done} native (layer, PE, SIMD) cost entries to {response_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

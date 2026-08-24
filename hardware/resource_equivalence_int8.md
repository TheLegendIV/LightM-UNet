# FINN resource equivalence at int8 (LUT↔DSP, URAM↔BRAM)

Derived directly from FINN v0.10.1's own cost-estimation source (read from
the `brave_lewin` container, 2026-08-21/22) -- not a general Xilinx claim,
specifically FINN's `MVAU_hls`/`ConvolutionInputGenerator_rtl` models at
W=8 (weights), A=8 (activations).

## LUT ↔ DSP (compute: MVAU multiply-accumulate)

Source: `finn/src/finn/custom_op/fpgadataflow/hls/matrixvectoractivation_hls.py`
(`MVAU_hls.lut_estimation()` / `.dsp_estimation()`). Only the *multiplier*
term differs between `resType="lut"` and `resType="dsp"` -- adder-tree,
accumulator, and threshold LUTs are identical either way.

- LUT path (`resType="lut"`/`"auto"` at this width): per PE×SIMD lane,
  `mult_luts = Q * (2*ceil((W+A)/6) - 1) * (W+A)`.
  At W=A=8: `(2*ceil(16/6)-1)*16 = (2*3-1)*16 = 80` raw LUTs/lane, scaled
  by FINN's 1.1 efficiency factor → **≈88 LUTs**.
- DSP path (`resType="dsp"`): `mult_dsp = P*Q*ceil((W+A)/48)`.
  At W=A=8: `ceil(16/48) = 1` **DSP** slice/lane.

**≈80-88 LUTs ≈ 1 DSP slice** for one int8×int8 MAC lane in FINN's model
(scales linearly with PE×SIMD, so the ratio is fold-independent). Matches
the observed swing in the `quantEnet_s19_single_block_int8` probe build
(`hardware/results.csv`): each MVAU's LUT count drops by roughly this
amount when `resType` is forced from `"auto"`→`"dsp"`, gaining exactly 1
DSP per node.

## URAM ↔ BRAM_18K (memory: MVAU weights, SWU line buffers)

Source: `finn/src/finn/custom_op/fpgadataflow/matrixvectoractivation.py`
(`MVAU.bram_estimation()`/`.uram_estimation()`) and
`finn/src/finn/custom_op/fpgadataflow/rtl/convolutioninputgenerator_rtl.py`
(same pattern for the SWU/ConvolutionInputGenerator line buffer -- confirmed
this DOES support `ram_style="ultra"`, contrary to an earlier, incorrect
assumption in this repo's notes).

Both estimators size against fixed physical primitives, independent of
data width:

| Primitive | Capacity |
|---|---|
| BRAM18K (`RAMB18E2`) | 18 Kibibit = 18,432 bits |
| URAM288 (`URAM288`, 4096×72) | 294,912 bits = 288 Kibibit |

$$\frac{294912}{18432} = 16$$

**1 URAM = 16× the raw bit capacity of 1 BRAM_18K block.** This is a fixed
silicon ratio (not int8-specific), but at int8 specifically:
- A BRAM_18K's native 18-bit-wide port packs 2 int8 values per row before
  needing width-multiplexing (`ceil(18/8)`≈2).
- A URAM's native 72-bit-wide port packs 9 int8 values per row
  (`ceil(72/8) = 9`) before needing width-multiplexing.

This repo's own ZU7EV budget bookkeeping already uses this ratio implicitly:
624 BRAM_18K-equivalent vs 96 URAM blocks (96 × 16 = 1536 BRAM_18K-equivalent
raw capacity in URAM form -- a separate physical resource pool, not
interchangeable at the primitive level, but directly comparable in bits).

## Where these numbers came from

- Grep target: `resType`, `ram_style`, `lut_estimation`, `dsp_estimation`,
  `bram_estimation`, `uram_estimation` in
  `finn/src/finn/custom_op/fpgadataflow/{matrixvectoractivation.py,
  hls/matrixvectoractivation_hls.py, rtl/convolutioninputgenerator_rtl.py}`
  inside the `brave_lewin` container.
- Cross-checked against the real per-node `estimate_layer_resources.json`
  from the `quantEnet_s19_single_block_int8` force-DSP/force-URAM probe
  build (`hardware/finn_enet_ip_build_s19_single_block.py`) -- see
  `hardware/results.csv` for the resulting LUT/DSP/URAM counts once that
  build's real Vivado OOC synthesis completes.

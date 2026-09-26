# Archived: pre-2026-09-17 cross-check ILP outputs

These 10 `layer_bits_folding_*`/`layer_bits_SITES_*` files are the last
git-committed versions (commit `a3f2a8d5bb`, "Sync", 2026-09-16 19:57 UTC+2)
of the S12-dense-warmstart150ep 5-alpha ILP sweep, saved here before the
current working-tree versions overwrote them in place.

The current (non-archived) versions were regenerated on 2026-09-17 after a
systematic FINN-source cross-check of `finn_cost_model.py` fixed several
real bugs found via fresh re-reads of the actual FINN source plus real
per-node Vivado calibration data:

- RTL MVAU DSP: was branching on bit-width (4 lanes/DSP for <=4-bit, 2 for
  >4-bit) with zero real evidence for the <=4-bit branch (every low-bit data
  point was at PE=1, where the branches are indistinguishable). Real
  per-node data showed a flat, unconditional `ceil(PE/2)*SIMD` -- exact on
  every node checked, at every bit-width.
- SWU (ConvolutionInputGenerator_rtl) LUT: was missing FINN's own
  `ram_luts` term entirely, and separately used the WRONG "parallel-style"
  buffer_depth formula (conflated with a different inline formula from
  bram_estimation(), not the one lut_estimation() actually calls).
- SWU node existence: FINN inserts NO ConvolutionInputGenerator_rtl node at
  all for 1x1-kernel convolutions (confirmed structurally, 38/38 vs 35/35
  real nodes) -- roughly half of every ENet bottleneck's layers were being
  charged a nonexistent SWU cost.
- Thresholding_rtl LUT/BRAM constants refit on 79 real nodes (up from 66).

See finn_cost_model.py's own module/function docstrings for full provenance
of each fix, and the session history for the real per-node calibration data
each was checked against.

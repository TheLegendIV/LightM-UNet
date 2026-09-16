# Archived: pre-crosscheck0917 S12 dense warmstart plots

These 2 plots (dated Sep 8) are from the ORIGINAL (un-suffixed) `ft15ep`
15-epoch/5-alpha QAT sweep for `12_dense_relu_warmstart150ep`, trained
against the per-layer bits/folding this repo's ILP produced BEFORE the
2026-09-17 FINN-source cross-check fixed several real cost-model bugs
(RTL MVAU DSP packing, missing/wrong SWU LUT terms, the 1x1-kernel-has-no-
SWU-node finding -- see compression/MILP/finn_cost_model.py's own
docstrings and compression/MILP/artifacts/12_dense_relu_warmstart150ep_
ILP_outputs_perlayer_forcedsp_lut70/archive_pre_20260917_swu_thr_dsp_
crosscheck/README.md).

The dice numbers themselves are still real (these checkpoints were really
trained), but the ILP-predicted latency they're plotted against (the
x-axis of the _vs_latency plot) reflects the OLD, now-superseded cost
model -- not comparable to the `_crosscheck0917`-suffixed plots generated
after the fix.

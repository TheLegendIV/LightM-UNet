# S12_dense_256_u4_analytical_v1

Whole-network folding for the S12 dense network (U4 widths (4, 16, 32, 16, 4), bottlenecks (4, 8, 8, 2, 1), plain nearest decoder, dense dilation pattern,
256x256 input, uniform INT6) assembled from the **analytical per-block models** (`MILP/analytical/net_fold.py`), written in the MILP flow's formats.
Not an ILP solve: the folds come from `bottleneck.py`, `dn_/up_/int_bottleneck.py` and `fnl_block.py`, costed with the MILP's own `layer_cost_pe_simd` /
`extra_node_options` (the MVAU cycles equal the MILP cost model's for every layer).

## Runs

| Folder | Targets | Result (model) |
|---|---|---|
| `int6_fps250_lat200/` | throughput 250 fps @ 100 MHz (every node <= 400,000 cycles/frame), latency to first output pixel <= 200 ms, downstream no slower than upstream, inter-block FIFOs fixed at depth 2 | 87.1k LUT (37.8%), 205 BRAM18 (32.9%), 0 URAM, 241 DSP (13.9%): nodes 76.6k LUT / 62 BRAM18, intra-block FIFOs 8.5k LUT / 143 BRAM18, DWCs 1.6k LUT, inter-block FIFOs 392 LUT; slowest node 147.7k cycles (677 fps); latency to first output pixel 9.9 ms |

Files per run (`final` tag, as in `S12_dense_256_fullwidth_joinsdist_v1/*/`):
`layer_bits_folding_final.json` (status / layer_weight_bits / layer_act_bits / per_layer / extra_nodes / _diagnostics / dataflow_graph, plus the extra top-level keys
`inter_block_fifos` and `intra_block_fifos`; the sizing record is `_diagnostics.fifo_sizing`), `layer_bits_SITES_final.json` (per-quantizer-site bits, builds `LayerQuantEnetFINN`),
`block_profile_final.csv`, `run_args.json`, and `enet_dataflow_final.onnx` (the whole sized design as one graph, open it in Netron: FINN op names, PE / SIMD / cycles / LUT /
BRAM per node, every DWC, every FIFO with depth / width / memory; intra-block FIFOs of depth <= 2 are not drawn because FINN removes them, inter-block FIFOs always are).

Regenerate: `python3 MILP/analytical/net_fold.py --bits 6 --fps 250 --clock-mhz 100 --max-latency-ms 200 --out-dir MILP/artifacts/S12_dense_256_u4_analytical_v1/int6_fps250_lat200`
(in `lightmunet_dev`; `--no-ratchet` drops the downstream-faster rule; with the 200 ms cap both settle on the same design because the legal (PE, SIMD) steps are coarse).

## Notes
* Latency is the time to the FIRST OUTPUT PIXEL: the sum over blocks of each block model's first-in -> first-out latency (`_diagnostics.latency_ms`, 9.9 ms; the chained
  whole-net simulation measured 6.7 ms, so the sum is a conservative bound). The MILP's "sum of every node's cycles" (169.5 ms, `sum_of_node_cycles_ms`, `total_cycles`) is kept for the
  MILP schema only: it is total work per frame, not a latency, because the blocks run concurrently. `--max-latency-ms 200` applies to the first-output latency and does not bind here.
* Downstream-faster rule: block k gets budget min(F, max(0.6 F, 1.04 x slowest foldable node of block k-1)). Achieved MILP DSR (cycles per output element,
  worst downstream / own) is 144 max, 16 median: element-rate matching needs the MILP solve itself.
* Inter-block FIFOs: each block owns the FIFO at its output and every one is FIXED at depth 2 (`--inter-fifo fixed`, `--inter-fifo-depth 2`; FINN's RemoveShallowFIFOs deletes such FIFOs, so the
  cost is ~0). An earlier pair-simulation search (`--inter-fifo sim`, `net_fifo.py`) found depth 2 sufficient at the 400,000 cycles/frame target for all 28 pairs, and one whole-net
  simulation with all 28 at depth 2 sustained 225,240 cycles/frame (target 400,000, no deadlock); the default run does not repeat that (`--whole-net-check` is opt-in).
  `--inter-fifo standard` gives the old one-BRAM18-per-edge rule (28 BRAM18). The memory that matters is inside the blocks: skip FIFOs and the prefetch FIFOs in front of
  every FMPadding (143 BRAM18 + 8.5k LUT in `intra_block_fifos`, depth > 2 only). See "FIFOs" below: all of it is a specification that no current build step applies.
* The initial block's branch-quant threshold sits UPSTREAM of its maxpool (`initial.pool_quant`, MILP kind `pool_quant`, added to the MILP itself on 2026-10-05 so the
  MILP and the analytical model agree: 65,536 cycles, 88 LUT at INT6). Not representable in the MILP schema: the InferPool route of the initial block (StreamingMaxPool is
  used) and the final layer's bias (FINN does not lower it; the no-bias layer is folded).
* Cross-check against the MILP solve itself on the same config (`finn_milp.py --candidate-bits 6 --force-dsp --min-resources --target-fps 250 --max-latency-ms 200`):
  92.4k LUT, 119 DSP, 339 fps, 19.97M cycles. Before net_fold priced every threshold like the analytical models (ram_style block) the assembly was within 0.4% of that LUT
  figure (92.7k); it is now 17% below it (76.6k). It spends more DSP (241) and runs faster (677 fps) because of the downstream-faster rule and the coarse (PE, SIMD) steps,
  where the MILP only has to meet the target.
* `LayerQuantENet.py` lost its import block, `VALID_CONTEXT_PATTERNS` and `ACT_SITE_TYPES` in commit 66e0590e8b (present at b16b35468a); `net_fold.py` injects them
  in-process to run `expand_layer_bits.py`. The file itself is untouched.

## FIFOs: handled manually, FINN autosizing must NOT run

FIFO depths are part of this design, not something to leave to FINN. Every FIFO in the built dataflow graph has to get its depth from this flow; FINN's own
FIFO sizing (`auto_fifo_depths=True`, the `largefifo_rtlsim` / `characterize` strategies, which simulate the stitched graph with a free-running source and
set depths from the occupancy they see) must not run for this network. The probe builds showed why: its depths are inflated by the source (whole-frame FIFOs
of 8,192 / 65,536 words) and unrelated to what the block needs, and they replace the verified depths the analytical sims computed.

How FINN takes manual depths: build with `auto_fifo_depths=False` and put the sizes in the folding config JSON (`folding_config_file`). `step_set_fifo_depths` then runs
`InsertFIFO`, `ApplyConfig(folding_config_file)` and `RemoveShallowFIFOs`. The file is keyed by FINN node name and holds folding and FIFO entries together, as in
`final_hw_config.json` / the U250 example (`U250_folding_config.json`):

    "MVAU_rtl_0":        {"PE": 1, "SIMD": 4, "ram_style": "block", "resType": "dsp", ...},
    "StreamingFIFO_rtl_3": {"ram_style": "auto", "depth": 2048, "impl_style": "vivado"},     # rtl for small depths, vivado above the SRL range
    "Thresholding_rtl_0": {"PE": 2, "depth_trigger_bram": 38000, ...}

What this means for this folder:
* `layer_bits_folding_final.json` carries the FOLDING (per_layer / extra_nodes) in the MILP schema; the S12 bridge (`build_partition_folding_config` in
  `hardware/finn_s12_build_steps.py`) turns that into the FINN folding config. It does not write FIFO entries and the S12 build runs the FINN autosizer, so as it stands
  `inter_block_fifos` / `intra_block_fifos` are ignored. Both have to change before a build: the bridge must add a `StreamingFIFO_*` entry (`depth`, `ram_style`, `impl_style`)
  for every FIFO listed there (the `mem` field says srl -> `rtl`, bram / uram -> `vivado` with `ram_style` block / ultra), and the build must run with `auto_fifo_depths=False`.
* Every FIFO needs an entry, not only the 28 inter-block ones. With autosizing off, a FIFO without an entry keeps the default depth 2. That is fine on a straight
  pipeline path, but the residual joins (skip FIFO into the `Add`, `FIFO main`, the next-frame prefetch FIFO in front of every `FMPadding`) need the deep values the
  analytical block sims compute; they are stored per block under `intra_block_fifos[<block>].fifos` (producer / consumer are the analytical node names, `Source` / `Sink` ends
  are the inter-block FIFOs). Width converters (`StreamingDataWidthConverter_*`) sit between the FIFOs and have their own `inFIFODepths` / `outFIFODepths` bookkeeping in the FINN config.
* FIFO node names (`StreamingFIFO_rtl_<n>`) exist only after `InsertFIFO` on the converted graph, so the entries have to be matched by producer / consumer node at build
  time (as `step_force_fifo_depths` does in `hardware/builds/bottleneck_probe_v1/finn_bottleneck_probe_build.py`), not written ahead of time.
* A forced depth only takes effect on an edge that exists in the landed graph, and the build must list the edges that found no entry ("edge not in prediction").
* The old standard inter-block FIFO (`--inter-fifo standard`) is 6 bit x 2048 in one BRAM18 (`vivado` impl, ram_style block). It is only a buffer between blocks: it does not replace the deep join FIFOs.

Status: the specification (`inter_block_fifos`, `intra_block_fifos`) exists and is verified in simulation; the bridge that merges it into the FINN folding config and the
`auto_fifo_depths=False` build are not written yet.

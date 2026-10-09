# S12_dense_256_u4_bilinear_analytical_v1

The bilinear-decoder counterpart of `S12_dense_256_u4_analytical_v2`: same network (U4 widths (4, 16, 32, 16, 4), bottlenecks (4, 8, 8, 2, 1), dense dilation, 256x256 input, uniform INT6) and **exactly the same
run conditions** (250 fps at 100 MHz, latency cap 200 ms, `--dsr-pct 4 --dsr-floor 0.6`, inter-block FIFOs fixed at depth 2, no MVAU width cap, 6 workers, the real row-buffer
`UpsampleNearestNeighbour` model). The only change is `--bilinear-up`: in each up block the main branch is the hardware substitute for `F.interpolate(bilinear)`. Analytical flow and block
simulation only, no MILP.

```
python3 MILP/analytical/net_fold.py --bilinear-up --bits 6 --fps 250 --clock-mhz 100 --max-latency-ms 200 --workers 6 --out-dir MILP/artifacts/S12_dense_256_u4_bilinear_analytical_v1/int6_fps250_lat200
```

## Where the bilinear implementation lives

`enet/nnunetv2/nets/LayerQuantEnetFINN.py`: `_nearest_depthwise_bilinear_kernel(channels)` builds `nn.Upsample(scale_factor=2, mode="nearest")` followed by a **frozen depthwise 3x3 `qnn.QuantConv2d`**
(INT8 `Int8WeightPerTensorFloat`, `groups=channels`, `requires_grad=False`) whose weights are the tent kernel `outer([1,2,1]/4, [1,2,1]/4)`. `FINNUpsamplingBottleneck` uses it as `main_up` when
`decoder_type="upsample_conv"` (the other two decoder types use a bare `nn.Upsample(nearest)`). It reproduces `F.interpolate(bilinear, align_corners=False)` exactly on interior pixels (~5e-7);
only the outermost 1 px ring differs (FINN has no edge-replicate padding). FINN lowers it to `UpsampleNearestNeighbour_hls -> FMPadding_rtl -> depthwise ConvolutionInputGenerator_rtl -> VVAU_hls`
(VVAU_rtl needs a Versal DSP58, so the zcu7ev always gets VVAU_hls). The verification script lives in `hardware/archive/pre_builds_refactor_20260926/12_dense_relu_warmstart150ep/testbench_bilinear_vs_nearest_depthwise_up4up5.py`.
The bridge already handles the slot (`depthwise_vvau_slot` in `hardware/finn_s12_build_steps.py`).

## What was added to the analytical flow

* `up_bottleneck.model_up_bottleneck(..., skip_conv=True, skip_dw=True)`: main branch `Dup -> MVAU_p -> Thr_p -> UpNN -> FMPad_k -> SWG_k (depthwise, SIMD = PE) -> VVAU_k (PE | C, SIMD | 9) -> Thr_k (= the block's skip_quant) -> FIFO main`.
  `bottleneck._search_vvau` picks (PE, SIMD) with the largest cycles within the budget (VVAU cycles = pixels x C/PE x 9/SIMD; SIMD > 1 = SWG parallel_window, a DWC narrows the 9-element word); weights are 8 bit (frozen tent kernel).
  Explicit folds (`explicit_folds({"skipdw": (pe, simd)})`) work for it.
* `up_bottleneck_sim`: the depthwise SWG / VVAU stream nodes in the block simulation (the VVAU consumes C/PE x 9/SIMD words per output pixel and emits C/PE), folding config (`swg_k.parallel_window = SIMD > 1`) and ONNX label `VVAU_hls`.
* `net_fold.py --bilinear-up`: traces the plain nearest config and inserts the layer `<up stage>.main_up.1` (depthwise 3x3, cin = cout = C, at output resolution, groups = C) between `upsample` and `skip_quant`;
  its MILP-format entry is priced with 8-bit weights and no own threshold (the threshold is `skip_quant`, whose input width is the VVAU accumulator). No SITES file (there is no ENet counterpart of the frozen layer).
* `node_names.py`: table `updw` (`<stage>.main_up.1` for FMPad_k / SWG_k / MVAU_k, `<stage>.skip_quant` for Thr_k). `net_fold._verify_task` now runs the full sizing schedule (8 tries) for every block kind.
* Tests: `TestUpBilinearDepthwise` (test_up_bottleneck.py), `TestBilinearUpNames` (test_node_names.py).

## Result against the nearest decoder (v2), same conditions

| | nearest (v2) | bilinear |
|---|---|---|
| node LUT | 76,918 | 78,962 (+2.0k: two VVAU + depthwise SWG + FMPad per up block, the `up.*.main_up.1` entries are 1,024 / 863 LUT each, 1 DSP each) |
| DSP | 241 | 243 |
| BRAM18 incl. all FIFOs | 207 | 208 |
| slowest node | 147,715 cycles (`regular5.0.conv`) | **196,608 cycles (`up4.main_up.1`, PE 1, SIMD 3)** = 509 fps |
| latency to first output pixel | 9.94 ms | 10.06 ms |
| `up4` skip FIFO / `FIFO main` (words) | 512 / 351 | **2,224 / 17** |
| `up5` skip FIFO / `FIFO main` (words) | 256 / 160 | **1,076 / 5** |
| extra FIFO | | `up4.UpNN->DWC->FMPad_k` 1,058 words, `up5` 522 (prefetch of the depthwise window) |

* The depthwise conv is the slowest node of the whole net because the search takes the slowest legal fold inside the block budget (240,000 cycles at floor 0.6): PE 1, SIMD 3 (196,608) is the largest cycle count the (PE | 16, SIMD | 9) lattice offers below it, the next one is SIMD 9 (65,536). The cost is one DSP.
* The two join FIFOs swap roles: with the windowed depthwise conv the **main** branch is the slow one (its window needs about a row of pixels), so the **ext-end skip FIFO** now holds the lead (about 4.3 input rows, 2,224 / 1,076 words) and `FIFO main` is small.
  `fifo_model.UP_SKIP_ROWS` (one row) was fitted to the nearest decoder and does not apply to this variant; the MILP does not model it.
* Not done: SITES, calibration / QAT (training job: `compression/slurm/stage_12_dense_relu_bilinear_upsample_256.job`), the hardware bridge run of this design, and a MILP counterpart.

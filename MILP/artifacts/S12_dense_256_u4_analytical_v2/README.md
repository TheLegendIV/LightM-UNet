# S12_dense_256_u4_analytical_v2

Same design, targets and settings as `S12_dense_256_u4_analytical_v1` (S12 dense network, U4 widths (4, 16, 32, 16, 4), 256x256 input, uniform INT6, 250 fps at 100 MHz,
latency cap 200 ms, --dsr-pct 4 % / --dsr-floor 0.6 (then called ratchet), inter-block FIFOs fixed at depth 2, 6 workers). The only change is the model of the nearest-neighbour upsampler in the
block simulation (`MILP/analytical/up_bottleneck_sim.py`, `UpNNNode`). Everything else is read from `net_fold.py` unchanged.

## Why v2

v1 sized the up-block join FIFOs with an upsampler model that emits two outputs per input pixel, immediately. The kernel in the build container does not do that.
FINN `v0.10.1-10-g39f0c9a6b` uses finn-hlslib `16e5847` (2023-05-24), whose `UpsampleNearestNeighbour` is a one-row-buffer loop with blocking reads: on an even output
row it reads input pixel `x` and writes `RowBuf[x / 2]` for `x < W`, so the first W outputs leave one per input pixel (output `x` waits for input pixel `x`, not `x / 2`),
then W more follow with no read, and the odd row replays the buffer. The transposed-conv branch needs only input pixel `x / 2` for output `x`, so with this kernel the **ext
branch leads the main branch** by up to about W pixels per row. In v1 the sim had the main branch leading, so the main FIFO got the lead (639 / 302 words) and the skip FIFO got
one pixel (17 / 5 words).

With the real kernel the v1 sizing deadlocks in the block simulation (up4 and up5, saturated input, first row of the first frame): the skip FIFO fills, `Dup` stalls on the ext side,
the upsampler waits at its blocking read for an input pixel `Dup` can no longer deliver, and `Add` waits for the upsampler. Main branch FIFOs empty and starved, ext branch FIFOs full.

## What changed (v2 against v1)

Folds (PE, SIMD of every layer and extra node), LUT, DSP, cycles and latency are identical. Four FIFOs change:

| FIFO | v1 | v2 |
|---|---|---|
| `up4.skip_FIFO` | 17 | 512 |
| `up4.FIFO_main` | 639 | 351 |
| `up5.skip_FIFO` | 5 | 256 |
| `up5.FIFO_main` | 302 | 160 |

Totals: intra-block FIFOs 145 BRAM18 (v1 143) and 8,998 LUT (v1 9,026); BRAM18 including all FIFOs 207 (v1 205); node LUT 76,918, DSP 241, slowest node 147,715 cycles (677 fps), latency to
first output pixel 9.94 ms, all unchanged. The new depths come from the same sizing procedure as before (`up_bottleneck.verify_with_sim`: measured need in a paced two-frame run plus one pixel).

Regenerate (in `lightmunet_dev`): `python3 MILP/analytical/net_fold.py --bits 6 --fps 250 --clock-mhz 100 --max-latency-ms 200 --workers 6 --out-dir MILP/artifacts/S12_dense_256_u4_analytical_v2/int6_fps250_lat200`.
Files per run are the same as v1 (`layer_bits_folding_final.json`, `layer_bits_SITES_final.json`, `block_profile_final.csv`, `run_args.json`, `enet_dataflow_final.onnx`) plus `build.log`.

## Caveats

* The kernel model is a line-by-line transcription of the hlslib loop where each iteration is all-or-nothing (a starved read or blocked write stalls the whole iteration). It is not validated
  against rtlsim of this block. On the hardware probe `up_cin32_cout16_in32_int4_noconv` it gives 21.45 cyc/px with every FIFO at 8 words against 20.53 measured; the earlier model gave 17.58.
* The join-PE rule in `up_bottleneck.py` (`_join_pe_for_upnn`, `d <= (t_in - c_p) / 2`) was derived from the old phase model and is unchanged. The block verification (sim reaches the target rate, no deadlock)
  passes with it for every up block here, so no PE changed, but the rule itself is not re-derived for the real kernel.
* Block simulations only. The FIFOs inside the transposed-conv branch stay at the minimum the simulation found (2 to 12 words); shrinking them further only costs throughput in the simulation,
  but the hardware probes showed that spare room in those FIFOs can hide a too-small skip FIFO, so a partition-level rtlsim is the real check.
* The ratchet-ablation analytical arms (`S12_dense_256_ratchet_ablation_v1/analytical_25pct*`, 25 % / floor 0.33) were not regenerated.

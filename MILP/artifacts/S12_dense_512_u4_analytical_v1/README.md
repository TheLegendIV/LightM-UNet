# S12_dense_512_u4_analytical_v1

`S12_dense_256_u4_analytical_v2` at **512x512** input: same network (S12 dense, U4 widths (4, 16, 32, 16, 4), bottlenecks (4, 8, 8, 2, 1), plain nearest decoder, dense dilation), same targets and settings
(uniform INT6, 250 fps at 100 MHz, latency cap 200 ms, `--dsr-pct 4 --dsr-floor 0.6`, inter-block FIFOs fixed at depth 2, no MVAU width cap, 6 workers, the real row-buffer
`UpsampleNearestNeighbour` model). The only change is the config (`config_12_dense_relu_nearest_upsample`, `INPUT_HW` 512). Analytical flow and block simulation only, no MILP.

```
python3 MILP/analytical/net_fold.py --config config_12_dense_relu_nearest_upsample --bits 6 --fps 250 --clock-mhz 100 --max-latency-ms 200 --workers 6 --out-dir MILP/artifacts/S12_dense_512_u4_analytical_v1/int6_fps250_lat200
```

Files per run as in the 256 artifacts: `layer_bits_folding_final.json`, `layer_bits_SITES_final.json` (85 weight / 110 activation sites), `block_profile_final.csv`, `enet_dataflow_final.onnx`, `run_args.json`, `run.log`.

## Cost of going from 256x256 to 512x512 (same targets, model estimates)

| | 256 (v2) | 512 | ratio |
|---|---|---|---|
| node LUT | 76,918 | 98,876 | 1.29 |
| FIFO + DWC LUT | 11,049 | 14,832 | 1.34 |
| **LUT incl. FIFO + DWC** | 87,967 | 113,709 | 1.29 |
| DSP | 241 | **540** | **2.24** |
| node BRAM18 (weights / SWG / thresholds) | 62 | 64 | 1.03 |
| FIFO BRAM18 (intra + inter block) | 145 | 241 | 1.66 |
| **BRAM18 incl. all FIFOs** | 207 | 305 | 1.47 |
| URAM | 0 | 0 | |
| slowest node | 147,715 cycles (`regular5.0.conv`), 677 fps | 327,680 cycles (`initial.pool`), 305 fps | |
| latency to first output pixel | 9.94 ms | 5.59 ms | 0.56 |
| FIFOs deeper than 2 | 64 | 230 | 3.6 |

* The frame has 4x the pixels at the same fps target, so every conv must process 4x the pixels per frame: DSP goes up 2.24x rather than 4x because the 512 blocks run slower per frame than the 256 ones
  (block budget 272,629 cycles against nodes at ~147k realised at 256: 4 / 1.85 = 2.2). Per group (DSP 256 -> 512): context stage2/3 128 -> 256, stage1 24 -> 80, decoder regular 15 -> 45, up 28 -> 56, down 19 -> 46, final 24 -> 48, initial 3 -> 9.
* Weight memory is unchanged (same parameters): node BRAM18 62 -> 64. The extra memory is FIFO: the skip FIFOs scale with the image width (dilation-8 / 16 blocks hold 34,144 words, 24 BRAM18 each) and the
  prefetch FIFOs in front of the padded convs scale with it too.
* The slowest node is the initial block's `StreamingMaxPool` (not foldable, 1.25 cycles per input pixel = 327,680 cycles at 512x512): the 512 network cannot run faster than 305 fps whatever the fold.
  250 fps is met with 22% margin. Latency to the first output pixel is lower than at 256 because the pixel period is 4x shorter at the same frame budget.

## Note on the generation

The first attempt failed in block `down1`: its nodes are all balanced at ~16 cycles/px against a 16.6 target, and the FIFO sizing schedule stopped at uniform depth 32 (18.9 cycles/px; 64 gives 16.0). The schedule
(`bottleneck._SIZING_SCHEDULE`) now has two more entries (uniform depth 64 and 128); only blocks that failed before reach them, so existing results are unchanged. `net_fold` / `net_explicit` run all 10 tries.

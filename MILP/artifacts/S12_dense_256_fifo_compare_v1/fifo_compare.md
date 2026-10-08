# FIFO sizes before simulation: analytical flow vs MILP

256x256 S12 dense nearest-upsample ReLU, uniform INT6, 250 fps (every node <= 400,000 cycles/frame), no MVAU width cap, default rate rule (--dsr-pct 4 --dsr-floor 0.6). ANALYTICAL = the block models' own closed-form estimates (no cycle simulation); MILP = `fifo_model.py` priced in the solve. Each flow keeps its own folding.

## Designs

| flow | DSP | slowest node (cycles) | fps |
|---|---|---|---|
| MILP | 145 | 147980 (down1.conv.0) | 676 |
| ANALYTICAL | 241 | 147715 (regular5.0.conv) | 677 |

## Totals by FIFO class (count / BRAM18 / LUT / URAM288 of the FIFO memory)

| class | MILP | ANALYTICAL |
|---|---|---|
| skip FIFO | 27 / 55 / 0 / 4 | 28 / 103 / 14 / 0 |
| FIFO main (up, initial) | 2 / 2 / 0 / 0 | 3 / 2 / 14 / 0 |
| window prefetch | 26 / 38 / 0 / 0 | 26 / 38 / 0 / 0 |
| DWCs | 105 / - / 2432 | 101 / - / 1659 |
| **FIFO memory + DWC LUT** | 95 BRAM18, 4 URAM, 2432 LUT | 143 BRAM18, 0 URAM, 1687 LUT |

## Skip FIFOs: depth in words (and pixels) at each flow's own folding

| block | MILP words (px, width, BRAM18) | ANALYTICAL words (px, width, BRAM18) | analytical px / MILP px | ANALYTICAL px at the MILP's width: BRAM18 |
|---|---|---|---|---|
| initial | - | 4 (1 px, 6 b, 0) | - | - |
| down1 | 1280 (80 px, 6 b, 1) | 1088 (68 px, 6 b, 1) | 0.85x | 1 |
| regular1.0 | 1152 (72 px, 6 b, 1) | 1120 (70 px, 6 b, 1) | 0.97x | 1 |
| regular1.1 | 1152 (72 px, 6 b, 1) | 1120 (70 px, 6 b, 1) | 0.97x | 1 |
| regular1.2 | 1152 (72 px, 6 b, 1) | 1120 (70 px, 6 b, 1) | 0.97x | 1 |
| regular1.3 | 1152 (72 px, 6 b, 1) | 1120 (70 px, 6 b, 1) | 0.97x | 1 |
| down2 | 1344 (42 px, 6 b, 1) | 1152 (36 px, 6 b, 1) | 0.86x | 1 |
| stage2.0 | 2336 (73 px, 6 b, 2) | 2208 (69 px, 6 b, 2) | 0.95x | 2 |
| stage2.1 | 4640 (145 px, 6 b, 3) | 4352 (136 px, 6 b, 3) | 0.94x | 3 |
| stage2.2 | 9216 (288 px, 6 b, 6) | 8640 (270 px, 6 b, 6) | 0.94x | 6 |
| stage2.3 | 2296 (574 px, 48 b, 0) | 17280 (540 px, 6 b, 12) | 0.94x | 0 |
| stage2.4 | 2336 (73 px, 6 b, 2) | 2208 (69 px, 6 b, 2) | 0.95x | 2 |
| stage2.5 | 4640 (145 px, 6 b, 3) | 4352 (136 px, 6 b, 3) | 0.94x | 3 |
| stage2.6 | 9216 (288 px, 6 b, 6) | 8640 (270 px, 6 b, 6) | 0.94x | 6 |
| stage2.7 | 2296 (574 px, 48 b, 0) | 17280 (540 px, 6 b, 12) | 0.94x | 0 |
| stage3.0 | 2336 (73 px, 6 b, 2) | 2208 (69 px, 6 b, 2) | 0.95x | 2 |
| stage3.1 | 4640 (145 px, 6 b, 3) | 4352 (136 px, 6 b, 3) | 0.94x | 3 |
| stage3.2 | 9216 (288 px, 6 b, 6) | 8640 (270 px, 6 b, 6) | 0.94x | 6 |
| stage3.3 | 2296 (574 px, 48 b, 0) | 17280 (540 px, 6 b, 12) | 0.94x | 0 |
| stage3.4 | 2336 (73 px, 6 b, 2) | 2208 (69 px, 6 b, 2) | 0.95x | 2 |
| stage3.5 | 4640 (145 px, 6 b, 3) | 4352 (136 px, 6 b, 3) | 0.94x | 3 |
| stage3.6 | 9216 (288 px, 6 b, 6) | 8640 (270 px, 6 b, 6) | 0.94x | 6 |
| stage3.7 | 2296 (574 px, 48 b, 0) | 17280 (540 px, 6 b, 12) | 0.94x | 0 |
| up4 | 512 (32 px, 6 b, 1) | 512 (32 px, 6 b, 1) | 1.00x | 1 |
| regular4.0 | 1152 (72 px, 6 b, 1) | 1120 (70 px, 6 b, 1) | 0.97x | 1 |
| regular4.1 | 1152 (72 px, 6 b, 1) | 1120 (70 px, 6 b, 1) | 0.97x | 1 |
| up5 | 256 (64 px, 6 b, 1) | 256 (64 px, 6 b, 1) | 1.00x | 1 |
| regular5.0 | 568 (142 px, 6 b, 1) | 564 (141 px, 6 b, 1) | 0.99x | 1 |

Skip FIFOs both flows price (27 blocks): MILP 55 BRAM18; ANALYTICAL 103 BRAM18 at its own PEs, 55 BRAM18 with its depth estimate at the MILP's PEs. (Last two differ by the PE choice: stream width -> BRAM aspect / pow2 rounding; MILP vs the last one differ by the depth estimate.)

## Up blocks: the two join FIFOs (words)

| block | FIFO | MILP closed form | ANALYTICAL pre-simulation | simulation (analytical_v2, real UpsampleNN kernel) |
|---|---|---|---|---|
| up4 | skip FIFO | 512 (6 b) | 512 | 512 |
| up4 | FIFO main | 1536 (6 b) | 1536 | 351 |
| up5 | skip FIFO | 256 (6 b) | 256 | 256 |
| up5 | FIFO main | 768 (6 b) | 768 | 160 |

(initial block `FIFO main`: analytical estimate 12 words, 3-4 words in simulation; not priced by the MILP.)

## Prefetch FIFOs

26 listed by the MILP, 26 by the analytical estimate; the same formula `(pad*W + pad + 1) * cf + 2` words, cf from each flow's own SWG SIMD; 2 of 26 have identical depth at the two foldings (BRAM18 38 vs 38).

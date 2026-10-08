# FIFO simulation of the MILP arms (`net_explicit.py`)

Each MILP folding was fed unchanged into the analytical block models and simulators (per block: saturated input, 3 frames, budget = the block's slowest MILP node; then the whole network chained with inter-block FIFOs at depth 2). FIFO depths are the smallest the simulation needs; a deadlocked chain grows the full FIFOs. `MILP` = the closed-form estimate in the MILP json (skip + prefetch FIFOs only, DWC LUT); `sim` = every FIFO of the simulation.


## ratchet_1pct

* blocks: 29; fail alone (cannot meet the network budget): **0** []; miss their own budget at any depth (window-fill gap, accepted if within the network budget): **9** (final +150.0%, stage2.2 +3.02%, stage2.3 +10.93%, stage2.6 +3.02%, stage2.7 +10.93%, stage3.2 +3.02%, stage3.3 +10.93%, stage3.6 +3.02%, stage3.7 +10.93%)
* whole-net chain (inter-block depth 2, target 327680 cycles/frame + 3%): first round deadlock = **False** (period 327899), rounds = 1, final ok = **True**, last period 327899

| class | FIFOs in sim | listed by MILP | MILP BRAM18 | sim BRAM18 | listed: median depth MILP -> sim | listed: depth up / down / same | not listed by MILP: sim depth > 2 | their BRAM18 |
|---|---|---|---|---|---|---|---|---|
| DWC side | 206 | 0 | 0 | 0 | - | 0 / 0 / 0 | 55 | 0 |
| dup output | 55 | 0 | 0 | 2 | - | 0 / 0 / 0 | 24 | 2 |
| other intra-block | 145 | 2 | 2 | 64 | 1152 -> 267 | 0 / 2 / 0 | 34 | 62 |
| skip FIFO | 28 | 27 | 55 | 56 | 2336 -> 2440 | 19 / 2 / 2 | 5 | 1 |
| window prefetch (in front of FMPad) | 31 | 26 | 38 | 26 | 538 -> 392 | 0 / 10 / 16 | 5 | 2 |

largest simulated FIFOs: stage2.3 3.fmpad->3.swg None->17646 (12 BRAM18); stage2.7 7.fmpad->7.swg None->17646 (12 BRAM18); stage3.3 3.fmpad->3.swg None->17646 (12 BRAM18); stage3.7 7.fmpad->7.swg None->17646 (12 BRAM18); stage2.2 2.thr_s->2.add 9216->9414 (6 BRAM18); stage2.6 6.thr_s->6.add 9216->9414 (6 BRAM18); stage3.2 2.thr_s->2.add 9216->9414 (6 BRAM18); stage3.6 6.thr_s->6.add 9216->9414 (6 BRAM18)

* inter-block FIFOs: 28 at depth [2] (MILP) -> depths [2] (sim), 0 BRAM18
* intra-block FIFO + DWC totals: MILP estimate 95.0 BRAM18 / 2905 LUT (skip + prefetch + DWC + inter) -> sim 148 BRAM18 / 13025 LUT; design total BRAM18 105.0 -> 158.0, LUT 99161 -> 109281


## ratchet_25pct

* blocks: 29; fail alone (cannot meet the network budget): **0** []; miss their own budget at any depth (window-fill gap, accepted if within the network budget): **9** (final +150.0%, stage2.2 +4.69%, stage2.3 +14.42%, stage2.6 +4.69%, stage2.7 +14.42%, stage3.2 +4.69%, stage3.3 +14.42%, stage3.6 +4.69%, stage3.7 +14.42%)
* whole-net chain (inter-block depth 2, target 327680 cycles/frame + 3%): first round deadlock = **False** (period 327899), rounds = 1, final ok = **True**, last period 327899

| class | FIFOs in sim | listed by MILP | MILP BRAM18 | sim BRAM18 | listed: median depth MILP -> sim | listed: depth up / down / same | not listed by MILP: sim depth > 2 | their BRAM18 |
|---|---|---|---|---|---|---|---|---|
| DWC side | 199 | 0 | 0 | 0 | - | 0 / 0 / 0 | 30 | 0 |
| dup output | 55 | 0 | 0 | 2 | - | 0 / 0 / 0 | 23 | 2 |
| other intra-block | 145 | 2 | 2 | 63 | 1152 -> 267 | 0 / 2 / 0 | 28 | 61 |
| skip FIFO | 28 | 27 | 55 | 56 | 2336 -> 2480 | 21 / 0 / 2 | 5 | 1 |
| window prefetch (in front of FMPad) | 31 | 26 | 38 | 28 | 538 -> 344 | 0 / 9 / 17 | 5 | 2 |

largest simulated FIFOs: stage2.3 3.fmpad->3.swg None->18750 (12 BRAM18); stage2.7 7.fmpad->7.swg None->18750 (12 BRAM18); stage3.3 3.fmpad->3.swg None->18750 (12 BRAM18); stage3.7 7.fmpad->7.swg None->18750 (12 BRAM18); stage2.2 2.thr_s->2.add 9216->10570 (6 BRAM18); stage2.6 6.thr_s->6.add 9216->10570 (6 BRAM18); stage3.2 2.thr_s->2.add 9216->10570 (6 BRAM18); stage3.6 6.thr_s->6.add 9216->10570 (6 BRAM18)

* inter-block FIFOs: 28 at depth [2] (MILP) -> depths [2] (sim), 0 BRAM18
* intra-block FIFO + DWC totals: MILP estimate 95.0 BRAM18 / 2841 LUT (skip + prefetch + DWC + inter) -> sim 149 BRAM18 / 13411 LUT; design total BRAM18 105.0 -> 159.0, LUT 98466 -> 109036


## ratchet_100pct

* blocks: 29; fail alone (cannot meet the network budget): **0** []; miss their own budget at any depth (window-fill gap, accepted if within the network budget): **5** (final +150.0%, stage2.3 +5.65%, stage2.7 +5.65%, stage3.3 +5.65%, stage3.7 +5.65%)
* whole-net chain (inter-block depth 2, target 327680 cycles/frame + 3%): first round deadlock = **False** (period 362022), rounds = 1, final ok = **True**, last period 362022

| class | FIFOs in sim | listed by MILP | MILP BRAM18 | sim BRAM18 | listed: median depth MILP -> sim | listed: depth up / down / same | not listed by MILP: sim depth > 2 | their BRAM18 |
|---|---|---|---|---|---|---|---|---|
| DWC side | 94 | 0 | 0 | 0 | - | 0 / 0 / 0 | 15 | 0 |
| dup output | 55 | 0 | 0 | 1 | - | 0 / 0 / 0 | 14 | 1 |
| other intra-block | 197 | 2 | 2 | 50 | 1152 -> 558 | 0 / 2 / 0 | 33 | 48 |
| skip FIFO | 28 | 27 | 55 | 56 | 2336 -> 2339 | 16 / 5 / 2 | 5 | 1 |
| window prefetch (in front of FMPad) | 31 | 26 | 38 | 82 | 538 -> 538 | 4 / 0 / 22 | 5 | 4 |

largest simulated FIFOs: stage2.2 2.fmpad->2.swg None->24592 (12 BRAM18); stage2.6 6.fmpad->6.swg None->24592 (12 BRAM18); stage3.2 2.fmpad->2.swg None->24592 (12 BRAM18); stage3.6 6.fmpad->6.swg None->24592 (12 BRAM18); stage2.2 2.thr_r->2.fmpad 2122->16976 (12 BRAM18); stage2.6 6.thr_r->6.fmpad 2122->16976 (12 BRAM18); stage3.2 2.thr_r->2.fmpad 2122->16976 (12 BRAM18); stage3.6 6.thr_r->6.fmpad 2122->16976 (12 BRAM18)

* inter-block FIFOs: 28 at depth [2] (MILP) -> depths [2] (sim), 0 BRAM18
* intra-block FIFO + DWC totals: MILP estimate 95.0 BRAM18 / 2004 LUT (skip + prefetch + DWC + inter) -> sim 189 BRAM18 / 9337 LUT; design total BRAM18 105.0 -> 199.0, LUT 95913 -> 103246


## ratchet_200pct

* blocks: 29; fail alone (cannot meet the network budget): **0** []; miss their own budget at any depth (window-fill gap, accepted if within the network budget): **4** (stage2.3 +5.65%, stage2.7 +5.65%, stage3.3 +5.65%, stage3.7 +5.65%)
* whole-net chain (inter-block depth 2, target 327680 cycles/frame + 3%): first round deadlock = **False** (period 362022), rounds = 1, final ok = **True**, last period 362022

| class | FIFOs in sim | listed by MILP | MILP BRAM18 | sim BRAM18 | listed: median depth MILP -> sim | listed: depth up / down / same | not listed by MILP: sim depth > 2 | their BRAM18 |
|---|---|---|---|---|---|---|---|---|
| DWC side | 92 | 0 | 0 | 0 | - | 0 / 0 / 0 | 15 | 0 |
| dup output | 55 | 0 | 0 | 1 | - | 0 / 0 / 0 | 14 | 1 |
| other intra-block | 198 | 2 | 2 | 50 | 1152 -> 558 | 0 / 2 / 0 | 33 | 48 |
| skip FIFO | 28 | 27 | 55 | 56 | 2336 -> 2339 | 16 / 5 / 2 | 5 | 1 |
| window prefetch (in front of FMPad) | 31 | 26 | 38 | 82 | 538 -> 538 | 4 / 0 / 22 | 5 | 4 |

largest simulated FIFOs: stage2.2 2.fmpad->2.swg None->24592 (12 BRAM18); stage2.6 6.fmpad->6.swg None->24592 (12 BRAM18); stage3.2 2.fmpad->2.swg None->24592 (12 BRAM18); stage3.6 6.fmpad->6.swg None->24592 (12 BRAM18); stage2.2 2.thr_r->2.fmpad 2122->16976 (12 BRAM18); stage2.6 6.thr_r->6.fmpad 2122->16976 (12 BRAM18); stage3.2 2.thr_r->2.fmpad 2122->16976 (12 BRAM18); stage3.6 6.thr_r->6.fmpad 2122->16976 (12 BRAM18)

* inter-block FIFOs: 28 at depth [2] (MILP) -> depths [2] (sim), 0 BRAM18
* intra-block FIFO + DWC totals: MILP estimate 95.0 BRAM18 / 1930 LUT (skip + prefetch + DWC + inter) -> sim 189 BRAM18 / 9159 LUT; design total BRAM18 105.0 -> 199.0, LUT 95558 -> 102787


## ratchet_off

* blocks: 29; fail alone (cannot meet the network budget): **0** []; miss their own budget at any depth (window-fill gap, accepted if within the network budget): **4** (stage2.3 +5.65%, stage2.7 +5.65%, stage3.3 +5.65%, stage3.7 +5.65%)
* whole-net chain (inter-block depth 2, target 327680 cycles/frame + 3%): first round deadlock = **False** (period 362022), rounds = 1, final ok = **True**, last period 362022

| class | FIFOs in sim | listed by MILP | MILP BRAM18 | sim BRAM18 | listed: median depth MILP -> sim | listed: depth up / down / same | not listed by MILP: sim depth > 2 | their BRAM18 |
|---|---|---|---|---|---|---|---|---|
| DWC side | 90 | 0 | 0 | 0 | - | 0 / 0 / 0 | 15 | 0 |
| dup output | 55 | 0 | 0 | 1 | - | 0 / 0 / 0 | 14 | 1 |
| other intra-block | 199 | 2 | 2 | 50 | 1152 -> 558 | 0 / 2 / 0 | 33 | 48 |
| skip FIFO | 28 | 27 | 55 | 56 | 2336 -> 2339 | 16 / 5 / 2 | 5 | 1 |
| window prefetch (in front of FMPad) | 31 | 26 | 38 | 82 | 538 -> 538 | 4 / 0 / 22 | 5 | 4 |

largest simulated FIFOs: stage2.2 2.fmpad->2.swg None->24592 (12 BRAM18); stage2.6 6.fmpad->6.swg None->24592 (12 BRAM18); stage3.2 2.fmpad->2.swg None->24592 (12 BRAM18); stage3.6 6.fmpad->6.swg None->24592 (12 BRAM18); stage2.2 2.thr_r->2.fmpad 2122->16976 (12 BRAM18); stage2.6 6.thr_r->6.fmpad 2122->16976 (12 BRAM18); stage3.2 2.thr_r->2.fmpad 2122->16976 (12 BRAM18); stage3.6 6.thr_r->6.fmpad 2122->16976 (12 BRAM18)

* inter-block FIFOs: 28 at depth [2] (MILP) -> depths [2] (sim), 0 BRAM18
* intra-block FIFO + DWC totals: MILP estimate 95.0 BRAM18 / 1887 LUT (skip + prefetch + DWC + inter) -> sim 189 BRAM18 / 8988 LUT; design total BRAM18 105.0 -> 199.0, LUT 95274 -> 102375

Rate = Cycles per Output Pixel (cyc/px)

Lower cyc/px = faster. "Rate" in this file always means cyc/px, so a node is
faster when its number is smaller.

Reference pixel: one output pixel of the bottleneck (H x W of the regular
bottleneck, all three convs run at this resolution). Strided / downsampling /
upsampling blocks change the pixel count per node; each node's cycles must then
be counted per bottleneck output pixel, or compared as cycles per frame.

For an ENet bottleneck

Main branch:
1x1 --> 3x3  --> 1x1
Cin --> Cmid --> Cout
Cmid = Cin/v
Cout = Cmid*z
(regular bottleneck: Cin = Cout, so v = z)

Expanded:
MVAU_r --> Thresh_r --> FMPad --> SWG_m --> MVAU_m --> Thresh_m --> MVAU_e --> Thresh_e

Skip branch:
Identity with requant thresholding
Thresh_s

Thresh_e and Thresh_s are the two operands' input quantizer of the residual add
(one shared quantizer in LayerQuantEnetFINN, so same bits and scale on both
operands). The add's own output quantizer is Thresh_out.

Complete Bottleneck

Dup --> MVAU_r --> Thresh_r --> FMPad --> SWG_m --> MVAU_m --> Thresh_m --> MVAU_e --> Thresh_e --> Add --> Thresh_out
  |                                                                                                   |
  --------------------------------------------> Thresh_s ---------------------------------------------


Goal:
To define a fixed architecture for this repeating bottleneck that is best for BRAM by balancing the rates across its nodes.


Notation

PE_x, SIMD_x   folding of node x (x in r, m, e)
PE_tx          PE of the thresholding after node x
P_x = PE_x * SIMD_x        MACs per cycle of MVAU x
K = 3, d = dilation
H_pad = H + 2d, W_pad = W + 2d
alpha = (H_pad * W_pad) / (H * W)          padding overhead, > 1
T = the bottleneck's target cyc/px (max over all nodes)


Per-node cycles per output pixel

MVAU (generic, MW x MH weight matrix):
  SF = MW / SIMD,  NF = MH / PE
  cyc/px = NF * SF = MW * MH / (PE * SIMD)

  MVAU_r  (MW = Cin,       MH = Cmid):  T_r = Cin  * Cmid / P_r = v * Cmid^2 / P_r
  MVAU_m  (MW = K^2 * Cmid, MH = Cmid): T_m = K^2 * Cmid^2 / P_m = 9 * Cmid^2 / P_m
  MVAU_e  (MW = Cmid,      MH = Cout):  T_e = Cmid * Cout / P_e = z * Cmid^2 / P_e

  In terms of Cout (Cmid = Cout/z, Cin = v*Cout/z):
    T_r = v * Cout^2 / z^2 / P_r
    T_m = 9 * Cout^2 / z^2 / P_m
    T_e =     Cout^2 / z   / P_e

  The MVAU accepts input only during the first SF cycles of its NF * SF cycles
  per pixel (it buffers the SF input words and replays them NF times). It emits one
  PE-wide word every SF cycles.

Thresholding after an MVAU (C = that MVAU's MH):
  cyc/px = C / PE_t
  Thresh_r: Cmid / PE_tr      Thresh_m: Cmid / PE_tm
  Thresh_e: Cout / PE_te      Thresh_s: Cout / PE_ts     Thresh_out: Cout / PE_tb

FMPad (before the 3x3):
  It emits the padded image: (H_pad * W_pad) pixels of Cmid / SIMD_p words.
  cyc/px = alpha * Cmid / SIMD_p
  with SIMD_p = SIMD_m (otherwise a width converter is inserted)

SWG_m (SIMD_swg = SIMD_m, stride 1):
  Output: K^2 * Cmid / SIMD_m = SF_m words per pixel, one word per cycle.
  Input:  alpha * Cmid / SIMD_m words per pixel.
  cyc/px = max(output, input) = K^2 * Cmid / SIMD_m = SF_m     (since K^2 > alpha)
  A word is SIMD channels of one kernel position. The window shifts once per
  patch (every SF_m words), not once per word.
  Each stored pixel is read up to K^2 times; the SWG is idle (NF_m - 1) / NF_m of the time.

Dup (DuplicateStreams): cyc/px = Cin / PE_d, with PE_d = PE of the incoming stream.

Add (AddStreams): cyc/px = Cout / PE_add


Rate-balancing rules

1. Reduce = middle = expand rate:
     T_r = T_m = T_e = T
   Substituting the MVAU equations and cancelling Cmid^2:
     v / P_r = 9 / P_m = z / P_e = 1 / (T / Cmid^2)
   so the MAC parallelism ratio is
     P_r : P_m : P_e = v : 9 : z
   For v = z = 4: 4 : 9 : 4.

2. Thresholding must not be slower than its MVAU (otherwise it backpressures the MVAU):
     C / PE_t <= NF * SF
   equivalently
     PE_t >= PE / SF
   Rule used: take C / PE_t = NF * SF when PE_t = PE / SF is an integer divisor
   of C; otherwise the smallest valid PE_t satisfying the inequality.
   The same applies to Thresh_e, Thresh_s, Thresh_out with T in place of NF * SF:
     Cout / PE_te <= T,   Cout / PE_ts <= T,   Cout / PE_tb <= T

3. FMPad, SWG, Dup, Add must not bind:
     alpha * Cmid / SIMD_m <= T_m         (equivalently PE_m <= K^2 * Cmid / alpha)
     K^2 * Cmid / SIMD_m <= T_m           (always true: NF_m >= 1)
     Cin / PE_d <= T
     Cout / PE_add <= T

4. Stream-width mismatches (policy: rate matching has priority, DWCs are accepted).
   A mismatch between adjacent nodes inserts a StreamingDataWidthConverter (DWC).
   Rule 2 (tight PE_t = PE / SF) is applied everywhere, so DWCs sit wherever
   adjacent stream widths differ. BRAM is the scarcest resource (buffers / FIFOs),
   and a DWC costs LUTs only, so LUTs are traded for BRAM.

   Still hard equalities (no DWC possible or wanted):
     SIMD_p = SIMD_swg = SIMD_m            (FMPad and SWG are folded into the conv, no DWC between them)
     PE_add = PE_te = PE_ts                (Add operands must have equal PE; with rule 2 and
                                            T_e = T this holds automatically: PE_te = PE_ts = Cout / T)

   Where DWCs appear with rule 2 (edge: width_in --> width_out):
     MVAU_r --> Thresh_r      PE_r * acc_bits       --> PE_tr * acc_bits     (accumulator-wide, costly)
     Thresh_r --> FMPad       PE_tr * act_bits      --> SIMD_m * act_bits
     MVAU_m --> Thresh_m      PE_m * acc_bits       --> PE_tm * acc_bits     (accumulator-wide, costly)
     Thresh_m --> MVAU_e      PE_tm * act_bits      --> SIMD_e * act_bits
     MVAU_e --> Thresh_e      PE_e * acc_bits       --> PE_te * acc_bits     (accumulator-wide, costly)
     Dup --> MVAU_r           PE_d * act_bits       --> SIMD_r * act_bits
   A DWC on an accumulator-wide edge (before thresholding) is much wider than one
   after thresholding. If a free PE/SIMD split exists for the same P_x = PE_x * SIMD_x,
   choose it to make an edge match (e.g. SIMD_r = PE_d, SIMD_e = PE_tm) and avoid that DWC.

   DWC rate condition on every converter edge (my inference, not verified in the repo):
     cyc/px_DWC ~ C / min(PE_in, PE_out) <= T
   Keep the two PEs integer multiples of each other (e.g. powers of two) so the
   cost stays at max(width) and not the lcm (see DWC cost below).

   Divisibility: SIMD_m | Cmid (the SWG never splits a word across kernel
   positions), SIMD_r | Cin, SIMD_e | Cmid, PE_r | Cmid, PE_m | Cmid, PE_e | Cout.


DWC cost (finn_cost_model.py dwc_cost, LUT only; BRAM/URAM/DSP = 0)

  in_width  = PE_in  * bits,   out_width = PE_out * bits
  intw      = lcm(in_width, out_width)
  raw_lut   = ceil(log2(intw/in_width)) + ceil(log2(intw/out_width))
              + (intw if in_width != intw) + (out_width if out_width != intw)
  LUT       = 0.8697 * raw_lut
  Integer width ratio: intw = max(in_width, out_width), so LUT ~ max(width) (plus a log term).
  Non-integer ratio: intw is the lcm and can be much larger.
  Caveat: per-node fit is weak (R^2 ~ 0.15); use only for total-budget estimates.
  DWC cycles are not modeled in the cost model.


Frame time

  cycles per frame = T * H * W + latency
  latency ~ pipeline fill of all nodes; dominated by the SWG fill of about
  (K - 1) rows plus K pixels of the padded image.


Memory equations

Weight memory per MVAU (BRAM / LUTRAM):
  depth = NF * SF = MW * MH / (PE * SIMD),   width = PE * SIMD * weight_bits

Threshold memory per thresholding node:
  per PE: depth = C / PE_t,  width = (2^act_bits - 1) * acc_bits
  total bits = C * (2^act_bits - 1) * acc_bits   (independent of PE_t)

SWG line buffer:
  depth = (K - 1) * W_pad * Cmid / SIMD_m + K * Cmid / SIMD_m,  width = SIMD_m * act_bits

Skip FIFO (after Thresh_s): must hold what arrives while the main branch fills.
  main-branch latency in pixels ~ (K - 1) * W_pad + K
  depth ~ ((K - 1) * W_pad + K) * Cout / PE_ts   words,   width = PE_ts * act_bits
  (estimate; add margin for the other nodes' pipeline depth)


FIFOs

FIFOs must only handle transients, not rate mismatches. The balance conditions
above are necessary but not sufficient: bursts (padding rows, SWG fill) stall the
upstream node if the FIFOs are too shallow, which drops the effective rate below T.
Only real buffer needed is on the skip branch after the threshold since it needs to
absorb the latency of the main branch (formula above). It sits after the
requantizing threshold because the stream there is narrow.


Simulation findings (bottleneck_sim.py) -- FIFO sizing on top of the rate balance

The rate balance is necessary, not sufficient. With a depth-2 FIFO on every edge the reference block (3x3, d=8, T=72) runs at
~80 cyc/px instead of 72 because:
1. At each output row start the SWG needs pad+1 new real pixels at once (flat windows at the row end let the upstream run ahead,
   then the next row jumps). The FIFO feeding FMPadding must hold >= pad+1 pixels, and the upstream must be faster than T to refill it.
2. MVAU_r reads its SF input words in a burst and then idles NF*SF-SF cycles; its feed FIFO needs more than 2 words.
3. The skip FIFO must hold what the skip branch delivers during the main-branch latency (dominated by the SWG fill, ~(k_eff-1)*W_pad+k_eff pixels).
With those three sized (see MILP/analytical/bottleneck.py verify_with_sim) and every other edge at 2 words, the simulated block holds 72.00 cyc/px.
Depth 1 would be a combinational ready path; real FINN FIFOs are >= 2 (skid buffer).
Hardware check of this claim: hardware/builds/bottleneck_probe_v1/ (see FINN_AGENT_HANDOFF.md).


Downsampling bottleneck (dn_bottleneck.py, dn_bottleneck_sim.py, test_dn_bottleneck.py)

    main:  Dup -> SWG_r -> MVAU_r (2x2, stride 2) -> Thr_r -> FMPad -> SWG_m -> MVAU_m (3x3, d=1) -> Thr_m -> MVAU_e -> Thr_e -+
    skip:  Dup -> MaxPool -> Thr_s -> [skip FIFO] -> FMPad_c (channel pad Cin -> Cout) ------------------------------------------+-> Add -> Thr_out

Budget. Two pixel domains: Dup, MaxPool and SWG_r see the H x W input, everything after the stride the (H/2) x (W/2) output.
Balance on FRAME cycles: every node needs  pixels_node * cyc_per_pixel <= F, with T_out = F / (H/2 * W/2) and T_in = F / (H*W).
Reported cyc_px is per OUTPUT pixel (= frame cycles / (H/2*W/2)), so util = cyc_px / T_out = frame cycles / F.

Work per output pixel (MACs): reduce 4*Cin*Cmid = Cin*Cout, 3x3 9*Cmid^2, expand Cmid*Cout, identity skip MVAU Cin*Cout.

MaxPool: StreamingMaxPool has no PE: one input pixel (all channels) per cycle plus one cycle per output, ~1.25*H*W cycles per frame.
It is a floor on F (rejected below it), not a rate problem otherwise, and it needs a Cin*A bit wide input stream (DWC from Dup).

Skip padding. The original export pads the skip with a padded-identity 1x1 MVAU (skip_pad="mvau", weights frozen at INT8): Cin*Cout
MACs per pixel, of which Cin are useful. Default (skip_pad="fmpad") pads the channels with FMPadding on a regrouped stream:
    [N, H, W, C]  ->  [N, H*W, C/s, s]   (no data moves; channels fastest)
    FMPadding: ImgDim = [H*W, C/s], NumChannels = s, SIMD = s, Padding = [0, 0, 0, (Cout-Cin)/s]
s (knob pad_group) must divide Cin and Cout-Cin; default max = gcd(Cin, Cout-Cin); cost Cout/s cycles per output pixel; stream width s*A bits.
skip_order="thr_pad" (default) thresholds on Cin channels and pads afterwards (smaller threshold, narrower/shorter skip FIFO);
"pad_thr" pads first and thresholds Cout channels. Untested in FINN: needs a custom pass that builds this FMPadding node and keeps the
real tensor shapes (see the handoff notes); FMPadding LUT is not calibrated in finn_cost_model (priced 0).

FIFOs. Only the skip branch (the short-latency one) gets a deep FIFO, narrow and deep after Thr_s (PE_ts words per pixel).
Everything else stays at 2 to 4 words (the sim's verify_with_sim doubles a uniform depth until T_out is reached, then shrinks every
FIFO to its observed occupancy). Reference down2-like block (16->32, 64x64 in, T_out=72): all main FIFOs 2-4 words, skip FIFO ~736 words x 4 bit.
Caveat found in the sim: a deep FIFO anywhere on the skip path (e.g. 64 words after the maxpool = 64 output pixels, one wide word each)
hides an undersized skip FIFO; the deadlock test therefore runs with depth-2 ordinary FIFOs.
Latency estimate: first strided window after (W+2)*T_in, MVAU_r, then the 3x3 fill (W/2+2)*T_out, then MVAU_m/Thr/MVAU_e; sim agrees to ~3%.


Upsampling bottleneck (up_bottleneck.py, up_bottleneck_sim.py, test_up_bottleneck.py), nearest-neighbour decoder

    main:  Dup -> MVAU_p (1x1) -> Thr_p -> UpNN (x2 nearest) -> [FMPad_k -> SWG_k -> MVAU_k (3x3) -> Thr_k] -> FIFO main -+
    ext:   Dup -> MVAU_r (1x1) -> Thr_r -> FMPadPix -> SWG_u (2x2) -> MVAU_u -> Thr_u -> MVAU_e (1x1) -> Thr_e -> skip FIFO -+-> Add -> Thr_out

Input H x W, output 2H x 2W, Cmid = Cin / v. skip_conv=True is decoder_type nearest_conv_upsample (the 3x3 skip_resize_conv is present), False is nearest_upsample.
Same frame-cycle budget as the downsampling block: Dup, MVAU_p, MVAU_r (+ thresholds) see the H*W input pixels, UpNN and everything after the 2H*2W output pixels;
T_out = F / (4*H*W), T_in = F / (H*W). Reported cyc_px is per OUTPUT pixel.

* Both branches are real compute (no identity skip). The 3x3 skip conv runs at OUTPUT resolution: 9*Cout^2 MACs per output pixel (2304 for Cout = 16), the heaviest MVAU
  (128 MACs/cycle at T_out = 18; 64 DSP at INT4 of the block's 100).
* ConvTranspose (K = S = 2) is lowered by InferPixelPaddingDeconv to FMPadding_Pixel (zero insertion, (2H+1) x (2W+1) image) + a 2x2 stride-1 window + MVAU:
  4*Cmid*Cmid MACs per output pixel, 3 of 4 window elements are inserted zeros. FMPadding_Pixel emits one word group per pixel of the zero-inserted image.
* UpsampleNearestNeighbour is not foldable: all channels per word, one OUTPUT pixel per cycle (Hout*Wout cycles per frame), a floor on F. Per input row it emits every
  pixel twice (2W cycles) and re-emits the buffered row (2W cycles).
* Two join FIFOs, both sized by the simulation: "skip FIFO" at the end of the ext branch (the shorter-latency branch for the conv variant) and "FIFO main".
* The sliding windows are small here (3x3 at d=1, 2x2), so the next-frame fill gap that hurt the dilated blocks is ~0.6% (18.10 vs 18 cyc/px).
* Untested in FINN: FMPadding_Pixel / UpsampleNearestNeighbour attributes (SIMD), their LUTs are priced 0 or provisional.


Initial block (int_bottleneck.py, int_bottleneck_sim.py, test_int_bottleneck.py), reference FINNInitialBlockConcat, U4: 1 -> 4 channels, 256x256 -> 128x128

    Thr_in -> Dup -> [FMPad -> SWG (3x3, s2) -> MVAU_c (9 -> 3) -> Thr_c] -> FIFO main -+
                     [MaxPool (2x2, s2) -> Thr_m] -----------------------> skip FIFO ---+-> Concat -> Thr_act

* The 2D StreamingMaxPool is not foldable and costs 1.25 cycles per input pixel in FINN's estimate: 81,920 cycles for a 256x256 input. That is a floor on the frame budget of
  the WHOLE network (the other probe blocks used 73,728). With one input channel every front node is pixel-serial (>= 65,536 cycles), so the floor cannot go below 65,536
  without a pixel-parallel maxpool and a multi-pixel input stream; FINN has neither. Practical levers: the clock (cycles are fixed, frames per second scale with f_clk) and
  using 81,920 as the network F (the other blocks then get 11% slack). Whether the real hardware needs the extra 0.25 is for the rtlsim of the init probes to show.
* The conv on one input channel has MW = 9: parallel-window SWG (SIMD 9), MVAU PE 1 x SIMD 9 (9 DSP).
* Concat is StreamingConcat (all channels per word, one output pixel per cycle). The shared branch_quant thresholds (Thr_c, Thr_m) sit before it and the BN + ReLU threshold
  (Thr_act) after it; composing each branch threshold with the matching channels of Thr_act (like the residual merge pass) would remove two threshold nodes -- not done.
* Strided windows can leave the last padded row / column unread: the sliding-window node must drain them before the next frame (bug found while building the sim).

Pool route for the 2D maxpool (initial and downsampling blocks, `pool_impl="swg_pool"`). FINN's `InferPool` lowers a MaxPool to a depthwise sliding window + `Pool_hls` instead of
`StreamingMaxPool`. Pool_hls is foldable: cycles = (C * K^2 / PE) * OH * OW, the depthwise SWG (SIMD = PE) adds fill and per-row overhead. Model rows `SWG_p` / `Pool` replace `MaxPool`.
* initial block (C = 1): the maxpool floor drops from 1.25 * H * W to ~1.0 * H * W, but the depthwise SWG (66,050) and the conv-branch FMPadding (66,564) then dominate; the first sensible
  frame budget is F = 69,632 (4.25 cyc per output pixel), simulated 4.06.
* downsampling block (C = 16, 2x2 window): PE 1 needs ~80.9k cycles, PE 2 fits F = 73,728 (`SWG_p` SIMD 2, `Pool` PE 2).
* Probes `init_..._pool` and `dn_..._mvau_pool` share the ONNX of their StreamingMaxPool twin; they need `InferPool` in the container's FINN (v0.10.1).

Final deconvolution (fnl_block.py, fnl_block_sim.py, test_fnl_block.py; not a bottleneck: single path), reference `LayerQuantEnetFINN.final` =
`QuantConvTranspose2d(c5 -> out_channels, k = 2, s = 2)`, U4: 4 -> 5 channels, 128x128 -> 256x256.

    Thr_in -> FMPadPix (zero insertion) -> SWG_u (2x2) -> MVAU_f (4*Cin -> Cout) -> [Bias (ChannelwiseOp add)]

* No Dup, no skip FIFO, no join, no output threshold (raw logit): only ordinary FIFOs; `BottleneckResult.skip_fifo` is None.
* Pixel domains: FMPadPix emits (2H+1)(2W+1) pixels (66,049 cycles at SIMD = Cin); SWG_u and MVAU_f work at OUTPUT resolution (65,536 pixels), with 4 * Cin * Cout MACs per output pixel
  (3 of 4 window elements are inserted zeros, as in the up-block's transposed conv). One output pixel per cycle is the floor: F >= 65,536. Reference F = 73,728 -> T_out = 1.125.
* U4 result: MVAU_f PE 5 x SIMD 16 (48 DSP), SWG / FMPadPix SIMD 4, bias PE 5; simulated steady 1.01 cyc/px, first-out latency ~263 cycles, all FIFOs 2-4 words (+4-word prefetch before FMPadPix).
* `bias=True` (LayerQuantEnetFINN default, `Int32Bias`) adds the integer bias add; `bias=False` matches the production export (`finn_enet_prod_export.py`). How FINN lowers the bias is
  untested: both variants are probed (6 cases `fnl_cin4_cout5_in128_int{b}_{bias,nobias}`).

PE of the nodes behind the nearest-neighbour upsampler (up_bottleneck.py, `join_pe`; found from the diagnostic probes `up_..._int4_noconv_{pe2,pe4,fjoin,fupnn,fall,pe4fall}`)

Mean-rate balance is not enough behind UpsampleNearestNeighbour. For each input row it runs two phases: phase 1 reads the row and emits every pixel twice (input-limited, one input
pixel per `c_p` cycles), phase 2 re-emits the stored row and needs no new input (output-limited, one output pixel per `d` cycles). The 1x1 projection in front of it (MVAU_p) can only
work during phase 1 unless buffering lets it run ahead. Per input row, with no run-ahead:

    W * c_p  +  2 * W * d   <=   W * t_in        <=>        d <= (t_in - c_p) / 2

* `t_in = F / (H*W)` is the budget per INPUT pixel (= 4 * T_out), `c_p` the cycles per input pixel of the slowest node in front of the upsampler (MVAU_p, NF * SF), `d = Cout / PE` the cycles per
  OUTPUT pixel of the node behind it (Thr_s; Add and Thr_out are widened with it). Plain rate balance would only ask `d <= t_in / 4`, which is why PE 1 looked fine.
* **The PE of Thr_s / Add / Thr_out has to scale up with the target**: `PE >= 2 * Cout / (t_in - c_p)`, rounded up to a divisor of Cout. The slack `t_in - c_p` shrinks as the budget tightens
  (and jumps when MVAU_p changes its fold), so the required PE grows; when `t_in - c_p < 2` even PE = Cout is not enough (the model warns) and the only levers are buffering
  (>= half an input row in front of the upsampler, or one output row behind it) or a looser target. The model derives it automatically for the noconv decoder (`join_pe=None`);
  the conv decoder does not need it because its 3x3 stage (FMPad + SWG line buffer) absorbs the re-emitted rows at full speed.
* U4 up4 (32 -> 16, 32x32 -> 64x64, T_out 18): `t_in` 72, `c_p` 64, so `d <= 4` and PE 4 (16 channels, 4 words per pixel). rtlsim of the INT4 probe: PE 1 (d = 16) 20.53, PE 2 (d = 8) 17.96,
  PE 4 (d = 4) 16.25 cyc per output pixel against the 18.0 target; the node-limited floor is 16.0 (65,536 cycles per frame).
* Buffering is the other cure and the FIFO probes confirm it: 40 words in front of + 91 behind the upsampler gives exactly 16.00 at PE 1; the join FIFO (`Thr_e -> Add`, 17 -> 512 words) does
  nothing (20.53). The model prefers the PE route (PE 4: 3768 LUT, no deep FIFO) over the 91-word FIFO (3885 LUT, 64 bit x 91).
* Why the simulation missed it: FIFO sizing shrank every FIFO to the smallest depth that met T = 18 within 2% (depth 8 -> 17.58), which sits on a steep part of the curve
  (depth 2 / 4 / 8 / 16 -> 20.58 / 19.58 / 17.58 / 16.0); the hardware behaved like depth ~2. A rule of thumb that follows: size FIFOs to the node-limited rate (here 16.0), not to the looser target.

Initial block: Thr_m upstream of the maxpool (updated model). The first init probes showed the landed graph is Dup -> Thr_m -> MaxPool -> Concat, not the exported MaxPool -> Quant: FINN's
`MoveMaxPoolPastMultiThreshold` (hardware/finn_enet_build.py) swaps MaxPool -> MultiThreshold into MultiThreshold -> MaxPool, which pays when the pool follows a wide accumulator but is pointless here
because the pool already reads the INT-A input stream. Consequences in the model: Thr_m runs on H*W pixels (65,536 cycles at PE 1, 80% of F), the skip FIFO sits at the pool output (maxpool -> concat), and
the DWCs are Dup->Thr_m, Thr_m->pool, pool->Concat. Rate and totals are unchanged (the sim still gives 5.02 cyc/px, MaxPool at 100% of F), so the order alone does not explain the 15.78 cyc/px measured in
hardware. What the first probes got wrong is where the join FIFO was forced: on thr_m -> concat, an edge that does not exist in hardware, leaving the real join edge (maxpool -> concat) at FINN's 2 words.
Rule: forced FIFO depths only take effect on edges present in the landed graph; check the edge list of `stages.fifo` ("edge not in prediction") after every new block.

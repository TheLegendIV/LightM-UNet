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

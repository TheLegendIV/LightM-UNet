# Cost formulae used for the fully-unfolded ENet estimate

Source: Blott et al., *FINN-R: An End-to-End Deep-Learning Framework for Fast Exploration
of Quantized Neural Networks*, ACM TRETS 2018 (arXiv:1809.04570), Sec. 3.2. Equation
numbers below refer to that paper.

Quantization assumed throughout: **A = 8** (activations), **W = 8** (weights), uniformly
across every layer.

## Column / symbol glossary

| Column | FINN-R symbol | Meaning |
|---|---|---|
| `cin`, `cout` | `C`, `C'` | input / output channels |
| `hin`,`win`,`hout`,`wout` | `N` (width only) | input/output feature map height & width. FINN-R's `N` is specifically the *width*, since that's the row-length the SWU line buffer streams. |
| `kh`, `kw` | `K` | kernel height / width. FINN-R assumes a square `K`x`K`; ENet has asymmetric `5x1`/`1x5` layers too, see note below. |
| `sh`, `sw` | `S` | **stride height / width** -- how far the window moves per step, vertically / horizontally. |
| `dh`, `dw` | -- (not in FINN-R) | **dilation height / width** -- spacing *between* kernel taps. `dh=dw=1` is an ordinary contiguous kernel; ENet's `stage2`/`stage3` bottlenecks use `dh=dw` up to 16, spreading a nominal 3x3 kernel's taps across a 33-row span. Not part of FINN-R's model -- see `K_eff` below. |
| `A` | `A` | activation bit width -- fixed at 8 |
| `W` | `W` | weight bit width -- fixed at 8 |
| `Q` | `Q` | SIMD width: reduction elements (`C_in*K_h*K_w`) processed per PE per cycle |
| `P` | `P` | PE count: output channels computed in parallel |
| `M` | `M` | multi-vector count: output pixels computed concurrently, sharing weights |

## What "fully unfolded" means here, concretely

For every MVAU-bearing layer: **Q = C_in·K_h·K_w** (whole reduction in one SIMD group),
**P = C_out** (every output channel in parallel, zero neuron-fold). Two variants of **M** are
produced: `Mmax` files use **M = H_out·W_out** (every output pixel replicated too); `M1`
files use **M = 1** (no spatial replication -- the more usual "fully parallel MVAU" reading).

## Conv2d (and FC, unused here): SWU + MVAU + weight memory + LUTs

**SWU BRAM (Eq. 4):**
```
BRAM_swu = M * (ceil(K_eff / S) + 1) * ceil(S*N / 512) * ceil(C*A / 36)
```
`K_eff = (K_h-1)*dh + 1` -- effective kernel span once dilation is accounted for (extension
past the paper's dilation=1 assumption). Asymmetric kernels (`5x1`/`1x5`) use `K_eff` from
the *height* dimension only, since that's what drives row-buffer depth -- a `1x5` kernel
needs 1 buffered row, a `5x1` needs 5. This slightly underestimates the `1x5` case, which the
paper's square-kernel model has no separate term for.

**Weight memory BRAM (Eq. 5):**
```
BRAM_wm = P * ceil(omega/512) * ceil(Q*W/36),   omega = K^2*C*C' / (Q*P)
```
Collapses to `omega = 1` always in this fully-unfolded regime (`Q*P` always equals the full
weight volume `K^2*C*C'`), so `BRAM_wm = P * ceil(Q*W/36)`. Identical between the `Mmax` and
`M1` files -- weight memory doesn't scale with spatial replication, only SWU/line-buffer cost
does (M *shares* weights, per the paper).

**LUT cost -- two separate components (Eq. 1: `LUT_CNV = LUT_SWU + LUT_MVU`):**
```
LUT_swu = M * 426        (Sec. 3.2: fixed per-instance SWU control-logic overhead
                           for a full feed-forward dataflow SWU; scaled by M here for
                           consistency with BRAM_swu's M-scaling -- M physical SWU
                           instances, each with this baseline)

LUT_mvu = c0 + c1 * M * (P*Q) * (W*A),   c0=300, c1=1.1
                          (Sec. 3.2.1 / Fig. 9 empirical fit -- literal formula from
                           the paper, M already appears inside it, not applied twice)

total_lut = LUT_swu + LUT_mvu
```

**MVAU compute:**
```
total_PE          = P * M           = C_out * M
total_SIMD_lanes  = P * Q * M       = C_out * (C_in*K_h*K_w) * M
```
Cross-check (still holds): when `M = H_out*W_out`, `total_SIMD_lanes` equals the layer's
total MAC count. Verified against the `macs` column in `enet_architecture.csv` for all 86
Conv2d layers, zero mismatches.

## MaxPool2d: SWU + comparator array, no MVAU, no weights

```
BRAM_swu = same Eq. 4, using the pool's own K, S, N, C
LUT_swu  = M * 426                    (same fixed SWU overhead as above)
LUT_mp   = M * A * C                  (Sec. 3.2.1: "roughly equivalent to the product
                                        of A and C" for the comparator tree; M-scaled
                                        here for the same consistency reason as LUT_swu
                                        -- the paper's own quote is implicitly the M=1
                                        base case)
total_lut = LUT_swu + LUT_mp
```
No PE/SIMD, no weight memory -- pooling is a comparator tree, not a MAC array.

## ConvTranspose2d: modeled via zero-insertion + ordinary convolution

A transposed convolution with kernel `K`, stride `S`, padding `p` is mathematically
equivalent to: (1) insert `S-1` zero rows/columns between every input pixel, (2) zero-pad
the result by `K-p-1` on each side, (3) run an *ordinary* stride-1, `K`x`K` convolution over
that expanded map (Dumoulin & Visin, *A guide to convolution arithmetic for deep learning*).
That's a real Conv2d in FINN's SWU/MVAU sense, so all the Conv2d formulae above (SWU, weight
memory, both LUT terms) apply directly -- these three layers are included in the totals, not
excluded.

All three of ENet's ConvTranspose2d layers (`up4.up.0`, `up5.up.0`, `final`) use `K=S=2`,
`p=0` (confirmed by back-solving PyTorch's output-size formula against the actual recorded
shapes). For each:
```
N_eff = (W_in - 1)*S + 1 + 2*(K - 1)        <- the zero-inserted, padded width the SWU
                                                 has to stream/buffer
S_derived = 1                                <- the equivalent conv always runs at stride 1
K, C_in, C_out                               <- unchanged from the transpose layer itself
```
`sh`/`sw` in the CSVs for these three rows are the *derived* stride (always 1), not the
transpose layer's own stride of 2 -- that's already folded into `N_eff`.

**Worth flagging explicitly -- this inflates the honest MAC count (and both LUT terms) by
4x.** With `K=S` and no padding, every output position's receptive field is real data at
exactly one of the four sub-pixel phases and zero at the other three, but a naive
zero-insert-then-convolve implementation doesn't know that and multiplies through the zeros
anyway. Checked directly: `total_SIMD_lanes` for all three layers comes out to exactly **4x**
the dense-equivalent MAC count already present in `enet_architecture.csv`'s `macs` column
(e.g. `up4.up.0`: 67,108,864 vs. 16,777,216). Since `LUT_mvu` scales with the same `P*Q*M`
product, it inherits the same 4x inflation. That's the real, quantifiable cost of
implementing learned upsampling this way rather than with a sub-pixel/phase-skipping
formulation.

## Not covered by these files

- Threshold memory cost -- per the paper, negligible at these bit widths.
- Any routing/interconnect or shell/platform overhead (Sec. 3.4) -- device- and
  platform-specific, out of scope here.

Say the word if you want either added.

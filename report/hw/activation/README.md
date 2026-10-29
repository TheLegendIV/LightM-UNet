# ReLU vs PReLU on the hardware baseline

Question: PReLU raises FP32 accuracy; what does it do after QAT, and what does it cost in hardware? "Hardware baseline" = S12 dense / ReLU / bilinear / U4 (`nnUNetTrainerENet_12_dense_relu`, QAT source `..._warmstart150ep`). All artifacts of this experiment live in this directory (exceptions: the trainer class and network code must sit under `enet/`).

## Files

| File | What |
|---|---|
| `stage_12_dense_prelu_all.job` | FP32 phase 1: hardware baseline with PReLU in encoder AND decoder, 150 epochs cold start |
| `stage_12_dense_prelu_all_warmstart150ep_v2.job` | FP32 phase 2 (gpu_mig): +150 epochs at LR 1e-4 warm-started from the FINISHED phase 1 (the ReLU QAT arms use the same 300-epoch recipe). Refuses to start unless phase 1 is at epoch 150 and `check_warmstart_transfer.py` confirms every parameter incl. all 82 PReLU alphas transfers bit-exactly |
| `check_warmstart_transfer.py` | Pre-flight for the warm start (nnU-Net's loader only warns when a key does not transfer) |
| `stage_12_dense_prelu_all_warmstart150ep.job` | SUPERSEDED v1 of phase 2: submitted while phase 1 was still running, it warm-started from phase 1's epoch-~27 checkpoint (dice 0.7621). Kept for the record; its results row is not comparable |
| `calibrate_prelu.py`, `bits/` | PTQ + calibration of the PReLU `LayerQuantEnetFINN`; uniform INT4/6/8 per-site bits (copied from the ReLU arm; the originals are gitignored under `ILP/artifacts/archive/`) |
| `qat_12_dense_prelu_all_warmstart150ep_bilinear_uniform_int468_ft15ep_array.job` | QAT INT4/6/8, 15 epochs, twin of the ReLU job. Trainer: `nnUNetTrainerLayerQuantEnetFINN_12_dense_prelu_bilinear_upsample_perlayer`; `QuantPReLU` in `enet/nnunetv2/nets/QuantENet.py` |
| `plot_prelu_alpha.py`, `4_8_dense_dilation_alpha.{png,csv}` | Learned alphas of the existing encoder-only PReLU checkpoint (per channel / per site) |
| `prelu_hw_cost.py`, `prelu_hw_cost_summary.csv`, `prelu_hw_cost_int{4,6,8}_sites.csv` | Hardware cost model below |

Run order: phase 1 -> phase 2 v2 (`sbatch --dependency=afterok:<phase1 id>`; the job also guards itself) -> QAT array (reads the v2 checkpoint). Phase 1 is done (dice 0.7979, vs 0.7876 ReLU baseline and 0.7960 encoder-only PReLU); v2 and the QAT array have not been submitted.

## Hardware cost model: assumptions (variant v1)

Reference designs: `ILP/artifacts/S12_dense_512_uniform_cap1_v2/uniform_int{4,6,8}` (S12 dense bilinear 512, uniform bits, 250 FPS @ 100 MHz, MILP folding, ZU7EV: 230,400 LUT / 624 BRAM18 / 1,728 DSP). Priced with `ILP/finn_cost_model.py`.

1. **ReLU site** = one standalone `Thresholding_rtl`, UINT b (2^b - 1 thresholds per channel). 82 sites: the `reduce`/`conv`/`up` conv thresholds (PE = the conv's `thr_pe`, input width = its accumulator `acc_bits`), plus the `out_act` and `initial.act` join thresholds (input INT10). Not activation sites, unchanged in both nets: `expand` thresholds, `skip_quant`, `residual_add` (linear requantisers).
2. **PReLU site** (unclamped, any alpha) = two ReLUs: `PReLU(x) = ReLU(x) + alpha*min(x,0)` -> `DuplicateStreams` -> two `Thresholding_rtl` (+ branch, - branch) -> `AddStreams`. Folding is unchanged (same PE, same cycles).
3. **Branch width.** The branches have disjoint support, so their sum is already on the signed b-bit output grid (no re-quantisation). The signed grid's thresholds are split over the branches: each branch is a (b-1)-bit Thresholding node (2^(b-1) - 1 steps), versus one b-bit node for ReLU.
4. **Negative branch is free.** The comparison direction / output sign of the - branch is folded into the threshold node itself (no extra negation op). Unverified in FINN; if it fails, add a x(-1) op per site (about +4.1k LUT at INT6, priced like the bias op, 50.6 LUT/PE).
5. **Fork and add prices** are the repo's measured constants at PE=1 on 6-8 bit streams: 65.84 LUT per PE (`DuplicateStreams`), 68.23 LUT per PE (`AddStreams`), assumed linear in PE. No width scaling (the fork here sits on a 10-20 bit accumulator stream; scaling it per bit would add about +4.9k LUT at INT6).
6. **Threshold prices** use the current `_thresholding_rtl_cost` (2026-10-17 refit) for both ReLU and PReLU. The artifacts were solved before that refit, so the baseline totals are summary.csv's solved LUT plus the re-pricing difference of the 82 ReLU thresholds. force_dsp calibration factor is 1.0 (raw = calibrated).
7. **Memory.** Every threshold in these solves is `distributed` (LUTRAM), so BRAM does not change. A `block` RAM-style design would change the BRAM numbers.
8. **Signed output.** The PReLU net's activations are signed (`QuantPReLU` uses a signed quantizer), so each site's output grid has 2^b levels split as above. The effect of signed inputs on the next conv's accumulator width (about one bit narrower) is not modelled.
9. **"Fusable" sensitivity.** A site whose trained alphas all lie in (0, 1) is monotone and fuses into a single b-bit threshold (= ReLU cost). Alphas come from the encoder-only checkpoint `4_8_dense_dilation` (11 of the 67 encoder sites qualify); the 15 decoder sites have no trained alphas and count as not fusable. The all-PReLU model's own alphas will replace this once trained.

Not priced: the extra depth-2 stream FIFOs (about 4 per site, about 12 LUT each for 6-bit streams, wider on the accumulator side; roughly +4k LUT at INT6), DWCs between the new nodes, and routing/timing effects. No real PReLU build exists; the Thresholding fit is an expected-value fit (single nodes can be 3-4x off), Dup/Add constants have only PE=1 evidence.

## Results (PReLU over ReLU, all 82 sites, v1)

| | INT4 | INT6 | INT8 |
|---|---|---|---|
| Baseline design LUT (% of device) | 93,288 (40.5%) | 110,515 (48.0%) | 139,820 (60.7%) |
| ReLU activation thresholds, LUT | 6,318 | 12,853 | 25,773 |
| PReLU activations (fork + 2 thr + add), LUT | 20,248 | 30,970 | 49,626 |
| **Delta LUT** | **+13,930** | **+18,117** | **+23,854** |
| Delta as % of design / pp of device | +14.9% / +6.0 | +16.4% / +7.9 | +17.1% / +10.4 |
| Device LUT after | 46.5% | 55.8% | 71.0% |
| of which fork + add | 11.0k | 11.0k | 11.0k |
| Delta BRAM18 / DSP / cycles | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Delta LUT if the 11 fusable sites fuse | +12,045 | +15,591 | +20,276 |

The fork + add overhead (10,994 LUT) is fixed across bit widths; the threshold part grows with b. Regenerate: `docker exec lightmunet_dev python3 /workspace/LightM-UNet/report/hw/activation/prelu_hw_cost.py`.

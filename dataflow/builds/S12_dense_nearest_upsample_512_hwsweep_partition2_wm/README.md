# S12_dense_nearest_upsample_512_hwsweep_partition2_wm -- MILP dsr/pbi-ratio hardware validation

Real Vivado OOC-synth validation of `MILP/artifacts/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/`'s
5 sweep points (`baseline_both_off`, `dsrSweep_pbiOff_dsr{1.5,3.0,5.0}`, `pbiSweep_dsrOff_pbi1.5`),
partition 2 only, PLUS one extra build for `baseline_both_off` using FINN's own
auto-fold instead of the MILP-chosen (`per_layer` pe/simd) folding -- 6 builds total.

Architecture: `12_dense_relu_nearest_upsample` (decoder_type="nearest_upsample",
CHANNELS=(4,16,32,16,4), see `../12_dense_relu_nearest_upsample_512/`), 512x512 input.

Trained weights: all 6 builds re-quantize the SAME real fine-tuned checkpoint
(`nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_upsample_perlayer_
12_dense_relu_nearest_upsample_wm_joint_alpha1.0_perlayer_candidatebits468_
forcedsp_lut50_bram50_dsp90_ft15ep`, `fold_0/checkpoint_best.pth.txt`) to each
sweep point's own `layer_bits_SITES_<tag>.json` bit-widths, via
`LayerQuantEnetFINN.from_pretrained()`.

No URAM: none of these builds run the custom `step_allocate_uram_fifos` step
that the 8-way full builds use -- all FIFOs stay at FINN's default
`ram_style` (BRAM/auto-inferred, never forced to `ultra`).

## Pipeline (per tag)

1. **Export** (pytorch container): `finn_export_trained.py --tag <tag> --bits-file <SITES.json>`
   -> `outputs/quantEnet_12_dense_relu_nearest_upsample_trained_<tag>_512x512.onnx`
2. **Conv-order dump** (pytorch container, run ONCE, order is bit-width-independent):
   reuse `../12_dense_relu_nearest_upsample_512/finn_hawq_dump_conv_order_12_dense_relu_nearest_upsample.py`
   as-is -> `quantEnet_12_dense_relu_nearest_upsample_dummy_int8_conv_order.json`
3. **Preamble** (FINN container, no Vivado): `finn_hawq_preamble_trained.py <tag>`
   -> `finn_deployment_outputs/S12_dense_nearest_upsample_512_hwsweep_wm_<tag>_preamble_<ts>/`
4. **Folding bridge** (FINN container): `finn_hawq_folding_bridge_nearest_upsample.py <preamble_dir> <layer_bits_folding_<tag>.json>`
   -> `<preamble_dir>/hawq_folding_config_partition2.json`
5. **Partition-2 OOC synth** (FINN container, real Vivado):
   `finn_ooc_partition2_trained.py <preamble_dir> <tag> [<bridged_folding_config.json>]`
   -- 3rd arg omitted => auto-fold (skips `step_apply_folding_config` entirely).

Steps 1-4 are cheap (no Vivado) and run sequentially per tag; step 5 is the
expensive real-synth step and is launched for all 6 tags in parallel (see
`run_sweep.sh`), following the precedent in `hardware/temp/run_partitions_3456_parallel.sh`.

Deployment: scripts are `docker cp`-ed flat into the FINN container's
`finn/notebooks/enet/` (see AGENTS.md) -- this folder's layout is git
organization only.

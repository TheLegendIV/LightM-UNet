# S12_dense_arms_bc_v1 -- MILP folding vs FINN auto-fold (partition 2 only)

Three builds, **same bits** (`MILP/artifacts/S12_dense_arms_bc_v1/lex_dsr2_fps305`, QAT checkpoint
re-quantized), same FPS target 305.17, same `mvau_wwidth_max = 80`; only the folding differs:

| Arm | Folding | Source |
|---|---|---|
| `armB_finn_autofold` | FINN `step_target_fps_parallelization` | `ARMS_TARGET_FPS=305.17` |
| `armC_dsr2` | MILP lexicographic pass 2, DSR 2 | `lex_dsr2_fps305/layer_bits_folding_lex_dsr2_fps305.json` |
| `armC_nodsr` | MILP pass 2, same bits/roof, no DSR | `lex_nodsr_fps305/layer_bits_folding_lex_nodsr_fps305.json` |

Scripts are reused from `../S12_dense_nearest_upsample_512_hwsweep_partition2_wm/` (container dir is flat);
they gained an `ARMS_TARGET_FPS` env override, and the bridge a 3rd out-path arg.

## Inputs (docker cp into the FINN container's `/home/thelegendiv/finn/notebooks/enet/`)
- `../S12_dense_nearest_upsample_512_hwsweep_partition2_wm/outputs/quantEnet_12_dense_relu_nearest_upsample_trained_lex_dsr2_fps305_512x512.onnx` (already exported from the new QAT `checkpoint_best.pth`)
- `../S12_dense_nearest_upsample_512_hwsweep_partition2_wm/outputs/quantEnet_12_dense_relu_nearest_upsample_dummy_int8_conv_order.json`
- the two MILP folding JSONs above
- patched scripts from the `_wm` build dir: `finn_hawq_preamble_trained.py`, `finn_hawq_folding_bridge_nearest_upsample.py`, `dump_autofold_config_all_partitions.py`, `finn_ooc_partition2_trained.py`
- `run_arms.sh`

Run: `docker exec -e HOME=/tmp/home_dir <finn_container> bash /home/thelegendiv/finn/notebooks/enet/run_arms.sh`

## Gates (before trusting any OOC number)
1. Bridge logs show `Bridged 5 SWU node(s)` with `parallel_window=1`, no WARNING lines.
2. `python hardware/check_milp_vs_landed_folding.py <milp folding json> <armC_* partition2 prefifo .onnx>` -> `OK` for both C arms.
3. Arm B: price with `MILP/utils/apply_folding_config_cost.py` (partition 2) and, for the whole network,
   `MILP/utils/price_autofold_all_partitions.py --dir autofold_armB_all_partitions`.

## Table columns
Estimated resources/FPS: cost model on each landed graph. Real resources/FPS: OOC (partition 2).
Total buffer: sum of FIFO depth and depth x width from the sized graph. Sensitivity: 0.02430 (norm mean),
identical for all arms. Accuracy: Dice of the QAT run `..._lex_dsr2_fps305_..._ft15ep` (same for all arms).

# S12_dense_256_u4_analytical_v1: per-shape standalone builds

One FINN partition per analytical **block** (29) instead of the 8-way stage split, and one standalone build (HLS + IP stitch + rtlsim, optional OOC synth) per distinct block **shape** (12), using the
S12 U4 analytical folding **unchanged** (`MILP/artifacts/S12_dense_256_u4_analytical_v1/int6_fps250_lat200/layer_bits_folding_final.json`, including its simulated `intra_block_fifos`). Purpose: rtlsim every block
shape alone, to find which block diverges from the cycle simulator (stalls, latency, FIFO occupancy). Written but NOT run: `py_compile` plus the cut finder tested on the eight specialised partition graphs of
`outputs/int6_fps250_lat200_milpfold_8way_20261005_235400` (all 90 weight nodes matched, one stream per cut). The FINN container is run by hand.

## Shapes (`block_shapes.json`, made by `make_block_shapes.py` in lightmunet_dev)

| shape | built block | identical blocks (same hardware) |
|---|---|---|
| 0 init | initial | - |
| 1 dn | down1 | - |
| 2 reg | regular1.0 | regular1.1, regular1.2, regular1.3, regular4.0, regular4.1 |
| 3 dn | down2 | - |
| 4 reg | stage2.0 | stage2.4, stage3.0, stage3.4 |
| 5 reg | stage2.1 | stage2.5, stage3.1, stage3.5 |
| 6 reg | stage2.2 | stage2.6, stage3.2, stage3.6 |
| 7 reg | stage2.3 | stage2.7, stage3.3, stage3.7 |
| 8 up | up4 | - |
| 9 up | up5 | - |
| 10 reg | regular5.0 | - |
| 11 final | final (+ bias, argmax) | - |

(`stage3.7` is hardware-identical to `stage2.3`: dilation 8 on a 32x32 map.) A shape is the block kind, geometry and every node's PE/SIMD; use `bash run_blocks.sh stage3.7` to build any single block by name.

## What changed in the flow (additive; the 8-way flow is untouched)

* `hardware/finn_s12_blocks.py` (new): block names from conv_order.json (`regular1.0.reduce.0` -> `regular1.0`), cut after each block's out_act (the Thresholding behind its AddStreams / StreamingConcat), fail-fast if
  the block's weight nodes are not contiguous or more than one stream crosses a cut; `assign_block_partition_ids` build step; `block_partitions.json` (id, stage, node range, logical conv / pool names per block).
* `finn_s12_preamble.py --blocks --conv-order ...`: the same preamble with that step in place of the 8-way tagging.
* `finn_s12_build.py --blocks --partitions shapes|<block names>|<ids> [--block-shapes block_shapes.json] [--no-ooc]`: standalone builds on the block partitions (the bridge needs nothing new: it is per partition);
  `finn_s12_build_steps.match_conv_order_to_nodes` is the 8-way bridge's positional matcher, factored out unchanged.

## Inputs (docker cp into the flat `/home/thelegendiv/finn/notebooks/enet/`)

* `hardware/finn_s12_blocks.py`, `finn_s12_preamble.py`, `finn_s12_build.py`, `finn_s12_build_steps.py` (changed) and the usual S12 flow files (`finn_s12_*`, `finn_stage_partition.py`, `finn_partition_build_steps.py`, ...).
* the QAT'd export `outputs/quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep.onnx` (+ `.onnx.data`) and `..._conv_order.json`.
* `MILP/artifacts/S12_dense_256_u4_analytical_v1/int6_fps250_lat200/layer_bits_folding_final.json` **as** `layer_bits_folding_u4_analytical.json`.
* `block_shapes.json`, `run_blocks.sh` from this folder.

## Run

```
docker exec -e HOME=/tmp/home_dir <container> bash -c 'cd /home/thelegendiv/finn/notebooks/enet && STEP=bridge bash run_blocks.sh'   # preamble + partitioning + bridge dry run, no Vivado
docker exec -e HOME=/tmp/home_dir <container> bash -c 'cd /home/thelegendiv/finn/notebooks/enet && bash run_blocks.sh'               # + all 12 builds (max 4 at a time), rtlsim, no OOC synth
docker exec -e HOME=/tmp/home_dir <container> bash -c 'cd /home/thelegendiv/finn/notebooks/enet && bash run_blocks.sh up4 up5'       # only these blocks
```

Check the first run: 29 `[blocks] partition ...` lines (stage names in dataflow order, each with its HW node count); then per block `weight node count == logical name count` (the bridge aborts otherwise) and
`FIFO role bridge: N matched / M unresolved`. Each build writes `partition<id>_blocks256_milpfold_stitched.onnx` and `report/rtlsim_performance.json` under
`finn_deployment_outputs/blocks256_milpfold_blocks12_<ts>/partition_<id>_<block>/`.

## Caveats

* The standalone rtlsim drives the block with FINN's stock `step_measure_rtlsim_performance` (saturated input, a few frames), the same as the existing partition builds. Compare its cycles with the cycle simulator's
  `frame_cycles` / `latency_first_out` for the same block (`MILP/analytical/net_explicit.py` json, `fifo_sim.blocks`).
* The first block (`initial`) contains the network input transpose and the last (`final`) the argmax; both are tagged like the 8-way flow's partitions 0 and 7.
* `validate_partition_single_output` is a warning in blocks mode (the 8-way flow raises).
* The bridge applies the analytical FIFO lists by role: FIFOs it cannot match stay at depth 2 (`--fifo-autosize fixed2`, the default). The inter-block FIFOs of the folding json are unmatched in a single-block build (depth 2).

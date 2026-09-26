# MILP/utils — occasional tools (not part of the MILP flow)

Maintained and runnable; each imports the core modules from `MILP/`.

| script | use |
|---|---|
| `critical_path_latency.py` | True DAG critical-path latency of a solved plan (max at joins) vs the ILP's sum-of-cycles proxy. |
| `scan_fork_join_mismatch.py` | Ranks cycle imbalance at every fork/join of a solved plan, with a predicted FIFO depth. |
| `compute_receptive_field.py` | Receptive field of a config's architecture. |
| `uniform_bits_same_folding.py` | Naive baseline (PIPELINE stage 6b): keeps a solve's folding, forces uniform bits, recomputes cost. |

For how a solve's folding/cycles look per node, prefer the `dataflow_*.onnx`
that `finn_milp.py` now writes next to every result (see `../finn_milp.md`).

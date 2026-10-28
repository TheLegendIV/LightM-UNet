"""Driver: runs the REAL, unmodified
finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_8way_full.py --
same module, same main(), same _build_one_partition_with_folding_and_dsp/
step_build_all_partitions_with_folding_and_dsp/step_force_dsp/
split_large_fifos machinery every real S12 8-way build uses -- against the
tiny 8x QuantRegularBottleneck network instead of the real S12 checkpoint.

Zero edits to that file. Only 3 things need to be redirected before calling
its own main():
  - CONV_ORDER_FILE / FOLDING_BLOCK_FILE (module-level path constants,
    looked up by name inside its own load_all_partition_logical_names()/
    main() at call time -- overriding the attribute on the already-imported
    module object redirects them with no source edit)
  - compute_8way_boundaries (imported by that module via `from
    finn_stage_partition import compute_8way_boundaries`, also just a
    module-global name looked up at call time) -- the real function detects
    down1/down2/up4/up5 stage boundaries via StreamingMaxPool/
    UpsampleNearestNeighbour node counts, which this flat, no-down/upsampling
    toy network has none of. Same DuplicateStreams-count boundary logic as
    finn_preamble_probe_8x_bottleneck_min_int4.py's own
    assign_stage_partition_ids_8way (must match EXACTLY -- both operate on
    the same step_enet_convert_to_hw.onnx checkpoint).

Run finn_export_probe_8x_bottleneck_min_int4.py then
finn_preamble_probe_8x_bottleneck_min_int4.py first (produces the
preamble_dir this script's PREAMBLE_DIR below must point at).

Usage (inside the FINN container, all 4 files docker cp'd into
finn/notebooks/enet/):
    python3 finn_test_8way_full_against_tiny_8x_bottleneck.py <preamble_dir> [output_dir]
"""
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.transformation.general import SortGraph  # noqa: E402

import finn_stage_partition  # noqa: E402
import finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_8way_full as target  # noqa: E402

N_BLOCKS = 8
PARTITION_RANGE_ORDER = target.PARTITION_RANGE_ORDER  # ["down1_start","down2_start","q2_start","q3_start","q4_start","up4_start","up5_start"]


def compute_8way_boundaries_flat(model):
    """Same DuplicateStreams-count boundary detection as
    finn_preamble_probe_8x_bottleneck_min_int4.py's own
    assign_stage_partition_ids_8way -- must stay in sync (both read the
    same step_enet_convert_to_hw.onnx checkpoint). Returns a dict keyed by
    PARTITION_RANGE_ORDER's 7 names (semantically meaningless for this flat
    topology -- _8way_full.py only uses them as ordered dict keys)."""
    model = model.transform(SortGraph())
    dup_idx = [i for i, n in enumerate(model.graph.node) if n.op_type == "DuplicateStreams"]
    assert len(dup_idx) == N_BLOCKS, (
        f"expected {N_BLOCKS} DuplicateStreams nodes, found {len(dup_idx)} -- "
        "did finn_preamble_probe_8x_bottleneck_min_int4.py run against the same checkpoint?"
    )
    boundaries = dup_idx[1:]
    assert len(boundaries) == len(PARTITION_RANGE_ORDER)
    return dict(zip(PARTITION_RANGE_ORDER, boundaries))


def main():
    if len(sys.argv) < 2:
        print("Usage: finn_test_8way_full_against_tiny_8x_bottleneck.py <preamble_dir> [output_dir]")
        sys.exit(1)
    preamble_dir = sys.argv[1]

    conv_order_file = os.path.join(preamble_dir, "conv_order.json")
    folding_block_file = os.path.join(preamble_dir, "folding_block.json")
    assert os.path.isfile(conv_order_file), f"missing {conv_order_file} -- run the preamble script first"
    assert os.path.isfile(folding_block_file), f"missing {folding_block_file} -- run the preamble script first"

    # redirect the 3 module-level things _8way_full.py needs pointed at this toy network
    finn_stage_partition.compute_8way_boundaries = compute_8way_boundaries_flat
    target.compute_8way_boundaries = compute_8way_boundaries_flat
    target.CONV_ORDER_FILE = conv_order_file
    target.FOLDING_BLOCK_FILE = folding_block_file

    print(f"Preamble dir       : {preamble_dir}")
    print(f"CONV_ORDER_FILE    : {target.CONV_ORDER_FILE}")
    print(f"FOLDING_BLOCK_FILE : {target.FOLDING_BLOCK_FILE}")

    # target.main() reads preamble_dir/output_dir straight from sys.argv,
    # which already carries this same script's own argv unchanged.
    target.main()


if __name__ == "__main__":
    main()

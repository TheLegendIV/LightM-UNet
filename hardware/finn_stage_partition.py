"""5-way stage-based graph partitioning for S19 (and structurally similar
ENet-family exports), to avoid the near-quadratic Vivado `create_bd_cell`
stitching blowup observed on the monolithic 1665-cell single-partition
stitch (see foundation_log.md / session notes, 2026-08 build attempts).

Boundary rule (per user spec): cut right after each down/upsampling
bottleneck's FIRST op, so each named stage owns its own down/upsample:

    partition 0  initial                       (initial block only)
    partition 1  stage1   = down1 + regular1
    partition 2  stage2/3 = down2 + stage2 + stage3   (no down/upsample
                            between stage2 and stage3, so kept merged)
    partition 3  stage4   = up4 + regular4
    partition 4  stage5   = up5 + regular5 + final

Boundaries are detected STRUCTURALLY (not by hardcoded node index), using
two facts confirmed against the real S19 build's own
step_enet_convert_to_hw.onnx checkpoint (580 HW nodes):

  - FINNDownsamplingBottleneck.forward() computes `shortcut_proj(
    shortcut_pool(x))` FIRST, before its main conv path. FINN lowers the
    nn.MaxPool2d shortcut into a single "StreamingMaxPool" HW node, which
    is therefore always the FIRST node of down1/down2. There are exactly
    2 StreamingMaxPool nodes in this network (down1, down2).

  - FINNUpsamplingBottleneck.forward() computes `main_act(main_bn(
    main_up(x)))` FIRST, before its reduce/up/expand path. FINN lowers
    each QuantConvTranspose2d into a zero-insertion step implemented as
    an "FMPadding_Pixel" HW node followed by a regular conv. Each
    upsampling bottleneck (up4, up5) has exactly 2 ConvTranspose2d calls
    (main_up + the "up" sub-path), so contributes 2 FMPadding_Pixel nodes;
    the final QuantConvTranspose2d layer contributes a 5th, trailing one.
    So the FIRST node of up4 is FMPadding_Pixel occurrence #0 (of 5), and
    the first node of up5 is occurrence #2 (of 5). The 5th occurrence
    (final's own transpose) is NOT a partition boundary -- it stays
    merged into the stage5 partition per the user's spec.

Usage: call assign_stage_partition_ids(model) as its own build step,
inserted directly after step_enet_convert_to_hw and before
step_create_dataflow_partition_multi (see finn_partition_build_steps.py).
"""

from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import get_by_name


def _node_index(model, node):
    return list(model.graph.node).index(node)


def _is_fpgadataflow_node(node):
    """Mirrors CreateDataflowPartition's own internal `assign_partition_id`
    lambda (create_dataflow_partition.py): only nodes with a `backend`
    attribute equal to "fpgadataflow" are considered part of any partition
    -- everything else (leftover Reshape/Transpose/Constant/etc. nodes that
    step_enet_convert_to_hw didn't turn into HW ops) is left alone by
    CreateDataflowPartition regardless (treated as -1 / stays in the
    parent graph). Calling getCustomOp() on one of those non-HW nodes
    raises `ValueError: Empty module name` (empty node.domain), so we must
    skip them here too."""
    backend = get_by_name(node.attribute, "backend")
    return backend is not None and backend.s.decode("UTF-8") == "fpgadataflow"


def find_stage_boundaries(model):
    """Returns a sorted list of 4 node indices marking the start of
    stage1, stage2/3, stage4, stage5 respectively (partition 0/initial
    always starts at index 0). Raises AssertionError if the expected
    marker node counts aren't found (2 StreamingMaxPool, 5
    FMPadding_Pixel) -- this is a deliberate fail-fast so a topology
    change doesn't silently mis-partition the graph."""

    maxpools = model.get_nodes_by_op_type("StreamingMaxPool")
    assert len(maxpools) == 2, (
        "Expected exactly 2 StreamingMaxPool nodes (down1, down2), found %d. "
        "Topology may have changed -- update find_stage_boundaries()." % len(maxpools)
    )
    fmpad = model.get_nodes_by_op_type("FMPadding_Pixel")
    assert len(fmpad) == 5, (
        "Expected exactly 5 FMPadding_Pixel nodes (up4 x2, up5 x2, final x1), "
        "found %d. Topology may have changed -- update find_stage_boundaries()."
        % len(fmpad)
    )

    # sort each group by their position in the graph, so "first occurrence"
    # is well defined regardless of get_nodes_by_op_type's internal order
    maxpools = sorted(maxpools, key=lambda n: _node_index(model, n))
    fmpad = sorted(fmpad, key=lambda n: _node_index(model, n))

    down1_start = _node_index(model, maxpools[0])
    down2_start = _node_index(model, maxpools[1])
    up4_start = _node_index(model, fmpad[0])
    up5_start = _node_index(model, fmpad[2])
    # fmpad[4] (final's own transpose) is intentionally not a boundary

    boundaries = [down1_start, down2_start, up4_start, up5_start]
    assert boundaries == sorted(boundaries), (
        "Detected stage boundaries are not in ascending topological order "
        "(%s) -- topology assumption (strictly sequential, no cross-stage "
        "skip connections) may not hold." % boundaries
    )
    return boundaries


def assign_stage_partition_ids(model, cfg=None):
    """Custom build step: sets the 'partition_id' nodeattr on every
    fpgadataflow node according to the 5-stage split described above.
    Must run after step_enet_convert_to_hw and before
    step_create_dataflow_partition_multi. `cfg` is accepted (and ignored)
    only so this matches the (model, cfg) step-function signature FINN's
    build_dataflow driver expects."""

    down1_start, down2_start, up4_start, up5_start = find_stage_boundaries(model)

    n_initial = n_stage1 = n_stage23 = n_stage4 = n_stage5 = 0
    n_skipped = 0
    for idx, node in enumerate(model.graph.node):
        if idx < down1_start:
            pid = 0
            n_initial += 1
        elif idx < down2_start:
            pid = 1
            n_stage1 += 1
        elif idx < up4_start:
            pid = 2
            n_stage23 += 1
        elif idx < up5_start:
            pid = 3
            n_stage4 += 1
        else:
            pid = 4
            n_stage5 += 1
        if not _is_fpgadataflow_node(node):
            # non-HW node (e.g. a leftover Reshape/Transpose/Constant) --
            # CreateDataflowPartition ignores these regardless of
            # partition_id (its own lambda assigns them -1), and
            # getCustomOp() would raise on their empty domain, so skip.
            n_skipped += 1
            continue
        inst = getCustomOp(node)
        inst.set_nodeattr("partition_id", pid)

    print(
        "[assign_stage_partition_ids] boundaries down1=%d down2=%d up4=%d up5=%d"
        % (down1_start, down2_start, up4_start, up5_start)
    )
    print(
        "[assign_stage_partition_ids] node counts: initial=%d stage1=%d "
        "stage2/3=%d stage4=%d stage5=%d (total=%d, skipped non-HW=%d)"
        % (n_initial, n_stage1, n_stage23, n_stage4, n_stage5, len(model.graph.node), n_skipped)
    )
    return model

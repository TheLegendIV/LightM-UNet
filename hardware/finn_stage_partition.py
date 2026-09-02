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
    is therefore always the FIRST node of down1/down2. There are either 2
    (down1, down2 only -- the original fresh-init single-conv initial
    block, e.g. 26_5_w24) or 3 (+ the real Concat-based initial block's
    own MaxPool branch, FINNInitialBlockHAWQ, added 2026-08-28, e.g.
    26_9_w24_ptq) StreamingMaxPool nodes in this network -- either way,
    down1/down2 are always the LAST two by node index (any initial-block
    pool always precedes down1 entirely).

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
from qonnx.transformation.general import SortGraph
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
    marker node counts aren't found (2 or 3 StreamingMaxPool, 5
    FMPadding_Pixel) -- this is a deliberate fail-fast so a topology
    change doesn't silently mis-partition the graph.

    2 StreamingMaxPool = down1/down2 shortcut_pool only (the original
    fresh-init single-conv initial block, e.g. 26_5_w24). 3 = same plus
    the real Concat-based initial block's own MaxPool branch
    (FINNInitialBlockHAWQ, added 2026-08-28, e.g. 26_9_w24_ptq) --
    whichever it is, down1/down2 are always the LAST two by node index
    (any initial-block pool always precedes down1 entirely)."""

    maxpools = model.get_nodes_by_op_type("StreamingMaxPool")
    assert len(maxpools) in (2, 3), (
        "Expected 2 (down1, down2) or 3 (+ initial's own pool branch) "
        "StreamingMaxPool nodes, found %d. Topology may have changed -- "
        "update find_stage_boundaries()." % len(maxpools)
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

    # down1/down2 are always the LAST two maxpools -- any earlier one
    # (index 0 of 3) is the initial block's own pool branch, not a boundary.
    down1_start = _node_index(model, maxpools[-2])
    down2_start = _node_index(model, maxpools[-1])
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


def _shrink_boundary_past_sandwiched_nonhw(model, prev_boundary, boundary):
    """create_generic_partitions' convexity check forbids a non-HW node
    (partition_id -1, e.g. a leftover Transpose) from sitting BETWEEN two
    same-partition HW nodes -- that would make the partition depend on
    itself once extracted as its own subgraph ("cycle-free graph
    violated"). This can happen when a fixup transform (e.g.
    MoveTransposePastJoinConcat, used by the Concat-based initial block,
    2026-08-28) leaves a genuine trailing Transpose right after a
    StreamingConcat/etc. node that the naive index-threshold boundary
    would otherwise still count as part of the SAME partition as its
    successor. Fix: scan the [prev_boundary, boundary) range for the
    EARLIEST non-HW node that still has at least one more node before
    `boundary` (i.e. it would be sandwiched); if found, shrink `boundary`
    to right after it, so that node's successor(s) become the first
    node(s) of the NEXT partition instead -- turning it into a genuine
    inter-partition edge (allowed) rather than an internal one
    (forbidden)."""
    nodes = list(model.graph.node)
    for idx in range(prev_boundary, boundary):
        node = nodes[idx]
        if _is_fpgadataflow_node(node):
            continue
        # only a genuine sandwich if it has a REAL HW predecessor (via an
        # actual graph edge, not list-adjacency) already inside this same
        # partition's range -- a bare top-level graph-input tap (e.g. the
        # network's own input Transpose(s), which have no node predecessor
        # at all) must NOT trigger this.
        preds = model.find_direct_predecessors(node) or []
        has_hw_pred_in_range = any(
            _is_fpgadataflow_node(p) and prev_boundary <= _node_index(model, p) < idx
            for p in preds
        )
        if has_hw_pred_in_range and idx + 1 < boundary:
            return idx + 1
    return boundary


def _shrink_boundary_past_open_fork(model, prev_boundary, boundary):
    """FINN's per-partition stitching (CreateStitchedIP/CreateDataflowPartition)
    assumes exactly ONE stream crosses each partition boundary. A residual
    block's `DuplicateStreams` (fork: 1 input -> 2 outputs, one for the main
    conv path, one for the shortcut) is only reconverged into a single stream
    again by its matching `AddStreams` later on -- if a naive node-count cut
    (e.g. find_stage23_quarter_boundaries' even quartering) lands strictly
    BETWEEN a DuplicateStreams and its matching AddStreams, the departing
    partition ends with 2 dangling output tensors (both produced by the same
    DuplicateStreams node) instead of 1. This doesn't fail at partition-
    assignment time -- it silently produces a 2-output partition sub-model,
    then crashes much later, deep inside a ProcessPoolExecutor worker, as
    `AssertionError: No producer for output global_out` inside
    CreateStitchedIP's `step_set_fifo_depths` (its internal rtlsim-based FIFO
    depth search needs a stitched IP too), with NO partition/node identity in
    the traceback. Found 2026-09-01 on `8_2_relu_no_reg_w20`'s stage2/3
    quarter-1/quarter-2 cut (landed right after `DuplicateStreams_9`, before
    its corresponding AddStreams in the next quarter).

    Only safe to apply within a range that's already confirmed to have a net
    fork depth of 0 end-to-end (e.g. the whole stage2/3 quarter range) --
    NOT every DuplicateStreams in this network is closed by a literal
    AddStreams (found 2026-09-01: `up4`/`up5`'s own main/reduce fork and the
    network's very last fork before `final` reconverge some other way, e.g.
    fused into a later BatchNorm/Threshold during streamlining, not a plain
    binary add -- 30 DuplicateStreams vs only 27 AddStreams in this model).
    So if scanning forward never finds a closing AddStreams, this is NOT a
    boundary artifact -- give up and return the ORIGINAL boundary unchanged
    rather than asserting/crashing.

    Fix: track fork "depth" across [prev_boundary, boundary) --
    DuplicateStreams opens one (+1), AddStreams closes one (-1). If depth is
    nonzero at `boundary` (an unresolved fork would cross the cut), keep
    scanning forward past `boundary` until depth returns to 0 (i.e. until the
    matching AddStreams is found), then cut right after that node instead --
    keeping the whole fork+join pair inside the same partition."""
    nodes = list(model.graph.node)
    depth = 0
    for idx in range(prev_boundary, boundary):
        op_type = nodes[idx].op_type
        if op_type.startswith("DuplicateStreams"):
            depth += 1
        elif op_type.startswith("AddStreams"):
            depth -= 1
    if depth <= 0:
        return boundary
    idx = boundary
    while depth > 0:
        if idx >= len(nodes):
            print(
                "[_shrink_boundary_past_open_fork] WARNING: no closing "
                "AddStreams found for an open fork spanning boundary %d -- "
                "this fork apparently never closes via a literal AddStreams "
                "(not a boundary artifact), leaving boundary unchanged."
                % boundary
            )
            return boundary
        op_type = nodes[idx].op_type
        if op_type.startswith("DuplicateStreams"):
            depth += 1
        elif op_type.startswith("AddStreams"):
            depth -= 1
        idx += 1
    return idx


def validate_partition_single_output(parent_model):
    """Fail-fast structural check mirroring the EXACT assertions
    CreateStitchedIP itself makes on a partition's own raw sub-model
    (pre-specialize, as written by CreateDataflowPartition) --
    see finn/transformation/fpgadataflow/create_stitched_ip.py:
        for output in model.graph.output:
            assert model.find_producer(output.name) is not None, \
                "No producer for output " + output.name
        for input in model.graph.input:
            cons = model.find_consumers(input.name)
            assert cons != [], "No consumer for input " + input.name
            assert len(cons) == 1, "Multiple consumers for input " + input.name
    NOTE: a partition legitimately CAN have >1 graph input/output -- e.g.
    this network's very last DuplicateStreams fork (before `final`) is only
    reconverged by a non-HW `Mul` (dequant scale) living in the PARENT
    graph outside any partition, so the last HW partition genuinely ends
    with 2 real, valid outputs (verified 2026-09-01: both have in-partition
    producers). An earlier version of this check asserted a blanket
    `== 1` count for inputs/outputs, which is what CreateStitchedIP
    actually requires in the COMMON case but is stricter than the real
    invariant -- it false-positived on this exact legitimate multi-output
    partition. The true bug class (found 2026-09-01, `8_2_relu_no_reg_w20`
    8-way build) is a boundary cut landing mid-fork such that a declared
    output/input references a tensor whose true producer/consumer ended up
    in a DIFFERENT partition -- i.e. exactly the two conditions above, not
    a raw count. This crashes hours later, deep inside a
    ProcessPoolExecutor worker, as `AssertionError: No producer for output
    global_out` (`global_out`/`global_in` are GENERIC per-model names
    GiveReadableTensorNames assigns to output[0]/input[0] of ANY model it's
    applied to -- not literally the overall network's final tensor), with
    NO partition/node identity in the traceback. Call this immediately
    after step_create_dataflow_partition_multi, BEFORE dispatching the
    expensive (hours-long, per-partition Vivado) parallel build -- this
    check itself only loads small pre-specialize ONNX graphs, so it costs
    seconds, not hours, catching a boundary bug immediately instead of
    hours into synthesis."""
    from qonnx.core.modelwrapper import ModelWrapper

    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    bad = []
    for i, sdp_node in enumerate(sdp_nodes):
        fn = getCustomOp(sdp_node).get_nodeattr("model")
        sub_model = ModelWrapper(fn)
        problems = []
        for outp in sub_model.graph.output:
            if sub_model.find_producer(outp.name) is None:
                problems.append(f"output {outp.name}: no in-partition producer")
        for inp in sub_model.graph.input:
            n_cons = len(sub_model.find_consumers(inp.name))
            if n_cons != 1:
                problems.append(f"input {inp.name}: {n_cons} in-partition consumers (need 1)")
        if problems:
            bad.append((i, sdp_node.name, fn, problems))
    if bad:
        detail = "; ".join(f"partition {i} ({name}): {problems} (file={fn})" for i, name, fn, problems in bad)
        raise AssertionError(
            "validate_partition_single_output: %d of %d partitions violate "
            "CreateStitchedIP's producer/consumer invariant -- a partition "
            "boundary landed mid-fork (a declared input/output references a "
            "tensor produced/consumed in a DIFFERENT partition). This WILL "
            "crash CreateStitchedIP hours into the build if not fixed now. "
            "Offending partitions: %s" % (len(bad), len(sdp_nodes), detail)
        )
    print(
        "[validate_partition_single_output] OK: all %d partitions satisfy "
        "CreateStitchedIP's producer/consumer invariant" % len(sdp_nodes)
    )


def assign_stage_partition_ids(model, cfg=None):
    """Custom build step: sets the 'partition_id' nodeattr on every
    fpgadataflow node according to the 5-stage split described above.
    Must run after step_enet_convert_to_hw and before
    step_create_dataflow_partition_multi. `cfg` is accepted (and ignored)
    only so this matches the (model, cfg) step-function signature FINN's
    build_dataflow driver expects."""

    # step_enet_convert_to_hw's HW-inference transforms (e.g. InferConcatLayer,
    # needed by the real Concat-based initial block, 2026-08-28) don't always
    # leave model.graph.node in a valid topological order -- and the flat
    # index-threshold logic below silently assumes it is. Re-sort defensively
    # (same pattern every other order-sensitive FINN/qonnx step uses).
    model = model.transform(SortGraph())

    down1_start, down2_start, up4_start, up5_start = find_stage_boundaries(model)
    down1_start = _shrink_boundary_past_sandwiched_nonhw(model, 0, down1_start)
    down2_start = _shrink_boundary_past_sandwiched_nonhw(model, down1_start, down2_start)
    up4_start = _shrink_boundary_past_sandwiched_nonhw(model, down2_start, up4_start)
    up5_start = _shrink_boundary_past_sandwiched_nonhw(model, up4_start, up5_start)

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


def find_stage23_quarter_boundaries(down2_start, up4_start):
    """Splits the stage2/3 node-index range [down2_start, up4_start) --
    the single partition that caused the near-quadratic multi-day
    CreateStitchedIP blowup, see PARTITIONED_BUILD_LOG.md -- into 4
    contiguous quarters, as evenly as possible. Any remainder (range
    length not divisible by 4) is distributed to the EARLIEST quarters,
    so quarter 1 always starts at down2_start itself, i.e. it always
    keeps down2's StreamingMaxPool op even if that makes it 1 node
    bigger than the others (per spec: "first stage2 part may include the
    downsample even though it breaks symmetry"). Returns the 3 internal
    cut points (q2_start, q3_start, q4_start)."""
    total = up4_start - down2_start
    base, rem = divmod(total, 4)
    sizes = [base + 1 if i < rem else base for i in range(4)]
    cuts = []
    pos = down2_start
    for sz in sizes[:-1]:
        pos += sz
        cuts.append(pos)
    return cuts


def compute_8way_boundaries(model):
    """Single source of truth for the 8-way partition boundaries -- extracted
    2026-09-01 so `assign_stage_partition_ids_8way` (which actually splits
    the graph) and `load_all_partition_logical_names` in
    finn_ooc_..._8way_full.py (which maps HAWQ conv-order logical names onto
    partitions for folding-config generation) can NEVER diverge again. Found
    the hard way: `load_all_partition_logical_names` used to recompute
    boundaries via a bare `find_stage_boundaries` +
    `find_stage23_quarter_boundaries` call, WITHOUT the
    `_shrink_boundary_past_sandwiched_nonhw`/`_shrink_boundary_past_open_fork`
    fixes -- so once those fixes started actually moving down1_start/
    down2_start/up4_start/up5_start (2026-09-01's fork-shrink fix), the two
    boundary sets silently diverged (e.g. down1_start=7 here vs the real,
    shrunk 21 used by the actual split), corrupting the logical-name-to-
    weight-node positional mapping and crashing
    `build_partition_folding_config` with a node-count/name-count mismatch.
    `model` must be the SAME pre-partition checkpoint (post
    step_enet_convert_to_hw, pre step_create_dataflow_partition_multi) in
    both call sites for the result to be meaningful. Returns a dict with
    keys down1_start/down2_start/q2_start/q3_start/q4_start/up4_start/
    up5_start."""
    down1_start, down2_start, up4_start, up5_start = find_stage_boundaries(model)
    down1_start = _shrink_boundary_past_sandwiched_nonhw(model, 0, down1_start)
    down1_start = _shrink_boundary_past_open_fork(model, 0, down1_start)
    down2_start = _shrink_boundary_past_sandwiched_nonhw(model, down1_start, down2_start)
    down2_start = _shrink_boundary_past_open_fork(model, down1_start, down2_start)
    up4_start = _shrink_boundary_past_sandwiched_nonhw(model, down2_start, up4_start)
    up4_start = _shrink_boundary_past_open_fork(model, down2_start, up4_start)
    up5_start = _shrink_boundary_past_sandwiched_nonhw(model, up4_start, up5_start)
    up5_start = _shrink_boundary_past_open_fork(model, up4_start, up5_start)
    q2_start, q3_start, q4_start = find_stage23_quarter_boundaries(down2_start, up4_start)
    q2_start = _shrink_boundary_past_sandwiched_nonhw(model, down2_start, q2_start)
    q2_start = _shrink_boundary_past_open_fork(model, down2_start, q2_start)
    q3_start = _shrink_boundary_past_sandwiched_nonhw(model, q2_start, q3_start)
    q3_start = _shrink_boundary_past_open_fork(model, q2_start, q3_start)
    q4_start = _shrink_boundary_past_sandwiched_nonhw(model, q3_start, q4_start)
    q4_start = _shrink_boundary_past_open_fork(model, q3_start, q4_start)
    return {
        "down1_start": down1_start, "down2_start": down2_start,
        "q2_start": q2_start, "q3_start": q3_start, "q4_start": q4_start,
        "up4_start": up4_start, "up5_start": up5_start,
    }


def assign_stage_partition_ids_8way(model, cfg=None):
    """8-way variant of assign_stage_partition_ids: same initial/stage1/
    stage4/stage5 boundaries as the 5-way split, but stage2/3 is further
    divided into 4 roughly-equal quarters (partition_id 2..5), so stage4
    and stage5 shift to partition_id 6 and 7 respectively:

        partition 0  initial
        partition 1  stage1   = down1 + regular1
        partition 2  stage2/3 quarter 1 (includes down2's StreamingMaxPool)
        partition 3  stage2/3 quarter 2
        partition 4  stage2/3 quarter 3
        partition 5  stage2/3 quarter 4
        partition 6  stage4   = up4 + regular4
        partition 7  stage5   = up5 + regular5 + final

    `cfg` accepted (and ignored) only so this matches the (model, cfg)
    step-function signature FINN's build_dataflow driver expects (same as
    assign_stage_partition_ids)."""

    # see assign_stage_partition_ids' matching comment: re-sort defensively,
    # convert_to_hw's HW-inference transforms don't guarantee topological
    # node order is preserved (surfaced by the new Concat-based initial
    # block's two parallel branches, 2026-08-28).
    model = model.transform(SortGraph())

    boundaries = compute_8way_boundaries(model)
    down1_start = boundaries["down1_start"]
    down2_start = boundaries["down2_start"]
    q2_start = boundaries["q2_start"]
    q3_start = boundaries["q3_start"]
    q4_start = boundaries["q4_start"]
    up4_start = boundaries["up4_start"]
    up5_start = boundaries["up5_start"]

    counts = [0] * 8
    n_skipped = 0
    for idx, node in enumerate(model.graph.node):
        if idx < down1_start:
            pid = 0
        elif idx < down2_start:
            pid = 1
        elif idx < q2_start:
            pid = 2
        elif idx < q3_start:
            pid = 3
        elif idx < q4_start:
            pid = 4
        elif idx < up4_start:
            pid = 5
        elif idx < up5_start:
            pid = 6
        else:
            pid = 7
        counts[pid] += 1
        if not _is_fpgadataflow_node(node):
            n_skipped += 1
            continue
        inst = getCustomOp(node)
        inst.set_nodeattr("partition_id", pid)

    print(
        "[assign_stage_partition_ids_8way] boundaries down1=%d down2=%d "
        "q2=%d q3=%d q4=%d up4=%d up5=%d"
        % (down1_start, down2_start, q2_start, q3_start, q4_start, up4_start, up5_start)
    )
    print(
        "[assign_stage_partition_ids_8way] node counts per partition (0..7): %s "
        "(total=%d, skipped non-HW=%d)"
        % (counts, len(model.graph.node), n_skipped)
    )
    return model

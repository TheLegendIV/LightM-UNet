"""Per-block partitioning of the S12 net: one FINN partition per analytical block (initial, down1, regular1.0, ..., stage3.7, up4, ..., final) instead of the 8-way
stage split, so that every distinct block shape can be built, folded (the analytical / MILP folding json, unchanged) and rtlsim'd on its own.

Used by finn_s12_preamble.py --blocks (adds `assign_block_partition_ids` as the partition-tagging step) and finn_s12_build.py --blocks (reads block_partitions.json).

How the cuts are found (structural, on the topologically sorted pre-partition graph; no hard-coded node indices):
  * anchors = the weight-bearing nodes (MVAU / VVAU / MaxPool) matched positionally to conv_order.json (finn_s12_build_steps.match_conv_order_to_nodes, the same
    matcher the 8-way bridge uses). A conv_order name belongs to a block: `regular1.0.reduce.0` -> `regular1.0`, `stage3.7.conv` -> `stage3.7`, `down1.shortcut_pool` -> `down1`,
    `up4.main_proj.0` -> `up4`, `initial.pool` -> `initial`, `final` -> `final`;
  * a block ends right after the Thresholding that consumes its join (AddStreams; StreamingConcat for the initial block) = the block's out_act. The cut must be a single-stream
    cut: exactly one tensor crosses it (checked, fail-fast). Everything up to the first anchor of the next block belongs to the previous one, so the next block's DuplicateStreams
    / padding start the next partition.
"""
from __future__ import annotations

import json
import os

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import SortGraph
from qonnx.util.basic import get_by_name

CONV_ORDER = None                      # path of <model>_conv_order.json, set by finn_s12_preamble.py --blocks before the step runs
JOIN_PREFIXES = ("AddStreams", "StreamingConcat")
PASS_PREFIXES = ("StreamingFIFO", "StreamingDataWidthConverter")
BLOCK_JSON = "block_partitions.json"


def stage_of(logical_name: str) -> str:
    """conv_order logical name -> block name (the analytical / MILP stage name)."""
    p = logical_name.split(".")
    if len(p) == 1 or p[0].startswith(("down", "up", "initial", "final")):
        return p[0]
    return ".".join(p[:2])


def block_kind(stage: str) -> str:
    if stage == "initial":
        return "init"
    if stage == "final":
        return "final"
    if stage.startswith("down"):
        return "dn"
    if stage.startswith("up"):
        return "up"
    return "reg"


def _is_hw(node) -> bool:
    backend = get_by_name(node.attribute, "backend")
    return backend is not None and backend.s.decode("UTF-8") == "fpgadataflow"


def crossing_tensors(model, p: int) -> set:
    """Tensors produced by nodes[0..p] (or graph inputs) and consumed by nodes[p+1..] or leaving the graph: the streams a cut after node p would have to carry."""
    nodes = list(model.graph.node)
    out = set()
    last_cons = {}
    for i, n in enumerate(nodes):
        for t in n.input:
            last_cons[t] = i
    for o in model.graph.output:
        last_cons[o.name] = len(nodes)
    produced = {}
    for i, n in enumerate(nodes):
        for t in n.output:
            produced[t] = i
    for t, c in last_cons.items():
        if model.get_initializer(t) is not None:
            continue
        pi = produced.get(t, -1)            # graph inputs count as produced before node 0
        if pi <= p < c:
            out.add(t)
    return out


def _follow_out_act(model, nodes, join_idx: int) -> int:
    """Index of the block's last node: from the join, pass through FIFO / DWC nodes and ONE Thresholding (the out_act)."""
    cur, took_thr = join_idx, False
    while True:
        outs = nodes[cur].output
        cons = [c for t in outs for c in (model.find_consumers(t) or [])]
        if len(cons) != 1:
            return cur
        c = cons[0]
        if c.op_type.startswith(PASS_PREFIXES):
            cur = nodes.index(c)
        elif c.op_type.startswith("Thresholding") and not took_thr:
            took_thr, cur = True, nodes.index(c)
        else:
            return cur


def find_block_ranges(model, anchors: list) -> list:
    """anchors = [(node_idx, stage)] in node order. Returns [dict(stage, start, end, cut_tensor)] with contiguous node index ranges [start, end) covering the whole graph.
    Fail-fast on every structural assumption (stage anchors contiguous, one join per block, one stream per cut)."""
    nodes = list(model.graph.node)
    stages, first, last = [], {}, {}
    for idx, st in anchors:
        if st not in first:
            first[st] = idx
            stages.append(st)
        elif stages[-1] != st:
            raise RuntimeError(f"block {st}: its weight nodes are not contiguous in the graph (anchor at node {idx} after block {stages[-1]})")
        last[st] = idx
    ranges, start = [], 0
    for k, st in enumerate(stages):
        if k == len(stages) - 1:
            ranges.append(dict(stage=st, start=start, end=len(nodes), cut_tensor=None))
            break
        nxt = stages[k + 1]
        joins = [i for i in range(first[st], first[nxt]) if nodes[i].op_type.startswith(JOIN_PREFIXES)]
        if not joins:
            raise RuntimeError(f"block {st}: no AddStreams / StreamingConcat between its first weight node ({first[st]}) and the next block's ({first[nxt]})")
        end_idx = _follow_out_act(model, nodes, joins[-1])
        if not (last[st] <= end_idx < first[nxt]):
            raise RuntimeError(f"block {st}: last node {end_idx} ({nodes[end_idx].name}) not between its last weight node ({last[st]}) and block {nxt}'s first ({first[nxt]})")
        cross = crossing_tensors(model, end_idx)
        if len(cross) != 1:
            raise RuntimeError(f"block {st}/{nxt}: {len(cross)} streams cross the cut after node {end_idx} ({nodes[end_idx].name}): {sorted(cross)} -- not a single-stream cut")
        ranges.append(dict(stage=st, start=start, end=end_idx + 1, cut_tensor=next(iter(cross))))
        start = end_idx + 1
    return ranges


def block_anchors(model, conv_order_file: str):
    """([(node_idx, stage)], {node_idx: conv_order entry}) for the sorted pre-partition graph."""
    from finn_s12_build_steps import match_conv_order_to_nodes
    weight_like_idx, node_idx_to_entry = match_conv_order_to_nodes(model, conv_order_file)
    return [(i, stage_of(node_idx_to_entry[i]["logical_name"])) for i in weight_like_idx], node_idx_to_entry


def describe_ranges(ranges: list, node_idx_to_entry: dict) -> list:
    out = []
    for pid, r in enumerate(ranges):
        entries = [e for i, e in sorted(node_idx_to_entry.items()) if r["start"] <= i < r["end"]]
        out.append(dict(
            id=pid, stage=r["stage"], kind=block_kind(r["stage"]), node_range=[r["start"], r["end"]], cut_tensor=r["cut_tensor"],
            logical_convs=[e["logical_name"] for e in entries if "MaxPool" not in e["module_type"]],
            logical_pools=[e["logical_name"] for e in entries if "MaxPool" in e["module_type"]],
        ))
    return out


def assign_block_partition_ids(model, cfg=None):
    """Build step (replaces assign_stage_partition_ids_8way): tag every fpgadataflow node with the partition id of its block and write block_partitions.json next to the checkpoints."""
    assert CONV_ORDER, "finn_s12_blocks.CONV_ORDER must be set (finn_s12_preamble.py --blocks --conv-order ...)"
    model = model.transform(SortGraph())
    anchors, node_idx_to_entry = block_anchors(model, CONV_ORDER)
    ranges = find_block_ranges(model, anchors)
    desc = describe_ranges(ranges, node_idx_to_entry)
    nodes = list(model.graph.node)
    for d in desc:
        n_hw = 0
        for idx in range(*d["node_range"]):
            if _is_hw(nodes[idx]):
                getCustomOp(nodes[idx]).set_nodeattr("partition_id", d["id"])
                n_hw += 1
        d["n_hw_nodes"] = n_hw
        print(f"[blocks] partition {d['id']:2d} {d['stage']:11s} nodes {d['node_range']} ({n_hw} HW), convs {len(d['logical_convs'])}, pools {len(d['logical_pools'])}")
    if cfg is not None and getattr(cfg, "output_dir", None):
        ckpt = os.path.join(cfg.output_dir, "intermediate_models")
        os.makedirs(ckpt, exist_ok=True)
        with open(os.path.join(ckpt, BLOCK_JSON), "w") as f:
            json.dump(desc, f, indent=2)
    return model


def load_block_partitions(preamble_dir: str) -> list:
    with open(os.path.join(preamble_dir, "intermediate_models", BLOCK_JSON)) as f:
        return json.load(f)


def resolve_partitions(blocks: list, wanted: list, shapes_file: str | None = None) -> list:
    """Partition ids for a --partitions list made of ints, block names, or the token `shapes` (one representative per distinct block shape, from block_shapes.json)."""
    by_stage = {b["stage"]: b["id"] for b in blocks}
    ids = []
    for w in wanted:
        if w == "shapes":
            assert shapes_file, "--partitions shapes needs --block-shapes block_shapes.json"
            with open(shapes_file) as f:
                shapes = json.load(f)
            ids += [by_stage[s["representative"]] for s in shapes["shapes"]]
        elif w in by_stage:
            ids.append(by_stage[w])
        else:
            ids.append(int(w))
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out

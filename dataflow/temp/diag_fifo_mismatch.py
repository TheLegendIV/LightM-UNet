"""Cross-reference a partition's fifo_force_report (what actually got applied to the
stitched IP) against its fifo_plan's "wanted" dict (what the MILP analytical model
expects), to find every FIFO where the two disagree -- unresolved roles, role pairs
that resolved but didn't match any "wanted" key (naming/edge-direction mismatch), or
roles that matched but got a different depth than the MILP wanted.
"""
import json
import os
import sys

DIAG_DIR = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(DIAG_DIR, "p5_p6_diag", name)) as f:
        return json.load(f)


def check_partition(idx):
    force_report = load(f"fifo_force_report_partition_{idx}.json")
    plan = load(f"fifo_plan_partition{idx}.json")
    wanted = plan["wanted"]
    # wanted is keyed "producer_role=>consumer_role" but may have duplicate keys
    # pointing at the same edge (e.g. "...=>skip_quant" / "...=>thr_s" aliases) --
    # build a lookup from (producer_role, consumer_role) -> depth using each entry's
    # own producer/consumer fields (authoritative), not just the dict key text.
    wanted_by_pair = {}
    for key, entry in wanted.items():
        wanted_by_pair[(entry["producer"], entry["consumer"])] = entry

    print(f"\n=== partition {idx} ===")
    n_null_producer = 0
    n_null_consumer = 0
    n_unmatched_pair = 0
    n_depth_mismatch = 0
    n_ok = 0
    n_boundary = 0
    for e in force_report:
        prole, crole = e["producer_role"], e["consumer_role"]
        if prole is None and crole is not None:
            n_boundary += 1
            continue
        if prole is None:
            n_null_producer += 1
            print(f"  NULL PRODUCER ROLE: {e['fifo']} producer_node={e['producer_node']} "
                  f"consumer_node={e['consumer_node']} forced_depth={e['forced_depth']}")
            continue
        if crole is None:
            n_null_consumer += 1
            print(f"  NULL CONSUMER ROLE: {e['fifo']} producer_node={e['producer_node']} "
                  f"consumer_node={e['consumer_node']} forced_depth={e['forced_depth']}")
            continue
        pair = (prole, crole)
        if pair not in wanted_by_pair:
            n_unmatched_pair += 1
            print(f"  NO WANTED ENTRY FOR ROLE PAIR: {e['fifo']} {prole} => {crole} "
                  f"forced_depth={e['forced_depth']}")
            continue
        wanted_depth = wanted_by_pair[pair]["depth"]
        if e["forced_depth"] != wanted_depth:
            n_depth_mismatch += 1
            print(f"  DEPTH MISMATCH: {e['fifo']} {prole} => {crole} "
                  f"forced_depth={e['forced_depth']} wanted_depth={wanted_depth}")
            continue
        n_ok += 1
    print(f"  summary: ok={n_ok} boundary(cross-partition)={n_boundary} "
          f"null_producer={n_null_producer} null_consumer={n_null_consumer} "
          f"unmatched_pair={n_unmatched_pair} depth_mismatch={n_depth_mismatch} "
          f"total={len(force_report)}")


if __name__ == "__main__":
    idxs = [int(a) for a in sys.argv[1:]] or [5, 6]
    for idx in idxs:
        check_partition(idx)

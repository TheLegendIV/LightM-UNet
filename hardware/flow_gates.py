"""Build-flow gates (container side; pure onnx, no FINN import needed).

Flow: export -> accuracy gate (verify_export.py) -> preamble (+ accuracy gate on step_enet_streamline)
      -> folding landed -> build -> FIFO sizes landed.

  gate_folding_landed(model, folding_file, label)   in-build, right after step_apply_folding_config
  gate_fifos_landed(model, report, label)           in-build, right after step_force_fifo_depths_from_milp
  python3 flow_gates.py fifos <output_dir>          re-check a finished build from its checkpoints
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WEIGHT_PREFIXES = ("MVAU", "VVAU")


def _attrs(node) -> dict:
    out = {}
    for a in node.attribute:
        if a.type == 2:      # INT
            out[a.name] = a.i
        elif a.type == 3:    # STRING
            out[a.name] = a.s.decode()
    return out


def gate_folding_landed(model, folding_file: str, label: str) -> None:
    """Every entry of the bridged folding config must be on its node, and no MVAU/VVAU may be left unfolded."""
    cfg = json.load(open(folding_file))
    nodes = {n.name: n for n in model.graph.node}
    errors = []
    for name, entry in cfg.items():
        if name == "Defaults":
            continue
        node = nodes.get(name)
        if node is None:
            errors.append(f"{name}: node not in model")
            continue
        have = _attrs(node)
        for key, want in entry.items():
            if key in have and have[key] != want:
                errors.append(f"{name}.{key}: config {want} != landed {have[key]}")
    unfolded = [n.name for n in model.graph.node if n.op_type.startswith(WEIGHT_PREFIXES) and n.name not in cfg]
    errors += [f"{n}: weight node has no folding entry (FINN auto-fold stayed)" for n in unfolded]
    print(f"{label} folding landed: {len(cfg) - 1} entries checked, {len(errors)} error(s)", flush=True)
    assert not errors, f"{label} folding did not land:\n  " + "\n  ".join(errors)


def gate_fifos_landed(model, report: list[dict], label: str) -> dict:
    """Each forced FIFO must exist and carry its forced depth; unforced FIFOs keep FINN's stock depth by design (reported only)."""
    nodes = {n.name: n for n in model.graph.node}
    bad, interior = [], []
    for e in report:
        node = nodes.get(e["fifo"])
        if node is None:
            bad.append(f"{e['fifo']}: FIFO missing from model")
        elif e["forced_depth"] is not None and _attrs(node).get("depth") != e["forced_depth"]:
            bad.append(f"{e['fifo']}: depth {_attrs(node).get('depth')} != forced {e['forced_depth']}")
        elif e["forced_depth"] is None and e["producer_node"] and e["consumer_node"]:
            interior.append(f"{e['fifo']}: {e['producer_node']} -> {e['consumer_node']}")
    summary = dict(n_fifos=len(report), n_forced=sum(e["forced_depth"] is not None for e in report),
                   n_interior_unassigned=len(interior), depth_mismatches=bad, interior_unassigned=interior)
    print(f"{label} fifos landed: {summary['n_forced']}/{summary['n_fifos']} forced, "
          f"{len(interior)} interior left at stock depth (info), {len(bad)} depth mismatch(es)", flush=True)
    assert not bad, f"{label} FIFO depths did not land:\n  " + "\n  ".join(bad)
    return summary


def _check_build_dir(out_dir: Path) -> None:
    import onnx

    class _M:  # minimal ModelWrapper stand-in
        def __init__(self, path):
            self.graph = onnx.load(str(path)).graph

    reports = sorted(out_dir.rglob("fifo_force_report_partition_*.json"))
    assert reports, f"no fifo_force_report_partition_*.json under {out_dir}"
    failed = []
    for rep in reports:
        idx = rep.stem.rsplit("_", 1)[1]
        ckpt = next(iter(sorted(out_dir.rglob(f"partition_{idx}_postfifo_autosize.onnx"))), None) \
            or next(iter(sorted(out_dir.rglob(f"partition{idx}_*_postfifo_autosize.onnx"))), None)
        assert ckpt is not None, f"no post-FIFO checkpoint for partition {idx}"
        try:
            gate_fifos_landed(_M(ckpt), json.load(open(rep)), f"[partition {idx}]")
        except AssertionError as e:
            failed.append(str(e))
    if failed:
        raise AssertionError("\n".join(failed))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["fifos"])
    p.add_argument("output_dir")
    a = p.parse_args()
    try:
        _check_build_dir(Path(a.output_dir))
        print("FIFO GATE OK")
    except AssertionError as e:
        print(f"FIFO GATE FAILED: {e}", file=sys.stderr)
        sys.exit(1)

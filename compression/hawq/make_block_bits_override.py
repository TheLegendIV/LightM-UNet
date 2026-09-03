"""Step 3 of the stuck-QAT debugging plan: produces a hand-overridden
per-block bit-assignment JSON by patching a small number of blocks in an
existing block_bits_*.json file, keeping every other block exactly as the
base file chose it.

WHY A SEPARATE FILE, NOT AN INLINE DICT PATCH: every other bit assignment
in this repo is a checked-in, diffable block_bits_*.json (compression/hawq/).
Hand-patching a dict inline in a job's Python heredoc would make the one
experiment most likely to isolate the stuck-QAT root cause non-reproducible
and impossible to review/diff -- breaking that convention here isn't worth
the shortcut.

Usage:
    python compression/hawq/make_block_bits_override.py \\
        --base compression/hawq/artifacts/block_bits_s19_acc2x.json \\
        --override regular5.0.weight_bits=8 \\
        --out compression/hawq/artifacts/block_bits_s19_acc2x_regular5_0_w8.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VALID_AXES = ("weight_bits", "act_bits")
AXIS_TO_KEY = {"weight_bits": "stage_weight_bits", "act_bits": "stage_act_bits"}


def parse_override(spec: str) -> tuple[str, str, int]:
    """'<block_name>.<weight_bits|act_bits>=<int>' -> (block_name, axis, bits)."""
    if "=" not in spec:
        raise ValueError(f"--override {spec!r} must be '<block_name>.<weight_bits|act_bits>=<int>'.")
    lhs, rhs = spec.split("=", 1)
    if "." not in lhs:
        raise ValueError(f"--override {spec!r}: left side must be '<block_name>.<weight_bits|act_bits>'.")
    block_name, axis = lhs.rsplit(".", 1)
    if axis not in VALID_AXES:
        raise ValueError(f"--override {spec!r}: axis must be one of {VALID_AXES}, got {axis!r}.")
    try:
        bits = int(rhs)
    except ValueError:
        raise ValueError(f"--override {spec!r}: right side must be an integer bit-width, got {rhs!r}.") from None
    return block_name, axis, bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="Existing block_bits_*.json to patch.")
    parser.add_argument(
        "--override", action="append", required=True, dest="overrides",
        help="'<block_name>.<weight_bits|act_bits>=<int>', repeatable for multiple overrides.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with open(args.base) as f:
        base = json.load(f)

    result = {
        "stage_weight_bits": dict(base["stage_weight_bits"]),
        "stage_act_bits": dict(base["stage_act_bits"]),
    }

    applied = []
    for spec in args.overrides:
        block_name, axis, bits = parse_override(spec)
        key = AXIS_TO_KEY[axis]
        if block_name not in result[key]:
            raise ValueError(f"--override {spec!r}: block {block_name!r} not found in {args.base}'s {key!r}.")
        old_bits = result[key][block_name]
        result[key][block_name] = bits
        applied.append((block_name, axis, old_bits, bits))
        print(f"  {block_name}.{axis}: {old_bits} -> {bits}")

    result["_diagnostics"] = {
        **base.get("_diagnostics", {}),
        "override_note": (
            f"Hand-overridden via make_block_bits_override.py from {args.base} -- "
            f"{len(applied)} block(s) patched: "
            + ", ".join(f"{b}.{a} {o}->{n}" for b, a, o, n in applied)
            + ". Every other block is UNCHANGED from the base file -- this is a single-variable "
              "ablation, not a fresh search result."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out} ({len(applied)} block(s) overridden from {args.base}).")


if __name__ == "__main__":
    main()

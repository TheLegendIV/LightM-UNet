"""Group the 29 blocks of a folding json by SHAPE (block kind + geometry + every node's PE / SIMD): one hardware build per group is enough, since blocks of one shape are the same hardware.

    python3 hardware/builds/S12_dense_256_u4_analytical_v1/blocks/make_block_shapes.py \\
        [--folding MILP/artifacts/S12_dense_256_u4_analytical_v1/int6_fps250_lat200/layer_bits_folding_final.json] [--out .../blocks/block_shapes.json]

Run in lightmunet_dev (needs the MILP stack; same grouping key as MILP/analytical/net_explicit.verify_blocks_keep). Writes block_shapes.json:
{"folding": ..., "shapes": [{"id": 0, "kind": "init", "representative": "initial", "members": ["initial"], "node_folds": {...}}, ...]}
The representative is the first member in dataflow order (the one finn_s12_build.py --blocks --partitions shapes builds).
"""
import argparse
import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for p in (REPO / "enet", REPO / "MILP", REPO / "MILP" / "utils", REPO / "MILP" / "analytical"):
    sys.path.insert(0, str(p))

import finn_milp  # noqa: E402
import net_explicit  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folding", type=Path, default=REPO / "MILP/artifacts/S12_dense_256_u4_analytical_v1/int6_fps250_lat200/layer_bits_folding_final.json")
    ap.add_argument("--out", type=Path, default=HERE / "block_shapes.json")
    ap.add_argument("--config", default="config_12_dense_relu_nearest_upsample_256")
    ap.add_argument("--bits", type=int, default=6)
    a = ap.parse_args()
    d = json.loads(a.folding.read_text())
    finn_milp.load_config(a.config)
    finn_milp.CANDIDATE_BITS = tuple(sorted(set(finn_milp.CANDIDATE_BITS) | {a.bits}))
    _model, geoms, extras, _pred, dmap, kinds = finn_milp.build_model_and_graph()
    ctx = (geoms, extras, {g.name: g for g in geoms}, {n.geom.name: n for n in extras}, dmap, kinds)
    blocks, _ = net_explicit.build_blocks(d, types.SimpleNamespace(bits=a.bits), ctx)
    groups: dict = {}
    for stage, kind, r in blocks:
        key = repr((kind, sorted(r.params.items()), [(n.name, n.pe, n.simd) for n in r.nodes]))
        g = groups.setdefault(key, dict(kind=kind, members=[], node_folds={n.name: [n.pe, n.simd] for n in r.nodes}))
        g["members"].append(stage)
    shapes = [dict(id=i, kind=g["kind"], representative=g["members"][0], members=g["members"], node_folds=g["node_folds"]) for i, g in enumerate(groups.values())]
    a.out.write_text(json.dumps(dict(folding=str(a.folding.relative_to(REPO)), n_blocks=len(blocks), shapes=shapes), indent=1))
    print(f"{len(blocks)} blocks -> {len(shapes)} shapes -> {a.out}")
    for s in shapes:
        print(f"  shape {s['id']:2d} {s['kind']:5s} build {s['representative']:11s} also: {', '.join(s['members'][1:]) or '-'}")


if __name__ == "__main__":
    main()

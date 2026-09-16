"""One-off: check the pooled-fit A-only imbalance model (mvu_lut UNCHANGED,
c0=300/c1=1.1; imbalance_luts = (184.2061*(PE/SIMD) + 338.1787*(MH/MW)) * A)
against the 6 small S12-context probe builds in hardware/results.csv --
these are real OOC-synthesized 2-bottleneck networks, much smaller scale
than the full 8-way production partitions the imbalance term was actually
fit on, so this checks whether it generalizes DOWN in scale too.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "compression" / "hawq"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finn_export_probe_s12_context_common import ProbeNet, dump_probe_geometry, INPUT_HW  # noqa: E402
from finn_cost_model import max_simd  # noqa: E402

K_PE, K_MH = 184.2061, 338.1787  # pooled dense+separable fit, A-only


def baseline_and_imbalance(geom, pe: int, simd: int, w: int, a: int) -> tuple[float, float]:
    mw = max_simd(geom)
    mh = geom.cout
    addertree_luts = (w + a) * (2 * simd - 1)
    alpha = math.log2(mw) + w + a - 2
    acc_luts = min(32, alpha + math.log2(1 + 2 ** -alpha) + 1)
    baseline = 426 + 300 + 1.1 * pe * (addertree_luts + acc_luts)
    imbalance = (K_PE * (pe / simd) + K_MH * (mh / mw)) * a
    return baseline, imbalance


# (label, separable_dilated, weight_bits, act_bits, pe_mode, real_LUT)
PROBES = [
    ("dense_int4", False, 4, 4, "pe1", 6174.0),
    ("separable_int4", True, 4, 4, "pe1", 7377.0),
    ("dense_int6", False, 6, 6, "pe1", 7378.0),
    ("separable_int6", True, 6, 6, "pe1", 9089.0),
    ("dense_int8_pemh", False, 8, 8, "pemh", 15252.0),
    ("separable_int8_pemh", True, 8, 8, "pemh", 20560.0),
]

print(f"{'probe':24s} {'old(no imb)':>12s} {'new(+imb)':>12s} {'real':>10s} {'real/old':>9s} {'real/new':>9s}")
for label, separable, w, a, pe_mode, real in PROBES:
    torch.manual_seed(0)
    model = ProbeNet(separable, w, a)
    geometries = dump_probe_geometry(model, INPUT_HW)

    old_total, new_total = 0.0, 0.0
    for g in geometries:
        pe = g.cout if pe_mode == "pemh" else 1
        simd = 1
        baseline, imbalance = baseline_and_imbalance(g, pe, simd, w, a)
        old_total += baseline
        new_total += baseline + imbalance

    print(f"{label:24s} {old_total:12.1f} {new_total:12.1f} {real:10.1f} "
          f"{real/old_total:9.3f} {real/new_total:9.3f}")

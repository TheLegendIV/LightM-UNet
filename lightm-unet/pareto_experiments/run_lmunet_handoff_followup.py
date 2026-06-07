from __future__ import annotations

import run_lmunet_pareto as pareto


pareto.EXPERIMENTS = [
    {
        "id": "B0",
        "name": "baseline",
        "channels": (12, 20, 32, 44, 64, 72),
        "edge_channels": 20,
        "hypothesis": "baseline reference from the first Pareto sweep",
    },
    {
        "id": "M0",
        "name": "middle_original",
        "channels": (16, 32, 56, 56, 72, 72),
        "edge_channels": 20,
        "hypothesis": "winning middle-capacity pattern with equal stage3-to-stage4 width",
    },
    {
        "id": "H1",
        "name": "pre_mamba_wide_only",
        "channels": (16, 32, 56, 44, 72, 72),
        "edge_channels": 20,
        "hypothesis": "tests whether widening the last conv stage alone explains the M0 gain",
    },
    {
        "id": "H2",
        "name": "mamba_entry_wide_only",
        "channels": (16, 32, 44, 56, 72, 72),
        "edge_channels": 20,
        "hypothesis": "tests whether widening the first PV-Mamba stage alone explains the M0 gain",
    },
    {
        "id": "H3",
        "name": "smooth_handoff_52",
        "channels": (16, 32, 52, 52, 72, 72),
        "edge_channels": 20,
        "hypothesis": "tests whether a smoother equal-width conv-to-Mamba handoff still helps at lower width",
    },
    {
        "id": "DS0",
        "name": "downscaled_smooth_handoff",
        "channels": (12, 24, 40, 40, 56, 56),
        "edge_channels": 20,
        "hypothesis": "downscaled version of the smooth handoff pattern starting at 12 channels",
    },
]


if __name__ == "__main__":
    raise SystemExit(pareto.main())


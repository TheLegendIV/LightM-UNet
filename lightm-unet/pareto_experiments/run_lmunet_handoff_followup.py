from __future__ import annotations

import run_lmunet_pareto as pareto


pareto.EXPERIMENTS = [
    {
        "id": "DS0",
        "name": "downscaled_smooth_handoff",
        "channels": (12, 24, 40, 40, 56, 56),
        "edge_channels": 20,
        "hypothesis": "downscaled smooth handoff pattern from previous follow-up",
    },
    {
        "id": "DS1",
        "name": "downscaled_smooth_handoff_plus",
        "channels": (12, 24, 44, 44, 60, 60),
        "edge_channels": 20,
        "hypothesis": "slightly wider downscaled smooth handoff near baseline parameter count",
    },
    {
        "id": "DS2",
        "name": "early_smooth_downscaled",
        "channels": (16, 28, 44, 44, 60, 60),
        "edge_channels": 20,
        "hypothesis": "downscaled smooth handoff with a stronger early-stage start at 16 channels",
    },
    {
        "id": "DC1",
        "name": "pre_mamba_control",
        "channels": (12, 24, 44, 36, 60, 60),
        "edge_channels": 20,
        "hypothesis": "control: wide last conv stage but narrower first PV-Mamba stage",
    },
    {
        "id": "DC2",
        "name": "mamba_entry_control",
        "channels": (12, 24, 36, 44, 60, 60),
        "edge_channels": 20,
        "hypothesis": "control: narrower last conv stage but wide first PV-Mamba stage",
    },
]


if __name__ == "__main__":
    raise SystemExit(pareto.main())

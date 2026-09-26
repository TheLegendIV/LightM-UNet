"""Single source of truth for the thesis's NAS-narrative stages, plus the
"harvest results.csv / report gaps" job.

The raw `results.csv` `stage` column does NOT line up 1:1 with the thesis
narrative -- e.g. raw stage `4_arch_probes` has 11 probes, only 7 of which
are about context-block topology; the narrative's Stage 4 spans 4 different
raw `stage` values (`12_dense_relu`, `12_separable_dense_relu`,
`8_2_relu_no_reg_fullwidth`, `9_dsc_projected_dense_dilation`). So stages
here are defined by an explicit `config_name` allowlist, not by filtering on
the raw `stage` column.

Deliberately excluded from every stage below (not silently dropped --
this is the documented boundary of the narrative): `3_transfer_original`
(warm-start experiment, orthogonal to this search); every reg_interleaved/
separable follow-on family (raw stages `5_arch_probe_pairs` through `27_*`)
-- none were carried forward to the final pick; and every quantization/QAT/
PTQ/HAWQ derivative row (blank `abbrev`, `joint_alpha*`/`uniform_int*`/
`ptq*`/`qat*` suffixes) -- that's the later quantization chapter's material,
not this one.

Usage (standalone gap-check job):
    python analysis/report/model_search/nas_stages.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_CSV = REPO_ROOT / "compression" / "results.csv"
REPORT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# House style, lifted from plot_dice_vs_cost.py so the notebook and any
# future script share one definition.
# ---------------------------------------------------------------------------
INK, SECONDARY_INK, MUTED, GRID, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb",
)
NAIVE_COLOR = "#0061d6"
# Same 5-color set as compression/analysis/qat_results/plot_forcedsp_lut70_
# alpha_sweep.py's ALPHA_COLORS -- reused everywhere a small fixed set of
# named series needs its own color.
SERIES_COLORS = ["#0061d6", "#00a78c", "#eb4300", "#c41500", "#5600bf", "#8a8a00", "#008a8a"]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors=INK, labelsize=9)


def style_title_and_labels(ax, title: str, xlabel: str, ylabel: str = "Dice") -> None:
    ax.set_xlabel(xlabel, color=SECONDARY_INK, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, color=SECONDARY_INK, fontsize=10, fontweight="bold")
    ax.set_title(title, color=INK, fontsize=10.5, fontweight="bold")


def style_legend(ax, loc: str = "lower right") -> None:
    ax.legend(loc=loc, frameon=True, facecolor=SURFACE, edgecolor="#c3c2b7",
              fontsize=8.5, labelcolor=SECONDARY_INK)


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------
STAGES: dict[str, dict] = {
    "stage_1": {
        "title": "Naive compression baseline",
        "description": (
            "Uniform channel-width divisors of the ENet-paper baseline "
            "(16,64,128,64,16): Original/U2/U4/U8/U16. All ReLU "
            "(prelu=0), upsample_conv (bilinear) decoder, "
            "context_pattern=default."
        ),
        "config_names": [
            "nnUNetTrainerENet_1_naive_baseline_Baseline",
            "nnUNetTrainerENet_1_naive_baseline_U2",
            "nnUNetTrainerENet_1_naive_baseline_U4",
            "nnUNetTrainerENet_1_naive_baseline_U8",
            "nnUNetTrainerENet_1_naive_baseline_U16",
            # ENet-paper-faithful companion curve (PReLU + max_unpool,
            # same 5-point channel grid) -- queued via
            # compression/slurm/stage_1_naive_baseline_prelu_maxunpool_array.job,
            # not yet landed in results.csv. harvest() only keeps rows that
            # exist, so these are simply absent from plots/tables until the
            # array completes.
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_Baseline",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U2",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U4",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U8",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U16",
        ],
        "labels": {
            "nnUNetTrainerENet_1_naive_baseline_Baseline": "ENet Original",
            "nnUNetTrainerENet_1_naive_baseline_U2": "U2",
            "nnUNetTrainerENet_1_naive_baseline_U4": "U4",
            "nnUNetTrainerENet_1_naive_baseline_U8": "U8",
            "nnUNetTrainerENet_1_naive_baseline_U16": "U16",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_Baseline": "ENet Original (PReLU+maxunpool)",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U2": "U2 (PReLU+maxunpool)",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U4": "U4 (PReLU+maxunpool)",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U8": "U8 (PReLU+maxunpool)",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U16": "U16 (PReLU+maxunpool)",
        },
        "known_gaps": [],
    },
    "stage_2": {
        "title": "Single-op ablation at U4",
        "description": (
            "Each probe flips exactly one flag off Stage 1's U4 row "
            "(dilated=1,asymmetric=1,strided=1,dsc=0,"
            "context_pattern=default,prelu=0): PReLU on, max_unpool "
            "decoder, dilation off, asymmetric-factorization off."
        ),
        "config_names": [
            "nnUNetTrainerENet_1_naive_baseline_U4",  # reference, not a probe
            "nnUNetTrainerENet_2_special_ops_prelu",
            "nnUNetTrainerENet_2_special_ops_maxunpool",
            "nnUNetTrainerENet_2_special_ops_no_dilated",
            "nnUNetTrainerENet_2_special_ops_no_asymmetric",
        ],
        "labels": {
            "nnUNetTrainerENet_1_naive_baseline_U4": "U4 (reference)",
            "nnUNetTrainerENet_2_special_ops_prelu": "+PReLU",
            "nnUNetTrainerENet_2_special_ops_maxunpool": "+max_unpool",
            "nnUNetTrainerENet_2_special_ops_no_dilated": "-dilation",
            "nnUNetTrainerENet_2_special_ops_no_asymmetric": "-asymmetric",
        },
        "known_gaps": [],
    },
    "stage_3": {
        "title": "Context-block topology search under PReLU",
        "description": (
            "Curated 7-of-11 subset of raw stage 4_arch_probes: only the "
            "probes that vary the dilated context block's internal "
            "topology. All PReLU (inherited from Stage 2's winner). "
            "dense_dilation (S4.10) wins. Excluded (different axis "
            "entirely, not context-block topology, footnoted only): "
            "S4.1 e1_shape, S4.2 extra_initial, S4.7 double_projections, "
            "S4.8 two_block_skip."
        ),
        "config_names": [
            "nnUNetTrainerENet_4_4_1_shallow_dilation",
            "nnUNetTrainerENet_4_4_2_separable_dilated",
            "nnUNetTrainerENet_4_4_3_merge_dilated_pairs",
            "nnUNetTrainerENet_4_4_4_dsc_dilated_only",
            "nnUNetTrainerENet_4_7_dsc_no_projection",
            "nnUNetTrainerENet_4_8_dense_dilation",
            "nnUNetTrainerENet_4_9_shallow_dilation_wide",
        ],
        "labels": {
            "nnUNetTrainerENet_4_4_1_shallow_dilation": "shallow dilation",
            "nnUNetTrainerENet_4_4_2_separable_dilated": "separable dilated",
            "nnUNetTrainerENet_4_4_3_merge_dilated_pairs": "merge dilated pairs",
            "nnUNetTrainerENet_4_4_4_dsc_dilated_only": "DSC dilated-only",
            "nnUNetTrainerENet_4_7_dsc_no_projection": "DSC no-projection",
            "nnUNetTrainerENet_4_8_dense_dilation": "dense dilation",
            "nnUNetTrainerENet_4_9_shallow_dilation_wide": "shallow dilation (wide)",
        },
        "known_gaps": [],
        "excluded_note": (
            "S4.1 (4_1_e1_shape), S4.2 (4_2_extra_initial), "
            "S4.7 (4_5_double_projections), S4.8 (4_6_two_block_skip) -- "
            "these probe channel-shape/skip-connection axes, not context-"
            "block topology, so they don't belong in this comparison."
        ),
    },
    "stage_4": {
        "title": "dense_dilation sub-variant re-validation under ReLU",
        "description": (
            "After discovering FINN can't fold per-channel PReLU, "
            "Stage 3's winning topology (dense_dilation) was re-checked "
            "under ReLU across its own sub-variants: plain dense, "
            "separable-dilated, DSC no-projection, DSC projected. "
            "Plain dense_dilation (S12 dense) is Pareto-optimal in "
            "params, MACs, and activation buffer memory simultaneously."
        ),
        "config_names": [
            "nnUNetTrainerENet_12_dense_relu",
            "nnUNetTrainerENet_12_separable_dense_relu",
            "nnUNetTrainerENet_8_2_relu_no_reg_fullwidth",
            "nnUNetTrainerENet_9_4_dense_dilation_dsc_projected_relu",
        ],
        "labels": {
            "nnUNetTrainerENet_12_dense_relu": "S12 dense",
            "nnUNetTrainerENet_12_separable_dense_relu": "S12 separable",
            "nnUNetTrainerENet_8_2_relu_no_reg_fullwidth": "DSC no-proj",
            "nnUNetTrainerENet_9_4_dense_dilation_dsc_projected_relu": "DSC proj",
        },
        "known_gaps": [],
    },
    "stage_5": {
        "title": "Hardware-legality revisit: decoder swap",
        "description": (
            "FINN also can't fold bilinear resize efficiently. "
            "Stage 4's winner (S12 dense, ReLU already settled) is "
            "re-checked under nearest-neighbor and learned-upsample "
            "decoders. No separable/DSC variant was ever re-run under "
            "any alternate decoder -- confirmed absent in results.csv -- "
            "so only the winner is checked here."
        ),
        "config_names": [
            "nnUNetTrainerENet_12_dense_relu",
            "nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample",
            "nnUNetTrainerENet_12_dense_relu_learned_upsample",
        ],
        "labels": {
            "nnUNetTrainerENet_12_dense_relu": "bilinear (Stage 4 winner)",
            "nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample": "nearest-neighbor",
            "nnUNetTrainerENet_12_dense_relu_learned_upsample": "learned upsample",
        },
        "known_gaps": [],
        "excluded_note": (
            "nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample_256 "
            "exists but uses context_pattern=dense_dilation_half (a "
            "distinct variant, not a decoder-only rerun of the Stage-4 "
            "winner) and belongs to the downstream 256-res hardware-build "
            "pipeline, not this controlled comparison."
        ),
    },
}

EXCLUDED_FAMILIES_NOTE = """\
Excluded entirely from this narrative (not plotted anywhere):

- `3_transfer_original` -- warm-start-from-checkpoint experiment, orthogonal
  to this topology/activation/decoder search.
- Every reg_interleaved / separable-dense-dilation follow-on family: raw
  stages `5_arch_probe_pairs`, `6_dscnoprojdense_variants`,
  `7_reginterleaved_shape_variants`, `8_reginterleaved_isolation` (minus the
  one fullwidth DSC-no-proj row already used in Stage 4),
  `10_reginterleaved_separable_projected`, `13_*`, `15_*`..`22_*`, `23_*`,
  `25_*`, `26_*`, `27_*` -- none of these were carried forward to the final
  pick (S12 dense).
- Every quantization/QAT/PTQ/HAWQ derivative row (blank `abbrev`,
  `joint_alpha*`/`uniform_int*`/`ptq*`/`qat*` suffixes) -- that's the later
  quantization chapter's material, not this NAS narrative.
"""


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df["macs"] = df["flops"] / 2
    return df


def harvest(stage_id: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows for one narrative stage, in the stage's own declared order."""
    if df is None:
        df = load_results()
    spec = STAGES[stage_id]
    found = df[df["config_name"].isin(spec["config_names"])].copy()
    present = set(found["config_name"])
    ordered = [c for c in spec["config_names"] if c in present]
    found = found.set_index("config_name").loc[ordered].reset_index()
    found["label"] = found["config_name"].map(spec["labels"])
    return found


def report_gaps(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = load_results()
    present_names = set(df["config_name"])
    print("NAS narrative gap report")
    print("=" * 60)
    for stage_id, spec in STAGES.items():
        missing = [c for c in spec["config_names"] if c not in present_names]
        found_n = len(spec["config_names"]) - len(missing)
        status = "OK" if not missing else "GAPS"
        print(f"\n{stage_id} -- {spec['title']} [{status}: {found_n}/{len(spec['config_names'])} present]")
        for c in missing:
            print(f"  MISSING (not trained yet): {c}")
        for note in spec.get("known_gaps", []):
            print(f"  KNOWN GAP (flagged, not trained): {note}")
        if spec.get("excluded_note"):
            print(f"  EXCLUDED FROM THIS STAGE: {spec['excluded_note']}")
    print("\n" + EXCLUDED_FAMILIES_NOTE)


if __name__ == "__main__":
    report_gaps()

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
    "stage_0": {
        "title": "ENet-paper-faithful origin point",
        "description": (
            "The literal starting point of this search, predating even "
            "Stage 1's naive width sweep: ENet's own paper-faithful config "
            "(channels=16,64,128,128,64,16 -- the real paper's own "
            "split-stage2/3 widths, not a shrunk U-variant --, "
            "decoder_type=max_unpool, prelu=1, bottlenecks=4,8,8,2,1, "
            "context_pattern=default). Trained as "
            "nnUNetTrainerENet_enet_original_prelu_maxunpool_3c on "
            "Dataset511_ARCADE_1x1_3c (compression/scripts/"
            "make_arcade_3c_dataset.py -- Dataset509_ARCADE_1x1_4c with LM "
            "merged into background, LAD/RCA/LCX untouched, "
            "compression/slurm/stage_enet_original_prelu_maxunpool_3c.job). "
            "Real held-out-test-split per-class Dice, collected via the "
            "normal compression/collect_results.py pipeline (same as every "
            "other row in results.csv -- imagesTs/labelsTs, not the "
            "internal CV val-split). Paired with the identical config "
            "re-trained on the full 4-class dataset "
            "(nnUNetTrainerENet_enet_original_prelu_maxunpool_4c, "
            "compression/slurm/stage_enet_original_prelu_maxunpool_4c.job, "
            "Dataset509_ARCADE_1x1_4c, LM NOT merged) to isolate whether "
            "dropping LM changes RCA/LCX/LAD Dice. Result: small, mixed "
            "deltas, not a clear win either way -- LAD +0.0057 and RCA "
            "+0.0046 favor dropping LM (3c), LCX -0.0105 favors keeping it "
            "(4c). LM itself scores 0.8055 in the 4c run -- not the "
            "hardest class despite being the rarest, so it isn't obviously "
            "'soaking up' the other three's errors as hypothesized."
        ),
        "config_names": [
            "nnUNetTrainerENet_enet_original_prelu_maxunpool_3c",
            "nnUNetTrainerENet_enet_original_prelu_maxunpool_4c",
        ],
        "labels": {
            "nnUNetTrainerENet_enet_original_prelu_maxunpool_3c": "ENet Original (3-class, LM dropped)",
            "nnUNetTrainerENet_enet_original_prelu_maxunpool_4c": "ENet Original (4-class)",
        },
        "known_gaps": [],
    },
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
            # same U-series channel grid) -- queued via
            # compression/slurm/stage_1_naive_baseline_prelu_maxunpool_array.job,
            # not yet landed in results.csv. harvest() only keeps rows that
            # exist, so these are simply absent from plots/tables until the
            # array completes. NOTE: the array's own Baseline task (5-tuple
            # channels=16,64,128,64,16) was REMOVED -- the real
            # ENet-paper-faithful baseline point now lives in Stage 0
            # (nnUNetTrainerENet_enet_original_prelu_maxunpool_4c/3c, 6-tuple
            # channels=16,64,128,128,64,16, the paper's own split-stage2/3
            # form), not duplicated here.
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U2",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U4",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U8",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U16",
            # Third companion curve: PReLU + upsample_conv (isolates PReLU
            # alone, no max_unpool indices/patch-size coupling) -- from
            # compression/slurm/stage_1_naive_baseline_prelu_upsample_conv_array.job.
            # U4 deliberately absent here: nnUNetTrainerENet_2_special_ops_prelu
            # (Stage 2's own "+PReLU" probe) is already exactly this point
            # (same channels/flags), borrowed at plot time instead of
            # duplicating the row under a new config_name.
            "nnUNetTrainerENet_1_naive_baseline_prelu_upsample_conv_Baseline",
            "nnUNetTrainerENet_1_naive_baseline_prelu_upsample_conv_U2",
            "nnUNetTrainerENet_1_naive_baseline_prelu_upsample_conv_U8",
            "nnUNetTrainerENet_1_naive_baseline_prelu_upsample_conv_U16",
        ],
        "labels": {
            "nnUNetTrainerENet_1_naive_baseline_Baseline": "ENet Original",
            "nnUNetTrainerENet_1_naive_baseline_U2": "U2",
            "nnUNetTrainerENet_1_naive_baseline_U4": "U4",
            "nnUNetTrainerENet_1_naive_baseline_U8": "U8",
            "nnUNetTrainerENet_1_naive_baseline_U16": "U16",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U2": "U2 (PReLU+maxunpool)",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U4": "U4 (PReLU+maxunpool)",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U8": "U8 (PReLU+maxunpool)",
            "nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U16": "U16 (PReLU+maxunpool)",
            "nnUNetTrainerENet_1_naive_baseline_prelu_upsample_conv_Baseline": "Baseline (PReLU+upsample_conv)",
            "nnUNetTrainerENet_1_naive_baseline_prelu_upsample_conv_U2": "U2 (PReLU+upsample_conv)",
            "nnUNetTrainerENet_1_naive_baseline_prelu_upsample_conv_U8": "U8 (PReLU+upsample_conv)",
            "nnUNetTrainerENet_1_naive_baseline_prelu_upsample_conv_U16": "U16 (PReLU+upsample_conv)",
        },
        "known_gaps": [],
    },
    "stage_2": {
        "title": "Single-op ablation at U4",
        "description": (
            "First flip: PReLU on, off Stage 1's ReLU U4 row "
            "(dilated=1,asymmetric=1,strided=1,dsc=0,"
            "context_pattern=default,prelu=0) -- PReLU wins. Every "
            "subsequent probe (dilation off, asymmetric-factorization off) "
            "is then referenced against that PReLU + upsample_conv U4 point "
            "(nnUNetTrainerENet_2_special_ops_prelu) instead of the ReLU "
            "row, so the whole ablation stays internally consistent with "
            "the activation this thesis actually carries forward and "
            "matches ENet-paper-native PReLU. -dilation/-asymmetric are "
            "from compression/slurm/stage_2_special_ops_prelu_reference_"
            "array.job. The decoder (max_unpool) probe is NOT repeated here "
            "-- it's already covered by Stage 1's own PReLU+max_unpool "
            "curve (nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U4 "
            "is the matching U4 point on that curve)."
        ),
        "config_names": [
            "nnUNetTrainerENet_2_special_ops_prelu",  # reference: PReLU + upsample_conv
            "nnUNetTrainerENet_2_special_ops_prelu_no_dilated",
            "nnUNetTrainerENet_2_special_ops_prelu_no_asymmetric",
        ],
        "labels": {
            "nnUNetTrainerENet_2_special_ops_prelu": "Reference",
            "nnUNetTrainerENet_2_special_ops_prelu_no_dilated": "Without dilation",
            "nnUNetTrainerENet_2_special_ops_prelu_no_asymmetric": "Without asymmetric",
        },
        "known_gaps": [],
        "excluded_note": (
            "The max_unpool decoder probe is deliberately absent from this "
            "stage -- see Stage 1's own PReLU+max_unpool curve instead "
            "(nnUNetTrainerENet_1_naive_baseline_prelu_maxunpool_U4 is the "
            "matching U4 point). The original ReLU-referenced max_unpool/"
            "no_dilated/no_asymmetric probes (nnUNetTrainerENet_2_special_"
            "ops_maxunpool/no_dilated/no_asymmetric) are also superseded "
            "here by their PReLU-referenced counterparts and not plotted -- "
            "they remain in results.csv (raw stage 2_special_ops) for "
            "reference."
        ),
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
    "stage_6": {
        "title": "Post-selection probe: does stage3 (the dilated context "
                  "stage) matter, at matched width?",
        "description": (
            "Not part of how S12 dense was picked (Stages 1-5 already "
            "closed that question) -- a later, separate probe on top of "
            "S12 dense's own recipe applied back onto S5.6's PReLU/"
            "separable_dilated lineage (5_6_separable_dense_dilation, "
            "channels=4,16,32,16,4, bottlenecks=4,8,8,2,1, "
            "context_pattern=dense_dilation, separable_dilated=1), asking "
            "whether the dilated context block can be collapsed into "
            "stage2 alone. Raw stage 26_s5_6_probe_family "
            "(compression/slurm/archive/stage_26_s5_6_probe_family_array.job) "
            "ran 8 probes, half of which add a 'd1 lead-in' "
            "(context_pattern=dense_dilation_lead1, a distinct dilation-"
            "cycle shape). This stage isolates the 4 non-lead-in probes "
            "(plain dense_dilation) plus their S5.6 reference, i.e. only "
            "bottlenecks_per_stage[2] (stage3's block count) varies at "
            "each matched width -- lead-in variants are excluded so the "
            "stage3-removal effect isn't confounded with the cycle-shape "
            "change. "
            "Two matched with/without-stage3 pairs: at S5.6's own width "
            "(16,32,32,16,4), removing stage3 (bottlenecks 4,8,8,2,1 -> "
            "4,8,0,2,1) costs -0.065 dice (0.7985 -> 0.7334, -8.2% "
            "relative) for -37.8% params / -64.8% MACs / -44.2% activation-"
            "buffer elements. At the narrower w24 width (8,24,24,8,4), the "
            "same removal costs -0.079 dice (0.7567 -> 0.6776, -10.4% "
            "relative, a LARGER relative hit) for -40.6% params / -22.2% "
            "MACs / -45.0% mem elements. clDice and n_components move the "
            "same direction as dice in both pairs (fragmentation worsens "
            "when stage3 is removed), so this isn't a Dice-only artifact. "
            "None of these 5 rows are flagged converged_flag=True (all "
            "~140-150 of a presumably longer schedule) -- read magnitudes "
            "as directional, not final."
        ),
        "config_names": [
            "nnUNetTrainerENet_5_6_separable_dense_dilation",
            "nnUNetTrainerENet_26_3_no_stage3",
            "nnUNetTrainerENet_26_5_w24",
            "nnUNetTrainerENet_26_6_w24_no_stage3",
            "nnUNetTrainerENet_26_1_u8style",
        ],
        "labels": {
            "nnUNetTrainerENet_5_6_separable_dense_dilation": "S5.6 base (stage3 present)",
            "nnUNetTrainerENet_26_3_no_stage3": "S5.6 width (stage3 removed)",
            "nnUNetTrainerENet_26_5_w24": "w24 (stage3 present)",
            "nnUNetTrainerENet_26_6_w24_no_stage3": "w24 (stage3 removed)",
            "nnUNetTrainerENet_26_1_u8style": "u8-style width (stage3 present)",
        },
        "known_gaps": [
            "26_1_u8style (channels=4,8,16,8,4, the narrowest width probed) "
            "has no non-lead-in no-stage3 counterpart -- that pairing only "
            "exists as the lead-in variant 26_8_w24_no_stage3_d1leadin_halved, "
            "excluded from this stage. Included anyway as the narrowest "
            "stage3-present width point.",
        ],
        "excluded_note": (
            "The 4 d1-lead-in probes from the same raw stage "
            "(26_2_d1leadin, 26_4_d1leadin_no_stage3, "
            "26_7_w24_no_stage3_d1leadin, "
            "26_8_w24_no_stage3_d1leadin_halved -- context_pattern="
            "dense_dilation_lead1) are a different axis (dilation-cycle "
            "shape, not stage3 removal) and are left out so the two "
            "changes aren't conflated. Also excluded: the later "
            "26_9/26_10/26_9_..._pruned5 follow-ons (w24 with stage1/4 "
            "width also bumped to 12, nonneg_block/relu PReLU-variant "
            "sweep) -- same raw stage tag, different question."
        ),
    },
    "stage_7": {
        "title": "S12 dense family: width variation",
        "description": (
            "Also post-selection (like Stage 6): raw stage "
            "12_dense_relu_width_sweep re-runs S12 dense's own recipe "
            "(context_pattern=dense_dilation, prelu=0, upsample_conv, "
            "bottlenecks=4,8,8,2,1 fixed throughout -- only stage2/3/4 "
            "channel width f2=f3 and f1=f4 move) at 5 narrower widths "
            "below the Stage-4/5 winner's own 16,32,32,16,4, plus the "
            "winner itself (nnUNetTrainerENet_12_dense_relu) as the "
            "widest point. Unlike Stage 6, bottlenecks_per_stage never "
            "changes here -- this isolates width alone, holding stage3 "
            "(and every other topology choice) fixed at the Stage 4/5 "
            "winner's own settings."
        ),
        "config_names": [
            "nnUNetTrainerENet_12_dense_relu_w4_8",
            "nnUNetTrainerENet_12_dense_relu_w8_16",
            "nnUNetTrainerENet_12_dense_relu_w8_20",
            "nnUNetTrainerENet_12_dense_relu_w12_20",
            "nnUNetTrainerENet_12_dense_relu_w12_24",
            "nnUNetTrainerENet_12_dense_relu",
        ],
        "labels": {
            "nnUNetTrainerENet_12_dense_relu_w4_8": "w4/8",
            "nnUNetTrainerENet_12_dense_relu_w8_16": "w8/16",
            "nnUNetTrainerENet_12_dense_relu_w8_20": "w8/20",
            "nnUNetTrainerENet_12_dense_relu_w12_20": "w12/20",
            "nnUNetTrainerENet_12_dense_relu_w12_24": "w12/24",
            "nnUNetTrainerENet_12_dense_relu": "w16/32 (Stage 4/5 winner)",
        },
        "known_gaps": [],
    },
    "final_results": {
        "title": "Final results summary",
        "description": (
            "Not a NAS-narrative stage in its own right (Stages 4-7's own "
            "notebook sections were retired once this summary landed) -- "
            "the three points that matter for the thesis's own bottom "
            "line, spanning the full search end-to-end: the ENet-paper "
            "naive-compression starting point, the Stage-3 topology "
            "winner under its ORIGINAL PReLU + bilinear decoder, and the "
            "actual FINN-legal S12 dense variant (ReLU + nearest-neighbor "
            "+ trainable resize-conv) carried into the hardware pipeline."
        ),
        "config_names": [
            "nnUNetTrainerENet_enet_original_prelu_maxunpool_4c",
            "nnUNetTrainerENet_4_8_dense_dilation",
            "nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample",
        ],
        "labels": {
            "nnUNetTrainerENet_enet_original_prelu_maxunpool_4c": "ENet naive baseline (PReLU + MaxUnpool)",
            "nnUNetTrainerENet_4_8_dense_dilation": "S12 dense (PReLU + Bilinear)",
            "nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample": "S12 dense, FINN-compatible (ReLU + Nearest Neighbor)",
        },
        "known_gaps": [],
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
  `25_*`, `27_*` -- none of these were carried forward to the final pick
  (S12 dense).
- Raw stage `26_s5_6_probe_family` -- mostly excluded (it's a post-selection
  probe on S5.6's own lineage, not part of how S12 dense was chosen), EXCEPT
  the 5 non-lead-in rows now isolated in Stage 6 (see its own excluded_note
  for exactly which 26_* rows still aren't shown anywhere: the 4 d1-lead-in
  variants and the later 26_9/26_10/pruned5 follow-ons).
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

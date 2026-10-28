"""Test-split Dice of every quantization scheme of the constrained 150 FPS MILP builds (ILP/artifacts/S12_512_ablation_v2) + the FP32 reference -> dice_scores.csv.

Dice = the QAT evaluation of nas/results.csv (300-image Dataset509 test split, checkpoint_best.pth, 15-epoch QAT fine-tune warm-started from the FP32 reference; collect_results.py). The uniform
INT4 / INT6 / INT8 rows are the uniform-bit QAT runs: the MILP only chooses the folding for them, bits are the same on all 88 layers. INT8 does not fit the 0.4 caps at 150 FPS (cap40_fps150/uniform_int8
is infeasible), so its MILP solve is the relaxed-caps one (relaxed_fps150/uniform_int8). The mixed 4/6/8 row is the cap40_fps150/mixed_468 allocation (QAT job qat_..._ablation_v2_cap40_fps150_mixed468_ft15ep).
The FINN auto-fold INT6 build has the INT6 bits (same Dice) and is not listed.

Run:  python report/MILP/constrained_builds/build_dice_csv.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SRC = os.path.join(REPO, "nas", "results.csv")
OUT = os.path.join(HERE, "dice_scores.csv")

T = "nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_perlayer"
LEGACY = "nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_256_perlayer"   # the uniform runs were trained before the trainer rename
W = "12_dense_relu_warmstart150ep"
# scheme id, label, bits, MILP solve folder, config_name in nas/results.csv
SCHEMES = [
    ("fp32", "FP32 reference", "FP32", "", "nnUNetTrainerENet_12_dense_relu_warmstart150ep"),
    ("int4", "INT4", "uniform 4/4", "cap40_fps150/uniform_int4", f"{LEGACY}_{W}_bilinear_uniform_int4_perlayer_ft15ep"),
    ("int6", "INT6", "uniform 6/6", "cap40_fps150/uniform_int6", f"{LEGACY}_{W}_bilinear_uniform_int6_perlayer_ft15ep"),
    ("int8", "INT8", "uniform 8/8", "relaxed_fps150/uniform_int8 (cap40 infeasible)", f"{LEGACY}_{W}_bilinear_uniform_int8_perlayer_ft15ep"),
    ("mixed468", "mixed 4/6/8", "mixed 4/6/8", "cap40_fps150/mixed_468", f"{T}_{W}_bilinear_ablation_v2_cap40_fps150_mixed468_perlayer_ft15ep"),
]
METRICS = ["dice", "dice_binary", "dice_LAD", "dice_RCA", "dice_LCX", "dice_LM", "cldice", "n_components"]

rows = {r["config_name"]: r for r in csv.DictReader(open(SRC, newline=""))}
fp32 = float(rows[SCHEMES[0][4]]["dice"])
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scheme", "label", "bits", "milp_solve", "n_images", "dice", "dice_vs_fp32", "dice_binary", "dice_LAD", "dice_RCA", "dice_LCX", "dice_LM", "cldice", "n_components", "epochs", "qat_run"])
    for sid, label, bits, solve, cfg in SCHEMES:
        r = rows[cfg]
        w.writerow([sid, label, bits, solve, 300, f"{float(r['dice']):.4f}", f"{float(r['dice']) - fp32:+.4f}"] + [f"{float(r[m]):.4f}" for m in METRICS[1:6]] +
                   [f"{float(r['cldice']):.4f}", f"{float(r['n_components']):.1f}", r["epochs"], cfg])
print(open(OUT).read())

import sys, math
sys.path.insert(0, r"c:\DEV\repos\LightM-UNet\compression\hawq")
import finn_cost_model as fcm

DENSE_LAYERS = [
    ("stem.0", 1, 32, 1, 1, 1, 1),
    ("bottlenecks.0.reduce.0", 32, 8, 1, 1, 1, 1),
    ("bottlenecks.0.conv", 8, 8, 3, 3, 2, 2),
    ("bottlenecks.0.expand.0", 8, 32, 1, 1, 1, 1),
    ("bottlenecks.1.reduce.0", 32, 8, 1, 1, 1, 1),
    ("bottlenecks.1.conv", 8, 8, 3, 3, 4, 4),
    ("bottlenecks.1.expand.0", 8, 32, 1, 1, 1, 1),
]
SEPARABLE_LAYERS = [
    ("stem.0", 1, 32, 1, 1, 1, 1),
    ("bottlenecks.0.reduce.0", 32, 8, 1, 1, 1, 1),
    ("bottlenecks.0.conv.0", 8, 8, 3, 1, 2, 2),
    ("bottlenecks.0.conv.3", 8, 8, 1, 3, 2, 2),
    ("bottlenecks.0.expand.0", 8, 32, 1, 1, 1, 1),
    ("bottlenecks.1.reduce.0", 32, 8, 1, 1, 1, 1),
    ("bottlenecks.1.conv.0", 8, 8, 3, 1, 4, 4),
    ("bottlenecks.1.conv.3", 8, 8, 1, 3, 4, 4),
    ("bottlenecks.1.expand.0", 8, 32, 1, 1, 1, 1),
]

def make_layer(name, cin, cout, kh, kw, dh, dw):
    return fcm.LayerGeometry(op_type="Conv2d", name=name, stage=name.split(".")[0],
                              cin=cin, hin=32, win=32, cout=cout, hout=32, wout=32,
                              kh=kh, kw=kw, sh=1, sw=1, dh=dh, dw=dw, groups=1)

def pieces(layer, W, A, P, Q):
    mw = fcm.max_simd(layer)
    addertree_luts = (W + A) * (2 * Q - 1)
    alpha = math.log2(mw) + W + A - 1 - 1
    acc_luts = min(32, alpha + math.log2(1 + 2 ** -alpha) + 1)
    c0, c1 = 300, 1.1
    base = c0 + c1 * P * (addertree_luts + acc_luts) + 426
    imb_standalone = 232.33 * max(0, P - Q) * A      # single-var fit, NO MH (the "drop MH" candidate)
    imb_joint = 167.38 * max(0, P - Q) * A           # joint-fit coefficient (used only alongside MH)
    mh_term = 274.14 * layer.cout
    return base, base + imb_standalone, base + imb_joint + mh_term

def totals(layers, W, A, mode):
    tb = ti = tboth = 0.0
    for spec in layers:
        layer = make_layer(*spec)
        pe, simd = (1, 1) if mode == "serial" else (layer.cout, 1)
        b, i, both = pieces(layer, W, A, pe, simd)
        tb += b; ti += i; tboth += both
    return tb, ti, tboth

probes = [
    ("dense_int4",          DENSE_LAYERS,     4, 4, "serial",      6174.0),
    ("separable_int4",      SEPARABLE_LAYERS, 4, 4, "serial",      7377.0),
    ("dense_int6",          DENSE_LAYERS,     6, 6, "serial",      7378.0),
    ("separable_int6",      SEPARABLE_LAYERS, 6, 6, "serial",      9089.0),
    ("dense_int8_pemh",     DENSE_LAYERS,     8, 8, "pemh_simd1", 15252.0),
    ("separable_int8_pemh", SEPARABLE_LAYERS, 8, 8, "pemh_simd1", 20560.0),
]

print(f"{'probe':22s} {'base':>9s} {'r_base':>7s}  {'+imb(232, no MH)':>18s} {'r_imb':>7s}  {'+imb+MH(adopted)':>17s} {'r_both':>7s}")
for name, layers, w, a, mode, real in probes:
    b, i, both = totals(layers, w, a, mode)
    print(f"{name:22s} {b:9,.0f} {b/real:7.3f}  {i:18,.0f} {i/real:7.3f}  {both:17,.0f} {both/real:7.3f}")

print()
print("--- Also: re-check fit quality on the ORIGINAL 89-row calibration set if MH is dropped ---")
CSV = r"c:\DEV\repos\LightM-UNet\hardware\mvau_lut_calibration_dataset.csv"
import csv as csvmod
rows = list(csvmod.DictReader(open(CSV, newline="")))

def baseline_mvu_lut(row):
    W = int(row["weight_bits"]); A = int(row["act_bits"]); P = int(row["PE"]); Q = int(row["SIMD"]); MW = int(row["MW"])
    c0, c1 = 300, 1.1
    addertree_luts = (W + A) * (2 * Q - 1)
    alpha = math.log2(MW) + W + A - 1 - 1
    acc_luts = min(32, alpha + math.log2(1 + 2 ** -alpha) + 1)
    return c0 + c1 * P * (addertree_luts + acc_luts)

def r2(y, yhat):
    ybar = sum(y) / len(y)
    ss_res = sum((a-b)**2 for a, b in zip(y, yhat))
    ss_tot = sum((a-ybar)**2 for a in y)
    return 1 - ss_res/ss_tot

real_lut = [float(r["real_LUT"]) for r in rows]
base_pred = [baseline_mvu_lut(r) for r in rows]
imb = [max(0, int(r["PE"]) - int(r["SIMD"])) * int(r["act_bits"]) for r in rows]
pred_imb_only = [bp + 232.33*x for bp, x in zip(base_pred, imb)]
print(f"in-sample (89-row calib set) R^2, imbalance-only (no MH): {r2(real_lut, pred_imb_only):.4f}")

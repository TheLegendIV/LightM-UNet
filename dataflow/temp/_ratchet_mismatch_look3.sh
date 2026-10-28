cd /tmp
grep -nE "^--- |landed MVAUs matched|WARNING|REALIGN|mismatch\(es\)|OK: landed|LANDED CHECK" ratchet256_bridge_p2.log | cut -c1-400
echo ==== per-arm landed MVAU list from the onnx (name PE SIMD cycles weight_count)
cd /home/thelegendiv/finn/notebooks/enet
cat > /tmp/dump_landed25.py <<'EOF'
import sys, json
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
import onnx
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
import check_milp_vs_landed_folding as c
P = "finn_deployment_outputs/ratchet256_preamble_20261009_003521/landed_partition2_ratchet_25pct_simfifo.onnx"
L = c.landed_mvaus(onnx.load(P))
for r in L:
    print(r["node"], r["pe"], r["simd"], r["cycles"], r["weight_count"], "swu", r["swu_simd"], r["swu_cycles"], "thr", r["thr_pe"])
J = json.load(open("/home/thelegendiv/finn/notebooks/enet/layer_bits_folding_ratchet_25pct_simfifo.json"))["per_layer"]
ks = [(k, e["pe"], e["simd"], e["mvu_cycles"]) for k, e in J.items() if e.get("mvu_cycles") is not None]
print("MILP mvau-type layers:", len(ks))
for k in ks:
    if k[0].startswith("stage2."): print(k)
EOF
python3 /tmp/dump_landed25.py 2>&1 | tail -60

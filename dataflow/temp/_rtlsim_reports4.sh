cd /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350 || exit 1
python3 - <<'EOF'
import json
for p in (1, 7):
    d = json.load(open(f"fifo_plan_partition{p}.json"))
    print("== partition", p, type(d).__name__, list(d)[:12] if isinstance(d, dict) else len(d))
    print(json.dumps(d, indent=0)[:2500])
EOF
grep -n -i -E 'stock|interior|gate|allowed' /tmp/ooc_S12_u8in_full.log | head -30

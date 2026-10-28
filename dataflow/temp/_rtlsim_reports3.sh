cd /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/report || exit 1
python3 - <<'EOF'
import json
d = json.load(open("rtlsim_per_partition.json"))
print(json.dumps(d, indent=1)[:3500])
EOF

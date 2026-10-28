$root = "hardware/temp/ablation_root"
New-Item -ItemType Directory -Force $root | Out-Null
$dirs = @(
 "ratchet_ablation_finn_autofold_autofold_partition2_20261009_154752",
 "ratchet_1pct_simfifo_milpfold_partition2_20261009_154757",
 "ratchet_25pct_simfifo_milpfold_partition2_20261009_154802",
 "ratchet_100pct_simfifo_milpfold_partition2_20261009_155407",
 "ratchet_200pct_simfifo_milpfold_partition2_20261009_175218",
 "ratchet_off_simfifo_milpfold_partition2_20261009_180323")
foreach ($d in $dirs) {
  New-Item -ItemType Directory -Force "$root/$d" | Out-Null
  docker cp "finn_persistent:/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/$d/report" "$root/$d/"
}
@{ ratchet_ablation_finn_autofold = 20; ratchet_1pct_simfifo = 25; ratchet_25pct_simfifo = 20; ratchet_100pct_simfifo = 15; ratchet_200pct_simfifo = 15; ratchet_off_simfifo = 15 } | ConvertTo-Json | Set-Content "$root/real_dsp.json"
.\.venv\Scripts\python.exe hardware/builds/S12_dense_256_ratchet_ablation_v1/collect_results_partitions.py --deployment-root $root --dsp-overrides-json "$root/real_dsp.json" --partition 2

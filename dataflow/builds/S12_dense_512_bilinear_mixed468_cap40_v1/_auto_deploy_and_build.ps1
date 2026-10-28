# Waits for (1) the export + verification gate of the mixed468 cap40 network (CPU container infallible_hertz) and (2) the net_explicit.py FIFO simulation of its MILP folding (host),
# then copies the network, its conv order, the verify reference and the folding json into finn_persistent and starts the preamble + checks + full 8-way build
# (_launch_mixed468_cap40_v1.sh all). Aborts without starting anything if the export did not end with "VERIFY OK." or a file is missing.
# Progress is written to _auto_build.log next to this script.   Start:  powershell -File _auto_deploy_and_build.ps1
$ErrorActionPreference = "Continue"
$repo = "C:\DEV\repos\LightM-UNet"
$job = "$repo\dataflow\builds\S12_dense_512_bilinear_mixed468_cap40_v1"
$log = "$job\_auto_build.log"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Out-File -Append -Encoding utf8 $log }
$stem = "quantEnet_S12_dense_512_bilinear_mixed468_cap40_v1"
$sim = "$repo\ILP\artifacts\S12_512_ablation_v2\cap40_fps150\mixed_468\simfifo\layer_bits_folding_mixed468_cap40_simfifo.json"
$E = "finn_persistent:/home/thelegendiv/finn/notebooks/enet"
Log "watcher started"
$deadline = (Get-Date).AddHours(4)
while ((Get-Date) -lt $deadline) {
    $exportRunning = (docker exec infallible_hertz bash -c "pgrep -f '[f]inn_export_S12_dense_512_bilinear_mixed468' | wc -l").Trim() -ne "0"    # [f] keeps pgrep from matching its own bash -c wrapper
    $simRunning = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'net_explicit' -and $_.CommandLine -match 'mixed468_cap40' }).Count -gt 0
    if (-not $exportRunning -and -not $simRunning) { break }
    Start-Sleep 30
}
Log "export running: $exportRunning, sim running: $simRunning"
$tail = docker exec infallible_hertz bash -c "tail -5 /tmp/export_mixed468_cap40.log"
if (-not ($tail -match "VERIFY OK")) { Log "ABORT: the export did not end with VERIFY OK. :`n$tail"; exit 1 }
if (-not (Test-Path $sim)) { Log "ABORT: FIFO simulation output missing: $sim"; exit 1 }
$out = "$job\outputs"
foreach ($f in "${stem}_ft15ep_u8in.onnx", "${stem}_conv_order.json", "${stem}_ft15ep_u8in_verify_ref.npz") {
    if (-not (Test-Path "$out\$f")) { Log "ABORT: missing $out\$f"; exit 1 }
}
foreach ($f in Get-ChildItem $out -Filter "${stem}_ft15ep_u8in.onnx*") { docker cp $f.FullName "$E/$($f.Name)" | Out-Null }
docker cp "$out\${stem}_conv_order.json" "$E/${stem}_conv_order.json" | Out-Null
docker cp "$out\${stem}_ft15ep_u8in_verify_ref.npz" "$E/${stem}_ft15ep_u8in_verify_ref.npz" | Out-Null
docker cp $sim "$E/layer_bits_folding_S12_512_mixed468_cap40_simfifo.json" | Out-Null
docker cp "$job\_launch_mixed468_cap40_v1.sh" "$E/_launch_mixed468_cap40_v1.sh" | Out-Null
docker exec finn_persistent bash -c "cd /home/thelegendiv/finn/notebooks/enet && tr -d '\r' < _launch_mixed468_cap40_v1.sh > _l.sh && mv _l.sh _launch_mixed468_cap40_v1.sh && ls -l ${stem}_ft15ep_u8in.onnx* layer_bits_folding_S12_512_mixed468_cap40_simfifo.json" 2>&1 | ForEach-Object { Log $_ }
$r = docker exec finn_persistent bash -c "cd /home/thelegendiv/finn/notebooks/enet && bash _launch_mixed468_cap40_v1.sh all"
Log "launched: $r"

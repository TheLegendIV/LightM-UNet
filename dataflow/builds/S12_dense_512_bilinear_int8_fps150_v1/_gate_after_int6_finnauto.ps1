# Waits until the INT6 FINN-auto job's run script (preamble + 8-way build in finn_persistent) has ended, then starts the INT8 fps150 deploy+build watcher.
$repo = "C:\DEV\repos\LightM-UNet"
$log = "$repo\hardware\builds\S12_dense_512_bilinear_int8_fps150_v1\_auto_build.log"
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  gate: waiting for the INT6 finnauto build to end" | Out-File -Append -Encoding utf8 $log
while ($true) {
    $n = (docker exec finn_persistent bash -c "pgrep -f '[_]run_S12_dense_512_bilinear_int6_cap40_finnauto_v1' | wc -l").Trim()
    if ($n -eq "0") { break }
    Start-Sleep 60
}
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  gate: INT6 finnauto build ended, starting the INT8 deploy watcher" | Out-File -Append -Encoding utf8 $log
$d = "MILP\artifacts\S12_512_ablation_v2\relaxed_fps150\uniform_int8"
& "$repo\hardware\builds\S12_dense_512_bilinear_uniform_jobs\_auto_deploy_and_build.ps1" -Job S12_dense_512_bilinear_int8_fps150_v1 -Stem quantEnet_S12_dense_512_bilinear_int8_relaxed_v1 -Tag S12_dense_512_bilinear_int8_fps150_v1 -SimJson "$d\simfifo\layer_bits_folding_int8_fps150_simfifo.json" -SimTag int8_fps150_simfifo -ExportTag int8_relaxed -ExportBits 8

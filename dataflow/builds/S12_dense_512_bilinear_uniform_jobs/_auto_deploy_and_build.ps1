# Generic watcher for one S12 512 bilinear build: waits for (1) the export + verification gate (CPU container infallible_hertz, log /tmp/export_<ExportTag>.log) and (2) the net_explicit.py FIFO
# simulation of its MILP folding (host), then copies the network files + simulated folding into finn_persistent and runs the generic launcher (_launch_job.sh ... all): preamble + checks, then the
# full 8-way build. Aborts without starting anything if the export did not end with "VERIFY OK." or a file is missing. Progress goes to <JobDir>\_auto_build.log.
# Usage: powershell -File _auto_deploy_and_build.ps1 -Job S12_dense_512_bilinear_int4_cap40_v1 -Stem quantEnet_S12_dense_512_bilinear_int4_cap40_v1 -Tag S12_dense_512_bilinear_int4_cap40_v1 `
#          -SimJson ILP\artifacts\...\simfifo\layer_bits_folding_int4_cap40_simfifo.json -SimTag int4_cap40 -ExportTag int4_cap40 -ExportBits 4
param(
    [Parameter(Mandatory)] [string]$Job, [Parameter(Mandatory)] [string]$Stem, [Parameter(Mandatory)] [string]$Tag,
    [Parameter(Mandatory)] [string]$SimJson, [Parameter(Mandatory)] [string]$SimTag, [Parameter(Mandatory)] [string]$ExportTag, [Parameter(Mandatory)] [int]$ExportBits
)
$ErrorActionPreference = "Continue"
$repo = "C:\DEV\repos\LightM-UNet"
$jobDir = "$repo\dataflow\builds\$Job"
$log = "$jobDir\_auto_build.log"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Out-File -Append -Encoding utf8 $log }
$sim = if ([System.IO.Path]::IsPathRooted($SimJson)) { $SimJson } else { "$repo\$SimJson" }
$E = "finn_persistent:/home/thelegendiv/finn/notebooks/enet"
$foldName = "layer_bits_folding_$Tag.json"
Log "watcher started ($Tag)"
$deadline = (Get-Date).AddHours(8)
while ((Get-Date) -lt $deadline) {
    # [f] / [n] keep pgrep / the command-line match from matching their own wrapper process
    $exportRunning = (docker exec infallible_hertz bash -c "pgrep -f '[f]inn_export_uniform.py --bits $ExportBits' | wc -l").Trim() -ne "0"
    $simRunning = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'net_explici[t]' -and $_.CommandLine -match [regex]::Escape($SimTag) }).Count -gt 0
    if (-not $exportRunning -and -not $simRunning) { break }
    Start-Sleep 30
}
Log "export running: $exportRunning, sim running: $simRunning"
$tail = docker exec infallible_hertz bash -c "tail -5 /tmp/export_$ExportTag.log"
if (-not ($tail -match "VERIFY OK")) { Log "ABORT: the export did not end with VERIFY OK. :`n$tail"; exit 1 }
if (-not (Test-Path $sim)) { Log "ABORT: FIFO simulation output missing: $sim"; exit 1 }
$out = "$jobDir\outputs"
foreach ($f in "${Stem}_ft15ep_u8in.onnx", "${Stem}_conv_order.json", "${Stem}_ft15ep_u8in_verify_ref.npz") {
    if (-not (Test-Path "$out\$f")) { Log "ABORT: missing $out\$f"; exit 1 }
}
foreach ($f in Get-ChildItem $out -Filter "${Stem}_ft15ep_u8in.onnx*") { docker cp $f.FullName "$E/$($f.Name)" | Out-Null }
docker cp "$out\${Stem}_conv_order.json" "$E/${Stem}_conv_order.json" | Out-Null
docker cp "$out\${Stem}_ft15ep_u8in_verify_ref.npz" "$E/${Stem}_ft15ep_u8in_verify_ref.npz" | Out-Null
docker cp $sim "$E/$foldName" | Out-Null
docker cp "$repo\dataflow\builds\S12_dense_512_bilinear_uniform_jobs\_launch_job.sh" "$E/_launch_job.sh" | Out-Null
docker exec finn_persistent bash -c "cd /home/thelegendiv/finn/notebooks/enet && tr -d '\r' < _launch_job.sh > _l.sh && mv -f _l.sh _launch_job.sh && ls -l ${Stem}_ft15ep_u8in.onnx* $foldName" 2>&1 | ForEach-Object { Log $_ }
$r = docker exec finn_persistent bash -c "cd /home/thelegendiv/finn/notebooks/enet && bash _launch_job.sh $Stem $Tag $foldName all"
Log "launched: $r"

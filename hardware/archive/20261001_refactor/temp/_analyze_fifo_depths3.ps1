$lines = Get-Content C:\DEV\repos\LightM-UNet\hardware\temp\_fifo_depths_raw3.txt
$parsed = foreach ($l in $lines) {
    if ($l -match '^(code_gen_ipgen_StreamingFIFO_rtl_(\d+))_\S+ DEPTH=(\d*) WIDTH=(\d*) MTIME=(\d+)$') {
        [PSCustomObject]@{
            NodeNum = [int]$Matches[2]
            Depth   = if ($Matches[3]) { [int]$Matches[3] } else { $null }
            Width   = if ($Matches[4]) { [int]$Matches[4] } else { $null }
            MTime   = [int64]$Matches[5]
        }
    }
}
$valid = $parsed | Where-Object { $null -ne $_.Depth -and $null -ne $_.Width }
Write-Output ("Valid rows: {0}" -f $valid.Count)

# collapse to latest attempt per NodeNum (final settled depth for that logical FIFO slot)
$latest = $valid | Group-Object NodeNum | ForEach-Object {
    $sorted = $_.Group | Sort-Object MTime -Descending
    $top = $sorted[0]
    [PSCustomObject]@{ NodeNum = [int]$_.Name; Depth = $top.Depth; Width = $top.Width; MTime = $top.MTime }
}
Write-Output ("Distinct FIFO node-number slots: {0}" -f $latest.Count)

# LUT estimate per FIFO using SRLC32E packing: ceil(depth/32) * width
$withLut = $latest | ForEach-Object {
    $lut = [Math]::Ceiling($_.Depth / 32.0) * $_.Width
    $_ | Add-Member -NotePropertyName EstLUT -NotePropertyValue $lut -PassThru
}

Write-Output "--- Full table sorted by depth desc ---"
$withLut | Sort-Object Depth -Descending | Format-Table NodeNum, Depth, Width, EstLUT -AutoSize

$total = ($withLut | Measure-Object EstLUT -Sum).Sum
Write-Output ("Total estimated LUTs (SRLC32E, all latest-settled FIFOs, dedup by node-number only): {0}" -f $total)

Write-Output "--- Threshold analysis: if we cap depth at X, how many LUTs saved / how many FIFOs affected ---"
foreach ($cap in 256,512,1024,2048,4096,8192,16384,32768) {
    $over = $withLut | Where-Object { $_.Depth -gt $cap }
    $lutIfCapped = ($withLut | ForEach-Object { [Math]::Ceiling([Math]::Min($_.Depth,$cap) / 32.0) * $_.Width } | Measure-Object -Sum).Sum
    Write-Output ("cap={0,6}  fifos_over_cap={1,3}  total_est_LUT_if_capped={2,10}  (vs uncapped {3})" -f $cap, $over.Count, $lutIfCapped, $total)
}

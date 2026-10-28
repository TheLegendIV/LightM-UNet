$lines = Get-Content C:\DEV\repos\LightM-UNet\hardware\temp\_fifo_depths_raw2.txt
$parsed = foreach ($l in $lines) {
    if ($l -match '^(code_gen_ipgen_StreamingFIFO_rtl_(\d+))_\S+ DEPTH=(\d*) MTIME=(\d+)$') {
        [PSCustomObject]@{ NodeNum = [int]$Matches[2]; Depth = if ($Matches[3]) { [int]$Matches[3] } else { $null }; MTime = [int64]$Matches[4] }
    }
}
Write-Output ("Parsed: {0}" -f $parsed.Count)
$valid = $parsed | Where-Object { $null -ne $_.Depth }
Write-Output ("Valid (non-null depth): {0}" -f $valid.Count)

Write-Output "--- Depth distribution (raw, all attempts across all iterations) ---"
$valid | Group-Object Depth | Sort-Object { [int]$_.Name } | ForEach-Object {
    Write-Output ("depth={0,8}  count={1}" -f $_.Name, $_.Count)
}

Write-Output "--- Per NodeNum: LATEST (by mtime) depth = likely final settled value ---"
$latest = $valid | Group-Object NodeNum | ForEach-Object {
    $sorted = $_.Group | Sort-Object MTime -Descending
    [PSCustomObject]@{ NodeNum = [int]$_.Name; LatestDepth = $sorted[0].Depth; LatestMTime = $sorted[0].MTime; NumAttempts = $_.Count; MaxDepthSeen = ($_.Group | Measure-Object Depth -Maximum).Maximum }
}
Write-Output ("Distinct logical FIFO node IDs: {0}" -f $latest.Count)
Write-Output "--- Distribution of LATEST depth per node ---"
$latest | Group-Object LatestDepth | Sort-Object { [int]$_.Name } | ForEach-Object {
    Write-Output ("latest_depth={0,8}  num_nodes={1}" -f $_.Name, $_.Count)
}
Write-Output "--- Top 40 nodes by latest depth ---"
$latest | Sort-Object LatestDepth -Descending | Select-Object -First 40 | Format-Table -AutoSize

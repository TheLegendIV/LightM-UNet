$lines = Get-Content C:\DEV\repos\LightM-UNet\hardware\temp\_fifo_depths_raw.txt
$parsed = foreach ($l in $lines) {
    if ($l -match '^(code_gen_ipgen_StreamingFIFO_rtl_(\d+))_\S+ DEPTH=(\d*)$') {
        [PSCustomObject]@{ NodeNum = [int]$Matches[2]; Depth = if ($Matches[3]) { [int]$Matches[3] } else { $null } }
    }
}
Write-Output ("Parsed: {0}" -f $parsed.Count)
$valid = $parsed | Where-Object { $null -ne $_.Depth }
Write-Output ("Valid (non-null depth): {0}" -f $valid.Count)
Write-Output "--- Depth distribution ---"
$valid | Group-Object Depth | Sort-Object { [int]$_.Name } | ForEach-Object {
    Write-Output ("depth={0,8}  count={1}" -f $_.Name, $_.Count)
}
Write-Output "--- Top 40 NodeNum by max depth seen ---"
$grouped = $valid | Group-Object NodeNum | ForEach-Object {
    [PSCustomObject]@{ NodeNum = [int]$_.Name; MaxDepth = ($_.Group | Measure-Object Depth -Maximum).Maximum; Count = $_.Count }
}
$grouped | Sort-Object MaxDepth -Descending | Select-Object -First 40 | Format-Table -AutoSize

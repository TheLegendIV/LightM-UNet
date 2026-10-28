# Host-side CPU package energy logger (Windows "Energy Meter" RAPL counter), read by analysis/benchmark/power_meter.py from inside lightmunet_dev
# through the bind-mounted repo (WSL2 containers have no /sys/class/powercap). Run on the Windows host while the benchmark runs; stop with Ctrl+C.
#   powershell -ExecutionPolicy Bypass -File analysis\benchmark\host_rapl_sampler.ps1
# Columns: unix_s (host clock), energy_pwh (cumulative RAPL_Package0_PKG energy in picowatt-hours; 1 count = 3.6e-9 J), power_mw (Windows' own average, for cross-checking only).
param([string]$Out = "$PSScriptRoot\power_logs\rapl.csv", [int]$IntervalMs = 100)
New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
$Flag = Join-Path (Split-Path $Out) "sample.flag"      # created / removed by power_meter.py (HostRaplMeter.resume / pause)
$e = New-Object System.Diagnostics.PerformanceCounter('Energy Meter', 'Energy', 'RAPL_Package0_PKG', $true)
$p = New-Object System.Diagnostics.PerformanceCounter('Energy Meter', 'Power', 'RAPL_Package0_PKG', $true)
$null = $p.NextValue()
$w = New-Object System.IO.StreamWriter($Out, $false)
$w.AutoFlush = $true
$w.WriteLine("unix_s,energy_pwh,power_mw")
$epoch = [DateTime]::SpecifyKind([DateTime]'1970-01-01', 'Utc')
try {
    while ($true) {
        if (-not (Test-Path $Flag)) { Start-Sleep -Milliseconds 200; continue }     # sample only while the benchmark holds the flag, so the sampler never runs during latency timing
        $t =([DateTime]::UtcNow - $epoch).TotalSeconds
        $w.WriteLine(("{0:F3},{1},{2:F0}" -f $t, $e.RawValue, $p.NextValue()))
        Start-Sleep -Milliseconds $IntervalMs
    }
} finally { $w.Close() }

# Host-side CPU clock logger for the CPU benchmarks (run on the Windows host next to host_rapl_sampler.ps1; WSL2 containers cannot see the real core clocks).
#   powershell -ExecutionPolicy Bypass -File report\hw\benchmark\host_cpu_clock_sampler.ps1 -Out <csv>        stop with Ctrl+C / kill
# Like host_rapl_sampler.ps1 it samples ONLY while power_logs\sample.flag exists (held by power_meter.HostRaplMeter between meter.start() and meter.stop(), i.e. during the idle windows and the
# timed streams of a --power run), so it never runs during the single-image latency timing.
# Columns: unix_s (host clock), p0..p15 = "% Processor Performance" per logical CPU (100 = nominal base clock of that core class, >100 = turbo), u0..u15 = "% Processor Utility" per logical CPU.
# Effective MHz is derived afterwards (summarize_benchmark_clocks.py): i7-1260P logical CPUs 0-7 = P-core threads (nominal 2100 MHz), 8-15 = E-cores (nominal 1500 MHz). Windows' own
# "Processor Frequency" counter is not used: on this hybrid CPU it jumps between 2100 / 1453 / 1500 for the same core class.
param([string]$Out = "$PSScriptRoot\power_logs\cpu_clock.csv", [int]$IntervalMs = 250, [string]$Flag = "$PSScriptRoot\power_logs\sample.flag")
New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
$cat = 'Processor Information'
$inst = (New-Object System.Diagnostics.PerformanceCounterCategory($cat)).GetInstanceNames() | Where-Object { $_ -notmatch '_Total' } | Sort-Object { [int](($_ -split ',')[1]) }
$perf = @(); $util = @()
foreach ($i in $inst) {
    $perf += New-Object System.Diagnostics.PerformanceCounter($cat, '% Processor Performance', $i, $true)
    $util += New-Object System.Diagnostics.PerformanceCounter($cat, '% Processor Utility', $i, $true)
}
function Prime { foreach ($c in $perf) { $null = $c.NextValue() }; foreach ($c in $util) { $null = $c.NextValue() } }
Prime
$w = New-Object System.IO.StreamWriter($Out, $false)
$w.AutoFlush = $true
$n = $inst.Count
$w.WriteLine("unix_s," + ((0..($n - 1) | ForEach-Object { "p$_" }) -join ",") + "," + ((0..($n - 1) | ForEach-Object { "u$_" }) -join ","))
$epoch = [DateTime]::SpecifyKind([DateTime]'1970-01-01', 'Utc')
$active = $false
try {
    while ($true) {
        if (-not (Test-Path $Flag)) { $active = $false; Start-Sleep -Milliseconds 100; continue }
        if (-not $active) { Prime; Start-Sleep -Milliseconds $IntervalMs; $active = $true }      # counters report the change since the previous read: re-prime after a pause
        $t = ([DateTime]::UtcNow - $epoch).TotalSeconds
        $p = $perf | ForEach-Object { '{0:F1}' -f $_.NextValue() }
        $u = $util | ForEach-Object { '{0:F1}' -f $_.NextValue() }
        $w.WriteLine(("{0:F3}," -f $t) + ($p -join ",") + "," + ($u -join ","))
        Start-Sleep -Milliseconds $IntervalMs
    }
} finally { $w.Close() }

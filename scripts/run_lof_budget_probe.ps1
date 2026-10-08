param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [int]$NumSamples = 3,
    [int]$SourceStart = 120,
    [string]$Root = "results\lof_budget_probe_20261008",
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$Rows = @()
foreach ($Dataset in @("gmc", "lending-club")) {
    foreach ($Variant in @("csvae_transformer", "standard_vae")) {
        $Summary = Import-Csv results\final_main_ablation_3000\per_seed.csv |
            Where-Object {$_.dataset -eq $Dataset -and $_.variant -eq $Variant -and $_.seed -eq "42"} |
            Select-Object -First 1
        if (-not $Summary) { throw "Missing reference configuration $Dataset/$Variant" }
        $Config = Join-Path (Split-Path -Parent $Summary.summary_path) "config.json"
        $Output = Join-Path $Root "$Dataset-$Variant"
        New-Item -ItemType Directory -Force -Path $Output | Out-Null
        $Complete = Join-Path $Output "complete.json"
        if ($Resume -and (Test-Path -LiteralPath $Complete)) {
            $Previous = Get-Content -LiteralPath (Join-Path $Output "probe_config.json") -Raw | ConvertFrom-Json
            if ($Previous.source_start -ne $SourceStart -or $Previous.num_samples -ne $NumSamples -or $Previous.probe_tolerance -ne 0.1) {
                throw "Existing probe protocol differs; use a separate result root."
            }
        }
        $Status = @{dataset=$Dataset; variant=$Variant; seed=42; source_start=$SourceStart; num_samples=$NumSamples; stage="generation_and_paired_selection"}
        $Status | ConvertTo-Json | Set-Content (Join-Path $Root "status.json") -Encoding utf8
        if (-not ($Resume -and (Test-Path -LiteralPath $Complete))) {
            & $Python scripts\probe_lof_budget_selection.py --config $Config --output $Output --source-start $SourceStart --num-samples $NumSamples --tolerance 0.1 `
                1> (Join-Path $Output "stdout.log") 2> (Join-Path $Output "stderr.log")
            if ($LASTEXITCODE -ne 0) { throw "Probe failed: $Dataset/$Variant; see $Output/stderr.log" }
        }
        $Pairs = Import-Csv (Join-Path $Output "paired_selection.csv")
        $Row = [ordered]@{dataset=$Dataset; variant=$Variant; seed=42; paired_pools=@($Pairs).Count; attempted_sources=$NumSamples}
        foreach ($Metric in @("diversity", "lof", "target")) {
            foreach ($Selector in @("reference", "budget")) {
                $Column = "$($Selector)_$Metric"
                $Row[$Column] = ($Pairs | ForEach-Object {[double]$_.$Column} | Measure-Object -Average).Average
            }
        }
        $Rows += [pscustomobject]$Row
        $Rows | Export-Csv (Join-Path $Root "comparison.csv") -NoTypeInformation -Encoding utf8
    }
}
@{stage="complete"; completed_runs=4; seed=42; source_start=$SourceStart; num_samples=$NumSamples} |
    ConvertTo-Json | Set-Content (Join-Path $Root "status.json") -Encoding utf8

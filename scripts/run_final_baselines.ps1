param(
    [string]$Python = "python",
    [string[]]$Datasets = @("adult", "bank", "default", "gmc", "lending-club"),
    [string[]]$Methods = @("dice", "dicoflex", "cchvae", "revise", "wachter"),
    [int[]]$Seeds = @(42, 43, 44),
    [int]$BlackBoxSeed = 42,
    [int]$NumSamples = 100,
    [int]$TotalCFs = 10,
    [int]$MaxIter = 3000,
    [string]$RunTag = "final_main_ablation_e2000",
    [string]$ResultsRoot = "results\final_baselines_multiseed_3000\runs",
    [int]$WaitForProcessId = 0,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\final_baselines"
$SummaryLog = Join-Path $LogDir "queue-$Timestamp.summary.log"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null

if ($WaitForProcessId -gt 0) {
    "WAIT  $(Get-Date -Format o) process=$WaitForProcessId" | Tee-Object -FilePath $SummaryLog -Append
    while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 60
    }
    "READY $(Get-Date -Format o) main queue completed" | Tee-Object -FilePath $SummaryLog -Append
}

function Invoke-PipelineCommand {
    param([string[]]$Arguments, [string]$Label)
    $SafeLabel = $Label -replace "[^A-Za-z0-9_-]", "_"
    $Out = Join-Path $LogDir "$SafeLabel-$Timestamp.stdout.log"
    $Err = Join-Path $LogDir "$SafeLabel-$Timestamp.stderr.log"
    "START $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
    $Process = Start-Process -FilePath $Python -ArgumentList (@("main.py") + @($Arguments)) `
        -Wait -PassThru -NoNewWindow -RedirectStandardOutput $Out -RedirectStandardError $Err
    if ($Process.ExitCode -ne 0) {
        throw "$Label failed with exit code $($Process.ExitCode). See $Out and $Err"
    }
    "DONE  $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
}

function Get-TargetClass {
    param([string]$Dataset)
    switch ($Dataset) {
        "adult" { return 1 }
        "bank" { return 1 }
        "default" { return 0 }
        "gmc" { return 0 }
        "lending-club" { return 1 }
        default { throw "No target class for dataset '$Dataset'." }
    }
}

function Get-MethodArgs {
    param([string]$Method)
    switch ($Method) {
        "dice" { return @("--dice_method", "random", "--proximity_weight_input", "1.0", "--diversity_weight", "1.0", "--latent_init_radius", "0.5") }
        "dicoflex" { return @("--dicoflex_epochs", "10", "--dicoflex_p_value", "2.0", "--dicoflex_temperature", "0.8", "--dicoflex_n_nearest", "32", "--dicoflex_train_samples", "2000") }
        "cchvae" { return @() }
        "revise" { return @() }
        "wachter" { return @() }
        default { throw "Unsupported formal baseline '$Method'." }
    }
}

function Test-BaselineComplete {
    param([string]$Dataset, [string]$Method, [int]$TargetClass, [int]$Seed)
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $ResultsRoot -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if (
            $Config.dataname -eq $Dataset -and $Config.method -eq $Method -and
            $Config.run_tag -eq $RunTag -and [int]$Config.black_box_seed -eq $BlackBoxSeed -and
            [int]$Config.seed -eq $Seed -and
            [int]$Config.target_class -eq $TargetClass -and [int]$Config.num_samples -eq $NumSamples -and
            [int]$Config.source_start -eq 0 -and [int]$Config.total_CFs -eq $TotalCFs
        ) {
            return Test-Path -LiteralPath (Join-Path $ConfigPath.DirectoryName "evaluation_summary.csv")
        }
    }
    return $false
}

foreach ($Dataset in $Datasets) {
    $TargetClass = Get-TargetClass -Dataset $Dataset
    $BlackBoxPath = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$RunTag]_seed[$BlackBoxSeed].pkl"
    if (-not (Test-Path -LiteralPath $BlackBoxPath)) {
        throw "Fixed formal black box not found: $BlackBoxPath"
    }

    foreach ($Seed in $Seeds) {
        $SeedResultsRoot = Join-Path $ResultsRoot "seed[$Seed]"
        New-Item -ItemType Directory -Force -Path $SeedResultsRoot | Out-Null
        foreach ($Method in $Methods) {
            if ($Resume -and (Test-BaselineComplete -Dataset $Dataset -Method $Method -TargetClass $TargetClass -Seed $Seed)) {
                "SKIP  $(Get-Date -Format o) $Dataset seed=$Seed method=$Method" | Tee-Object -FilePath $SummaryLog -Append
                continue
            }
            $Args = @(
                "--dataname", $Dataset, "--method", $Method, "--num_encoding", "minmax",
                "--gumbel_softmax", "True", "--sigmoid", "True", "--hidden_dims", "16",
                "--seed", "$Seed", "--black_box_seed", "$BlackBoxSeed", "--run_tag", $RunTag,
                "--save_path", $SeedResultsRoot, "--num_samples", "$NumSamples", "--source_start", "0",
                "--target_class", "$TargetClass", "--total_CFs", "$TotalCFs",
                "--min_iter", "500", "--max_iter", "$MaxIter"
            ) + (Get-MethodArgs -Method $Method)
            Invoke-PipelineCommand -Arguments ($Args + @("--mode", "sample")) -Label "sample-$Dataset-$Seed-$Method"
            Invoke-PipelineCommand -Arguments ($Args + @("--mode", "evaluate", "--get_stats", "False", "--get_changed", "False")) -Label "evaluate-$Dataset-$Seed-$Method"
        }
    }
}

& $Python "scripts\collect_evaluation_summaries.py" "--results_root" $ResultsRoot "--output" "results\final_baselines_multiseed_3000\all_methods.csv"
if ($LASTEXITCODE -ne 0) { throw "Final baseline summary failed with exit code $LASTEXITCODE" }
"COMPLETE $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append

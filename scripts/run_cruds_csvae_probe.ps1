param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string[]]$Datasets = @("adult", "bank", "lending-club"),
    [int]$Seed = 42,
    [int]$BlackBoxSeed = 42,
    [int]$Epochs = 1000,
    [int]$NumSamples = 20,
    [int]$MaxIter = 3000,
    [string]$ResultsRoot = "results\cruds_csvae_probe\runs",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$RunTag = "cruds_csvae_probe"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\cruds_csvae_probe"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null
$SummaryLog = Join-Path $LogDir "probe-$Timestamp.summary.log"

function Get-GenerationArgs {
    param([string]$Dataset)
    switch ($Dataset) {
        "adult" { return @("--candidate_CFs", "30", "--diversity_weight", "1.0", "--cf_selection_metric", "category_first", "--plausibility_weight_latent", "0.05", "--posthoc_sparsity", "True", "--posthoc_diversity", "True") }
        "bank" { return @("--candidate_CFs", "30", "--diversity_weight", "3.0", "--cf_selection_metric", "lof_dpp") }
        "lending-club" { return @("--candidate_CFs", "50", "--diversity_weight", "3.0", "--cf_selection_metric", "lof_dpp", "--plausibility_weight_latent", "0.05") }
        default { throw "No generation configuration for $Dataset" }
    }
}

function Invoke-Python {
    param([string[]]$Arguments, [string]$Label)
    $SafeLabel = $Label -replace "[^A-Za-z0-9_-]", "_"
    $Out = Join-Path $LogDir "$SafeLabel-$Timestamp.log"
    $Err = Join-Path $LogDir "$SafeLabel-$Timestamp.stderr.log"
    "START $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
    $Process = Start-Process -FilePath $Python -ArgumentList (@("main.py") + $Arguments) `
        -Wait -PassThru -NoNewWindow -RedirectStandardOutput $Out -RedirectStandardError $Err
    if ($Process.ExitCode -ne 0) {
        throw "$Label failed with exit code $($Process.ExitCode). See $Out and $Err"
    }
    "DONE  $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
}

function Test-EvaluationComplete {
    param([string]$Dataset)
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $ResultsRoot -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if ($Config.dataname -eq $Dataset -and $Config.run_tag -eq $RunTag -and [int]$Config.seed -eq $Seed -and [int]$Config.num_samples -eq $NumSamples) {
            return Test-Path -LiteralPath (Join-Path $ConfigPath.DirectoryName "evaluation_summary.csv")
        }
    }
    return $false
}

foreach ($Dataset in $Datasets) {
    if ($Resume -and (Test-EvaluationComplete -Dataset $Dataset)) {
        "SKIP  $(Get-Date -Format o) $Dataset already complete" | Tee-Object -FilePath $SummaryLog -Append
        continue
    }

    $BlackBoxSource = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[multiseed]_seed[42].pkl"
    $BlackBoxTarget = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$RunTag]_seed[42].pkl"
    if (-not (Test-Path -LiteralPath $BlackBoxSource)) {
        throw "Fixed seed-42 black box not found: $BlackBoxSource"
    }
    Copy-Item -LiteralPath $BlackBoxSource -Destination $BlackBoxTarget -Force

    $CommonArgs = @(
        "--dataname", $Dataset, "--num_encoding", "minmax",
        "--gumbel_softmax", "True", "--sigmoid", "True",
        "--hidden_dims", "16", "--seed", "$Seed",
        "--black_box_seed", "$BlackBoxSeed", "--run_tag", "cruds_csvae_probe",
        "--latent_model", "csvae", "--csvae_backbone", "transformer",
        "--csvae_epochs", "$Epochs", "--validation_fraction", "0.1",
        "--lr_scheduler_patience", "10", "--lr_scheduler_factor", "0.95",
        "--min_lr", "0", "--beta_schedule", "adaptive",
        "--z_adversary_weight", "1.0"
    )
    Invoke-Python -Arguments ($CommonArgs + @("--method", "csvae", "--mode", "train", "--reuse_checkpoint", "True")) -Label "train_$Dataset"

    $ExperimentArgs = $CommonArgs + @(
        "--method", "mdpcf", "--save_path", $ResultsRoot,
        "--num_samples", "$NumSamples", "--total_CFs", "10",
        "--min_iter", "500", "--max_iter", "$MaxIter",
        "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.1",
        "--proximity_input_loss", "reconstruction", "--validity_loss", "hinge",
        "--validity_margin", "1.0", "--latent_init_radius_strategy", "auto_nn",
        "--latent_init_radius_scale", "0.2", "--categorical_straight_through", "True",
        "--enforce_constraints", "False"
    ) + (Get-GenerationArgs -Dataset $Dataset)
    Invoke-Python -Arguments ($ExperimentArgs + @("--mode", "sample")) -Label "sample_$Dataset"
    Invoke-Python -Arguments ($ExperimentArgs + @("--mode", "evaluate", "--get_stats", "False", "--get_changed", "False")) -Label "evaluate_$Dataset"
}

& $Python "scripts\summarize_cruds_csvae_probe.py"
if ($LASTEXITCODE -ne 0) {
    throw "Probe summary failed with exit code $LASTEXITCODE"
}

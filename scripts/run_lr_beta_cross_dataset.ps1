param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string[]]$Datasets = @("bank", "default", "gmc", "lending-club"),
    [string[]]$Configs = @("A", "B"),
    [int]$Epochs = 1000,
    [int]$NumSamples = 20,
    [string]$ResultsRoot = "results\lr_beta_cross_dataset\runs"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\lr_beta_cross_dataset"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null
$SummaryPath = Join-Path $LogDir "cross-dataset-$Timestamp.summary.log"

function Get-PolicyArgs {
    param([string]$Config)
    switch ($Config.ToUpperInvariant()) {
        "A" { return @("--lr_scheduler_patience", "10", "--lr_scheduler_factor", "0.95", "--min_lr", "0", "--beta_schedule", "adaptive") }
        "B" { return @("--lr_scheduler_patience", "50", "--lr_scheduler_factor", "0.5", "--min_lr", "1e-6", "--beta_schedule", "adaptive") }
        default { throw "Unknown configuration: $Config" }
    }
}

function Get-GenerationArgs {
    param([string]$Dataset)
    switch ($Dataset) {
        "bank" { return @("--candidate_CFs", "30", "--diversity_weight", "3.0", "--cf_selection_metric", "lof_dpp") }
        "default" { return @("--candidate_CFs", "30", "--diversity_weight", "2.0", "--cf_selection_metric", "lof_dpp") }
        "gmc" { return @("--candidate_CFs", "50", "--diversity_weight", "2.0", "--cf_selection_metric", "dpp") }
        "lending-club" { return @("--candidate_CFs", "50", "--diversity_weight", "3.0", "--cf_selection_metric", "lof_dpp", "--plausibility_weight_latent", "0.05") }
        default { throw "No generation configuration for $Dataset" }
    }
}

function Test-TrainingComplete {
    param([string]$Dataset, [string]$RunTag)
    $Root = "mdpcf\csvae\ckpt\$Dataset"
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $Root -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if ($Config.run_tag -eq $RunTag -and [int]$Config.seed -eq 42) {
            $History = Join-Path $ConfigPath.DirectoryName "training_history.csv"
            $Model = Join-Path $ConfigPath.DirectoryName "model.pt"
            if ((Test-Path -LiteralPath $History) -and (Test-Path -LiteralPath $Model)) {
                return (Import-Csv -LiteralPath $History).Count -eq $Epochs
            }
        }
    }
    return $false
}

function Test-EvaluationComplete {
    param([string]$Dataset, [string]$RunTag)
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $ResultsRoot -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if ($Config.dataname -eq $Dataset -and $Config.run_tag -eq $RunTag -and [int]$Config.seed -eq 42 -and [int]$Config.num_samples -eq $NumSamples) {
            return Test-Path -LiteralPath (Join-Path $ConfigPath.DirectoryName "evaluation_summary.csv")
        }
    }
    return $false
}

function Invoke-Python {
    param([string[]]$Arguments, [string]$Label)
    $SafeLabel = $Label -replace "[^A-Za-z0-9_-]", "_"
    $Out = Join-Path $LogDir "$SafeLabel-$Timestamp.log"
    $Err = Join-Path $LogDir "$SafeLabel-$Timestamp.stderr.log"
    "START $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryPath -Append
    $Process = Start-Process -FilePath $Python -ArgumentList (@("main.py") + $Arguments) -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $Out -RedirectStandardError $Err
    if ($Process.ExitCode -ne 0) {
        throw "$Label failed with exit code $($Process.ExitCode). See $Out and $Err"
    }
    "DONE  $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryPath -Append
}

foreach ($Dataset in $Datasets) {
    $BlackBoxSource = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[multiseed]_seed[42].pkl"
    if (-not (Test-Path -LiteralPath $BlackBoxSource)) { throw "Fixed black box not found: $BlackBoxSource" }
    foreach ($Config in $Configs) {
        $Config = $Config.ToUpperInvariant()
        $RunTag = "lr_beta_probe_$($Config.ToLowerInvariant())"
        $BlackBoxTarget = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$RunTag]_seed[42].pkl"
        Copy-Item -LiteralPath $BlackBoxSource -Destination $BlackBoxTarget -Force
        $CommonArgs = @(
            "--dataname", $Dataset, "--num_encoding", "minmax", "--gumbel_softmax", "True",
            "--sigmoid", "True", "--hidden_dims", "16", "--seed", "42", "--black_box_seed", "42",
            "--source_residual_mode", "encoder_predicted_label", "--run_tag", $RunTag,
            "--latent_model", "csvae", "--csvae_backbone", "transformer", "--csvae_epochs", "$Epochs",
            "--validation_fraction", "0.1"
        ) + (Get-PolicyArgs -Config $Config)

        if (-not (Test-TrainingComplete -Dataset $Dataset -RunTag $RunTag)) {
            Invoke-Python -Arguments ($CommonArgs + @("--method", "csvae", "--mode", "train", "--reuse_checkpoint", "False")) `
                -Label "train_${Dataset}_${Config}"
        }

        if (-not (Test-EvaluationComplete -Dataset $Dataset -RunTag $RunTag)) {
            $ExperimentArgs = $CommonArgs + @(
                "--method", "mdpcf", "--save_path", $ResultsRoot, "--num_samples", "$NumSamples",
                "--total_CFs", "10", "--min_iter", "500", "--max_iter", "3000",
                "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.1",
                "--proximity_input_loss", "reconstruction", "--validity_loss", "hinge", "--validity_margin", "1.0",
                "--latent_init_radius_strategy", "auto_nn", "--latent_init_radius_scale", "0.2",
                "--categorical_straight_through", "True", "--enforce_constraints", "False"
            ) + (Get-GenerationArgs -Dataset $Dataset)
            Invoke-Python -Arguments ($ExperimentArgs + @("--mode", "sample")) -Label "sample_${Dataset}_${Config}"
            Invoke-Python -Arguments ($ExperimentArgs + @("--mode", "evaluate", "--get_stats", "False", "--get_changed", "False")) `
                -Label "evaluate_${Dataset}_${Config}"
        }
    }
}

& $Python "scripts\summarize_lr_beta_cross_dataset.py"
if ($LASTEXITCODE -ne 0) { throw "Cross-dataset summary failed with exit code $LASTEXITCODE" }

param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [int[]]$Seeds = @(43, 44),
    [string[]]$Configs = @("A", "B"),
    [int]$Epochs = 1000,
    [int]$NumSamples = 20,
    [string]$ResultsRoot = "results\lr_beta_probe\runs"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\lr_beta_probe"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null
$SummaryPath = Join-Path $LogDir "multiseed-$Timestamp.summary.log"
$BlackBoxSource = Join-Path $RepoRoot "data\adult\black_box_mlp_hidden_16_run[multiseed]_seed[42].pkl"

function Get-PolicyArgs {
    param([string]$Config)
    switch ($Config.ToUpperInvariant()) {
        "A" { return @("--lr_scheduler_patience", "10", "--lr_scheduler_factor", "0.95", "--min_lr", "0", "--beta_schedule", "adaptive") }
        "B" { return @("--lr_scheduler_patience", "50", "--lr_scheduler_factor", "0.5", "--min_lr", "1e-6", "--beta_schedule", "adaptive") }
        default { throw "Unknown probe configuration: $Config" }
    }
}

function Test-TrainingComplete {
    param([string]$RunTag, [int]$Seed)
    foreach ($ConfigPath in Get-ChildItem -LiteralPath "mdpcf\csvae\ckpt\adult" -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if ($Config.run_tag -eq $RunTag -and [int]$Config.seed -eq $Seed) {
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
    param([string]$RunTag, [int]$Seed)
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $ResultsRoot -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if ($Config.run_tag -eq $RunTag -and [int]$Config.seed -eq $Seed -and [int]$Config.num_samples -eq $NumSamples) {
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

foreach ($Seed in $Seeds) {
    foreach ($Config in $Configs) {
        $Config = $Config.ToUpperInvariant()
        $RunTag = "lr_beta_probe_$($Config.ToLowerInvariant())"
        $BlackBoxTarget = Join-Path $RepoRoot "data\adult\black_box_mlp_hidden_16_run[$RunTag]_seed[42].pkl"
        Copy-Item -LiteralPath $BlackBoxSource -Destination $BlackBoxTarget -Force
        $CommonArgs = @(
            "--dataname", "adult", "--num_encoding", "minmax", "--gumbel_softmax", "True",
            "--sigmoid", "True", "--hidden_dims", "16", "--seed", "$Seed", "--black_box_seed", "42",
            "--source_residual_mode", "encoder_predicted_label", "--run_tag", $RunTag,
            "--latent_model", "csvae", "--csvae_backbone", "transformer", "--csvae_epochs", "$Epochs",
            "--validation_fraction", "0.1"
        ) + (Get-PolicyArgs -Config $Config)

        if (-not (Test-TrainingComplete -RunTag $RunTag -Seed $Seed)) {
            Invoke-Python -Arguments ($CommonArgs + @("--method", "csvae", "--mode", "train", "--reuse_checkpoint", "False")) `
                -Label "train_${Config}_seed${Seed}"
        }

        if (-not (Test-EvaluationComplete -RunTag $RunTag -Seed $Seed)) {
            $ExperimentArgs = $CommonArgs + @(
                "--method", "mdpcf", "--save_path", $ResultsRoot, "--num_samples", "$NumSamples",
                "--total_CFs", "10", "--candidate_CFs", "30", "--min_iter", "500", "--max_iter", "3000",
                "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.1",
                "--proximity_input_loss", "reconstruction", "--validity_loss", "hinge", "--validity_margin", "1.0",
                "--latent_init_radius_strategy", "auto_nn", "--latent_init_radius_scale", "0.2",
                "--categorical_straight_through", "True", "--enforce_constraints", "False",
                "--diversity_weight", "1.0", "--cf_selection_metric", "category_first",
                "--plausibility_weight_latent", "0.05", "--posthoc_sparsity", "True", "--posthoc_diversity", "True"
            )
            Invoke-Python -Arguments ($ExperimentArgs + @("--mode", "sample")) -Label "sample_${Config}_seed${Seed}"
            Invoke-Python -Arguments ($ExperimentArgs + @("--mode", "evaluate", "--get_stats", "False", "--get_changed", "False")) `
                -Label "evaluate_${Config}_seed${Seed}"
        }
    }
}

& $Python "scripts\summarize_lr_beta_multiseed.py"
if ($LASTEXITCODE -ne 0) {
    throw "Multiseed summary failed with exit code $LASTEXITCODE"
}

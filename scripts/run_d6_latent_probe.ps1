param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string[]]$Datasets = @("adult", "default", "gmc"),
    [string]$ResultsRoot = "results\d6_latent_probe\runs",
    [int]$Seed = 42,
    [int]$BlackBoxSeed = 42,
    [int]$NumSamples = 10,
    [int]$SourceStart = 110,
    [int]$MaxIter = 3000,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\d6_latent_probe"
$SummaryLog = Join-Path $LogDir "d6-$Timestamp.summary.log"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null

$Variants = @(
    @{ Name = "csvae"; RunTag = "d6_csvae_e2000"; LatentModel = "csvae" },
    @{ Name = "standard_vae"; RunTag = "d6_standardvae_e2000"; LatentModel = "standard_vae" }
)

function Get-TargetClass {
    param([string]$Dataset)
    switch ($Dataset) {
        "default" { return 0 }
        "gmc" { return 0 }
        default { return 1 }
    }
}

function Get-SearchArgs {
    param([string]$Dataset)
    switch ($Dataset) {
        "adult" { return @("--latent_init_radius_scale", "0.1", "--diversity_weight", "2.0", "--cf_selection_metric", "lof_dpp", "--plausibility_weight_latent", "0.0") }
        "default" { return @("--latent_init_radius_scale", "0.2", "--diversity_weight", "2.0", "--cf_selection_metric", "lof_dpp", "--plausibility_weight_latent", "0.0") }
        "gmc" { return @("--latent_init_radius_scale", "0.2", "--diversity_weight", "3.0", "--cf_selection_metric", "lof_dpp", "--plausibility_weight_latent", "0.0") }
        default { throw "No frozen D6 search configuration for $Dataset" }
    }
}

function Invoke-Python {
    param([string[]]$Arguments, [string]$Label)
    $SafeLabel = $Label -replace "[^A-Za-z0-9_-]", "_"
    $Out = Join-Path $LogDir "$SafeLabel-$Timestamp.log"
    $Err = Join-Path $LogDir "$SafeLabel-$Timestamp.stderr.log"
    $Started = Get-Date
    "START $($Started.ToString('o')) $Label" | Tee-Object -FilePath $SummaryLog -Append
    $Process = Start-Process -FilePath $Python -ArgumentList (@("main.py") + $Arguments) `
        -Wait -PassThru -NoNewWindow -RedirectStandardOutput $Out -RedirectStandardError $Err
    if ($Process.ExitCode -ne 0) {
        throw "$Label failed with exit code $($Process.ExitCode). See $Out and $Err"
    }
    $Finished = Get-Date
    "DONE  $($Finished.ToString('o')) $Label seconds=$([math]::Round(($Finished - $Started).TotalSeconds, 3))" |
        Tee-Object -FilePath $SummaryLog -Append
}

function Test-TrainingComplete {
    param([string]$Dataset, [string]$RunTag, [string]$LatentModel)
    $Root = Join-Path $RepoRoot "mdpcf\csvae\ckpt\$Dataset"
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $Root -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if (
            $Config.run_tag -eq $RunTag -and
            $Config.latent_model -eq $LatentModel -and
            [int]$Config.seed -eq $Seed -and
            [int]$Config.token_dim -eq 6 -and
            [int]$Config.latent_dim -eq 6 -and
            [int]$Config.csvae_epochs -eq 2000
        ) {
            $History = Join-Path $ConfigPath.DirectoryName "training_history.csv"
            $Model = Join-Path $ConfigPath.DirectoryName "model.pt"
            if ((Test-Path -LiteralPath $History) -and (Test-Path -LiteralPath $Model)) {
                return (Import-Csv -LiteralPath $History).Count -eq 2000
            }
        }
    }
    return $false
}

function Test-EvaluationComplete {
    param([string]$SavePath, [string]$Dataset, [string]$RunTag, [string]$LatentModel, [int]$TargetClass)
    if (-not (Test-Path -LiteralPath $SavePath)) { return $false }
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $SavePath -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if (
            $Config.dataname -eq $Dataset -and
            $Config.run_tag -eq $RunTag -and
            $Config.latent_model -eq $LatentModel -and
            [int]$Config.token_dim -eq 6 -and
            [int]$Config.latent_dim -eq 6 -and
            [int]$Config.target_class -eq $TargetClass -and
            [int]$Config.source_start -eq $SourceStart -and
            [int]$Config.num_samples -eq $NumSamples -and
            [int]$Config.max_iter -eq $MaxIter
        ) {
            return Test-Path -LiteralPath (Join-Path $ConfigPath.DirectoryName "evaluation_summary.csv")
        }
    }
    return $false
}

foreach ($Dataset in $Datasets) {
    $TargetClass = Get-TargetClass -Dataset $Dataset
    $FixedBlackBox = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[multiseed]_seed[$BlackBoxSeed].pkl"
    if (-not (Test-Path -LiteralPath $FixedBlackBox)) {
        throw "Fixed black box not found: $FixedBlackBox"
    }

    foreach ($Variant in $Variants) {
        $RunBlackBox = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$($Variant.RunTag)]_seed[$BlackBoxSeed].pkl"
        Copy-Item -LiteralPath $FixedBlackBox -Destination $RunBlackBox -Force
        $CommonArgs = @(
            "--dataname", $Dataset, "--num_encoding", "minmax",
            "--gumbel_softmax", "True", "--sigmoid", "True",
            "--hidden_dims", "16", "--seed", "$Seed",
            "--black_box_seed", "$BlackBoxSeed", "--run_tag", $Variant.RunTag,
            "--latent_model", $Variant.LatentModel, "--csvae_backbone", "transformer",
            "--token_dim", "6", "--latent_dim", "6", "--w_dim", "6",
            "--w_structure", "token", "--w_kl_multiplier", "1.0",
            "--csvae_epochs", "2000", "--validation_fraction", "0.1",
            "--lr_scheduler_patience", "10", "--lr_scheduler_factor", "0.95",
            "--min_lr", "0", "--beta_schedule", "adaptive", "--z_adversary_weight", "1.0"
        )

        if (-not ($Resume -and (Test-TrainingComplete -Dataset $Dataset -RunTag $Variant.RunTag -LatentModel $Variant.LatentModel))) {
            Invoke-Python -Arguments ($CommonArgs + @(
                "--method", "csvae", "--mode", "train", "--reuse_checkpoint", "False"
            )) -Label "train_$Dataset`_$($Variant.Name)"
        }
        else {
            "SKIP  $(Get-Date -Format o) train $Dataset $($Variant.Name)" | Tee-Object -FilePath $SummaryLog -Append
        }

        $SavePath = Join-Path $ResultsRoot "$Dataset\$($Variant.Name)"
        if ($Resume -and (Test-EvaluationComplete -SavePath $SavePath -Dataset $Dataset -RunTag $Variant.RunTag -LatentModel $Variant.LatentModel -TargetClass $TargetClass)) {
            "SKIP  $(Get-Date -Format o) evaluate $Dataset $($Variant.Name)" | Tee-Object -FilePath $SummaryLog -Append
            continue
        }
        $GenerationArgs = $CommonArgs + @(
            "--method", "mdpcf", "--save_path", $SavePath,
            "--target_class", "$TargetClass", "--num_samples", "$NumSamples",
            "--source_start", "$SourceStart", "--total_CFs", "10", "--candidate_CFs", "50",
            "--min_iter", "500", "--max_iter", "$MaxIter",
            "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.0",
            "--proximity_input_loss", "reconstruction", "--validity_loss", "hinge",
            "--validity_margin", "1.0", "--latent_init_radius_strategy", "auto_nn",
            "--categorical_straight_through", "True", "--enforce_constraints", "False"
        ) + (Get-SearchArgs -Dataset $Dataset)
        Invoke-Python -Arguments ($GenerationArgs + @("--mode", "sample")) -Label "sample_$Dataset`_$($Variant.Name)"
        Invoke-Python -Arguments ($GenerationArgs + @(
            "--mode", "evaluate", "--get_stats", "False", "--get_changed", "False"
        )) -Label "evaluate_$Dataset`_$($Variant.Name)"
    }
}

& $Python "scripts\summarize_d6_latent_probe.py" --root $ResultsRoot
if ($LASTEXITCODE -ne 0) { throw "D6 summary failed with exit code $LASTEXITCODE" }
"COMPLETE $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append

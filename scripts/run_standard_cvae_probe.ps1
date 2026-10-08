param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string[]]$Datasets = @("adult", "default", "gmc"),
    [string]$ResultsRoot = "results\standard_cvae_probe\runs",
    [int]$Seed = 42,
    [int]$BlackBoxSeed = 42,
    [int]$SourceStart = 110,
    [int]$NumSamples = 10,
    [int]$MaxIter = 3000,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$RunTag = "standard_cvae_val10_e2000"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\standard_cvae_probe"
$SummaryLog = Join-Path $LogDir "cvae-$Timestamp.summary.log"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null

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
        "adult" { return @("--latent_init_radius_scale", "0.1", "--diversity_weight", "2.0") }
        "default" { return @("--latent_init_radius_scale", "0.2", "--diversity_weight", "2.0") }
        "gmc" { return @("--latent_init_radius_scale", "0.2", "--diversity_weight", "3.0") }
        default { throw "No frozen CVAE search configuration for $Dataset" }
    }
}

function Invoke-Python {
    param([string[]]$Arguments, [string]$Label)
    $Out = Join-Path $LogDir "$Label-$Timestamp.log"
    $Err = Join-Path $LogDir "$Label-$Timestamp.stderr.log"
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
    param([string]$Dataset)
    $Root = Join-Path $RepoRoot "mdpcf\csvae\ckpt\$Dataset"
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $Root -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if (
            $Config.run_tag -eq $RunTag -and
            $Config.latent_model -eq "cvae" -and
            [int]$Config.seed -eq $Seed -and
            [int]$Config.black_box_seed -eq $BlackBoxSeed -and
            [int]$Config.token_dim -eq 4 -and
            [int]$Config.latent_dim -eq 4 -and
            [int]$Config.csvae_epochs -eq 2000 -and
            [double]$Config.validation_fraction -eq 0.1
        ) {
            $History = Join-Path $ConfigPath.DirectoryName "training_history.csv"
            $Metadata = Join-Path $ConfigPath.DirectoryName "latent_artifact_protocol.json"
            if ((Test-Path -LiteralPath $History) -and (Test-Path -LiteralPath $Metadata)) {
                $Protocol = Get-Content -LiteralPath $Metadata -Raw | ConvertFrom-Json
                return (Import-Csv -LiteralPath $History).Count -eq 2000 -and
                    $Protocol.latent_artifact_protocol -eq "conditional_vae_v1"
            }
        }
    }
    return $false
}

function Test-EvaluationComplete {
    param([string]$Dataset, [int]$TargetClass)
    if (-not (Test-Path -LiteralPath $ResultsRoot)) { return $false }
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $ResultsRoot -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if (
            $Config.dataname -eq $Dataset -and
            $Config.run_tag -eq $RunTag -and
            $Config.latent_model -eq "cvae" -and
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
    $RunBlackBox = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$RunTag]_seed[$BlackBoxSeed].pkl"
    if (-not (Test-Path -LiteralPath $FixedBlackBox)) {
        throw "Fixed black box not found: $FixedBlackBox"
    }
    Copy-Item -LiteralPath $FixedBlackBox -Destination $RunBlackBox -Force

    $CommonArgs = @(
        "--dataname", $Dataset, "--num_encoding", "minmax",
        "--gumbel_softmax", "True", "--sigmoid", "True",
        "--hidden_dims", "16", "--seed", "$Seed", "--black_box_seed", "$BlackBoxSeed",
        "--run_tag", $RunTag, "--latent_model", "cvae", "--csvae_backbone", "transformer",
        "--token_dim", "4", "--latent_dim", "4", "--csvae_epochs", "2000",
        "--validation_fraction", "0.1", "--lr_scheduler_patience", "10",
        "--lr_scheduler_factor", "0.95", "--min_lr", "0", "--beta_schedule", "adaptive"
    )

    if (-not ($Resume -and (Test-TrainingComplete -Dataset $Dataset))) {
        Invoke-Python -Arguments ($CommonArgs + @(
            "--method", "csvae", "--mode", "train", "--reuse_checkpoint", "False"
        )) -Label "train_$Dataset"
    } else {
        "SKIP  $(Get-Date -Format o) train $Dataset" | Tee-Object -FilePath $SummaryLog -Append
    }

    if ($Resume -and (Test-EvaluationComplete -Dataset $Dataset -TargetClass $TargetClass)) {
        "SKIP  $(Get-Date -Format o) evaluate $Dataset" | Tee-Object -FilePath $SummaryLog -Append
        continue
    }
    $GenerationArgs = $CommonArgs + @(
        "--method", "mdpcf", "--save_path", $ResultsRoot,
        "--target_class", "$TargetClass", "--source_start", "$SourceStart",
        "--num_samples", "$NumSamples", "--total_CFs", "10", "--candidate_CFs", "50",
        "--min_iter", "500", "--max_iter", "$MaxIter",
        "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.0",
        "--proximity_input_loss", "reconstruction", "--diversity_loss", "average_distance",
        "--cf_selection_metric", "lof_dpp", "--validity_loss", "hinge",
        "--validity_margin", "1.0", "--latent_init_radius_strategy", "auto_nn",
        "--categorical_straight_through", "True", "--enforce_constraints", "False",
        "--plausibility_weight_latent", "0.0"
    ) + (Get-SearchArgs -Dataset $Dataset)
    Invoke-Python -Arguments ($GenerationArgs + @("--mode", "sample")) -Label "sample_$Dataset"
    Invoke-Python -Arguments ($GenerationArgs + @(
        "--mode", "evaluate", "--get_stats", "False", "--get_changed", "False"
    )) -Label "evaluate_$Dataset"
}

"COMPLETE $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append

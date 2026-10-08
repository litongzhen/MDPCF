param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string[]]$Datasets = @("bank", "default", "gmc", "lending-club"),
    [int]$Seed = 42,
    [int]$BlackBoxSeed = 42,
    [int]$Epochs = 1000,
    [int]$NumSamples = 20,
    [int]$SourceStart = 100,
    [int]$MaxIter = 3000,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$TransformerTag = "cruds_aligned_multiseed"
$MatchedMlpTag = "cruds_mlp_matched_probe"
$TransformerRoot = "results\cruds_backbone_matched_probe_transformer"
$MatchedMlpRoot = "results\cruds_backbone_matched_probe_matched_mlp"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\backbone_matched_probe"
$SummaryLog = Join-Path $LogDir "probe-$Timestamp.summary.log"
New-Item -ItemType Directory -Force -Path $LogDir, $TransformerRoot, $MatchedMlpRoot | Out-Null

function Get-GenerationArgs {
    param([string]$Dataset)
    switch ($Dataset) {
        "adult" { return @("--candidate_CFs", "30", "--diversity_weight", "1.0", "--cf_selection_metric", "category_first", "--plausibility_weight_latent", "0.05", "--posthoc_sparsity", "True", "--posthoc_diversity", "True") }
        "bank" { return @("--candidate_CFs", "30", "--diversity_weight", "3.0", "--cf_selection_metric", "lof_dpp") }
        "default" { return @("--candidate_CFs", "30", "--diversity_weight", "2.0", "--cf_selection_metric", "lof_dpp") }
        "gmc" { return @("--candidate_CFs", "50", "--diversity_weight", "2.0", "--cf_selection_metric", "dpp") }
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
    param([string]$Root, [string]$Dataset, [string]$RunTag, [string]$Backbone)
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $Root -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if (
            $Config.dataname -eq $Dataset -and
            $Config.run_tag -eq $RunTag -and
            $Config.csvae_backbone -eq $Backbone -and
            [int]$Config.seed -eq $Seed -and
            [int]$Config.black_box_seed -eq $BlackBoxSeed -and
            [int]$Config.num_samples -eq $NumSamples -and
            [int]$Config.source_start -eq $SourceStart -and
            [int]$Config.max_iter -eq $MaxIter
        ) {
            return Test-Path -LiteralPath (Join-Path $ConfigPath.DirectoryName "evaluation_summary.csv")
        }
    }
    return $false
}

foreach ($Dataset in $Datasets) {
    $FixedBlackBox = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[multiseed]_seed[$BlackBoxSeed].pkl"
    if (-not (Test-Path -LiteralPath $FixedBlackBox)) {
        throw "Fixed black box not found: $FixedBlackBox"
    }

    foreach ($Variant in @(
        @{ Name = "transformer"; Tag = $TransformerTag; Root = $TransformerRoot; Match = "False" },
        @{ Name = "mlp"; Tag = $MatchedMlpTag; Root = $MatchedMlpRoot; Match = "True" }
    )) {
        if ($Resume -and (Test-EvaluationComplete -Root $Variant.Root -Dataset $Dataset -RunTag $Variant.Tag -Backbone $Variant.Name)) {
            "SKIP  $(Get-Date -Format o) $Dataset $($Variant.Name) already complete" | Tee-Object -FilePath $SummaryLog -Append
            continue
        }

        $BlackBoxTarget = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$($Variant.Tag)]_seed[$BlackBoxSeed].pkl"
        Copy-Item -LiteralPath $FixedBlackBox -Destination $BlackBoxTarget -Force

        $CommonArgs = @(
            "--dataname", $Dataset, "--num_encoding", "minmax",
            "--gumbel_softmax", "True", "--sigmoid", "True",
            "--hidden_dims", "16", "--seed", "$Seed",
            "--black_box_seed", "$BlackBoxSeed", "--run_tag", $Variant.Tag,
            "--latent_model", "csvae", "--csvae_backbone", $Variant.Name,
            "--mlp_parameter_match", $Variant.Match,
            "--w_structure", "token", "--w_dim", "4", "--w_kl_multiplier", "1.0",
            "--csvae_epochs", "$Epochs", "--validation_fraction", "0.0",
            "--lr_scheduler_patience", "10", "--lr_scheduler_factor", "0.95",
            "--min_lr", "0", "--beta_schedule", "adaptive",
            "--z_adversary_weight", "1.0"
        )

        Invoke-Python -Arguments ($CommonArgs + @(
            "--method", "csvae", "--mode", "train", "--reuse_checkpoint", "True"
        )) -Label "train_$Dataset`_$($Variant.Name)"

        $ExperimentArgs = $CommonArgs + @(
            "--method", "mdpcf", "--save_path", $Variant.Root,
            "--num_samples", "$NumSamples", "--source_start", "$SourceStart",
            "--total_CFs", "10", "--min_iter", "500", "--max_iter", "$MaxIter",
            "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.0",
            "--proximity_input_loss", "reconstruction", "--validity_loss", "hinge",
            "--validity_margin", "1.0", "--latent_init_radius_strategy", "auto_nn",
            "--latent_init_radius_scale", "0.2", "--categorical_straight_through", "True",
            "--enforce_constraints", "False"
        ) + (Get-GenerationArgs -Dataset $Dataset)

        Invoke-Python -Arguments ($ExperimentArgs + @("--mode", "sample")) -Label "sample_$Dataset`_$($Variant.Name)"
        Invoke-Python -Arguments ($ExperimentArgs + @(
            "--mode", "evaluate", "--get_stats", "False", "--get_changed", "False"
        )) -Label "evaluate_$Dataset`_$($Variant.Name)"
    }
}

"COMPLETE $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append

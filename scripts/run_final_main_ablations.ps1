param(
    [string]$Python = "python",
    [string[]]$Datasets = @("adult", "bank", "default", "gmc", "lending-club"),
    [int[]]$Seeds = @(42, 43, 44),
    [int]$BlackBoxSeed = 42,
    [string[]]$Variants = @("csvae_transformer", "csvae_mlp", "standard_vae"),
    [int]$NumSamples = 100,
    [int]$TotalCFs = 10,
    [int]$MinIter = 500,
    [int]$MaxIter = 3000,
    [int]$CSVAEpochs = 2000,
    [string]$RunTag = "final_main_ablation_e2000",
    [string]$BlackBoxRunTag = "multiseed",
    [string]$ResultsRoot = "results\final_main_ablation_3000\runs",
    [string]$OutputDir = "results\final_main_ablation_3000",
    [switch]$SmokeOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

if ($SmokeOnly) {
    $Datasets = @("adult")
    $Seeds = @(42)
    $Variants = @("csvae_transformer")
    $NumSamples = 1
}

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\final_main_ablation"
$SummaryLog = Join-Path $LogDir "queue-$Timestamp.summary.log"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null

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

function Get-DatasetGenerationArgs {
    param([string]$Dataset)
    switch ($Dataset) {
        "adult" { return @("--candidate_CFs", "50", "--diversity_weight", "2.0", "--cf_selection_metric", "lof_dpp", "--latent_init_radius_scale", "0.1", "--plausibility_weight_latent", "0.0") }
        "bank" { return @("--candidate_CFs", "50", "--diversity_weight", "3.0", "--cf_selection_metric", "category_first", "--latent_init_radius_scale", "0.2", "--plausibility_weight_latent", "0.0") }
        "default" { return @("--candidate_CFs", "50", "--diversity_weight", "2.0", "--cf_selection_metric", "lof_dpp", "--latent_init_radius_scale", "0.2", "--plausibility_weight_latent", "0.0") }
        "gmc" { return @("--candidate_CFs", "50", "--diversity_weight", "3.0", "--cf_selection_metric", "lof_dpp", "--latent_init_radius_scale", "0.2", "--plausibility_weight_latent", "0.0") }
        "lending-club" { return @("--candidate_CFs", "50", "--diversity_weight", "3.0", "--cf_selection_metric", "lof_dpp", "--latent_init_radius_scale", "0.2", "--plausibility_weight_latent", "0.05") }
        default { throw "No final generation configuration for dataset '$Dataset'." }
    }
}

function Get-VariantArgs {
    param([string]$Variant)
    switch ($Variant) {
        "csvae_transformer" { return @("--latent_model", "csvae", "--csvae_backbone", "transformer", "--mlp_parameter_match", "False") }
        "csvae_mlp" { return @("--latent_model", "csvae", "--csvae_backbone", "mlp", "--mlp_parameter_match", "True") }
        "standard_vae" { return @("--latent_model", "standard_vae", "--csvae_backbone", "transformer", "--mlp_parameter_match", "False") }
        default { throw "Unknown variant '$Variant'." }
    }
}

function Test-SeededEvaluationComplete {
    param([string]$Dataset, [int]$Seed, [string]$Variant, [int]$TargetClass)

    foreach ($ConfigPath in Get-ChildItem -LiteralPath $ResultsRoot -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        $ConfigVariant = if ($Config.latent_model -eq "standard_vae") { "standard_vae" } elseif ($Config.csvae_backbone -eq "mlp") { "csvae_mlp" } else { "csvae_transformer" }
        if (
            $Config.dataname -eq $Dataset -and $ConfigVariant -eq $Variant -and
            [int]$Config.seed -eq $Seed -and [int]$Config.black_box_seed -eq $BlackBoxSeed -and
            $Config.run_tag -eq $RunTag -and [int]$Config.target_class -eq $TargetClass -and
            [int]$Config.num_samples -eq $NumSamples -and [int]$Config.source_start -eq 0 -and
            [int]$Config.max_iter -eq $MaxIter -and [int]$Config.csvae_epochs -eq $CSVAEpochs -and
            [double]$Config.validation_fraction -eq 0.1 -and
            ($Variant -ne "csvae_mlp" -or [bool]$Config.mlp_parameter_match)
        ) {
            return Test-Path -LiteralPath (Join-Path $ConfigPath.DirectoryName "evaluation_summary.csv")
        }
    }
    return $false
}

foreach ($Dataset in $Datasets) {
    $TargetClass = Get-TargetClass -Dataset $Dataset
    $BlackBoxArgs = @(
        "--dataname", $Dataset, "--num_encoding", "minmax", "--gumbel_softmax", "True",
        "--sigmoid", "True", "--hidden_dims", "16", "--seed", "$BlackBoxSeed",
        "--black_box_seed", "$BlackBoxSeed", "--run_tag", $BlackBoxRunTag
    )
    Invoke-PipelineCommand -Arguments ($BlackBoxArgs + @("--method", "black_box", "--mode", "train")) -Label "$Dataset-fixed-black-box"
    $BlackBoxSource = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$BlackBoxRunTag]_seed[$BlackBoxSeed].pkl"
    $BlackBoxTarget = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$RunTag]_seed[$BlackBoxSeed].pkl"
    Copy-Item -LiteralPath $BlackBoxSource -Destination $BlackBoxTarget -Force

    foreach ($Seed in $Seeds) {
        $CommonArgs = @(
            "--dataname", $Dataset, "--num_encoding", "minmax", "--gumbel_softmax", "True",
            "--sigmoid", "True", "--hidden_dims", "16", "--seed", "$Seed",
            "--black_box_seed", "$BlackBoxSeed", "--run_tag", $RunTag,
            "--w_structure", "token", "--w_dim", "4", "--w_kl_multiplier", "1.0",
            "--csvae_epochs", "$CSVAEpochs", "--validation_fraction", "0.1",
            "--lr_scheduler_patience", "10", "--lr_scheduler_factor", "0.95",
            "--min_lr", "0", "--beta_schedule", "adaptive", "--z_adversary_weight", "1.0"
        )

        foreach ($Variant in $Variants) {
            if ($Resume -and (Test-SeededEvaluationComplete -Dataset $Dataset -Seed $Seed -Variant $Variant -TargetClass $TargetClass)) {
                "SKIP  $(Get-Date -Format o) $Dataset seed=$Seed variant=$Variant" | Tee-Object -FilePath $SummaryLog -Append
                continue
            }
            $VariantArgs = Get-VariantArgs -Variant $Variant
            Invoke-PipelineCommand -Arguments ($CommonArgs + $VariantArgs + @("--method", "csvae", "--mode", "train", "--reuse_checkpoint", "True")) -Label "train-$Dataset-$Seed-$Variant"

            $ExperimentArgs = $CommonArgs + $VariantArgs + @(
                "--method", "mdpcf", "--save_path", $ResultsRoot,
                "--num_samples", "$NumSamples", "--source_start", "0", "--target_class", "$TargetClass",
                "--total_CFs", "$TotalCFs", "--min_iter", "$MinIter", "--max_iter", "$MaxIter",
                "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.0",
                "--proximity_input_loss", "reconstruction", "--validity_loss", "hinge",
                "--validity_margin", "1.0", "--latent_init_radius_strategy", "auto_nn",
                "--categorical_straight_through", "True", "--enforce_constraints", "False"
            ) + (Get-DatasetGenerationArgs -Dataset $Dataset)
            Invoke-PipelineCommand -Arguments ($ExperimentArgs + @("--mode", "sample")) -Label "sample-$Dataset-$Seed-$Variant"
            Invoke-PipelineCommand -Arguments ($ExperimentArgs + @("--mode", "evaluate", "--get_stats", "False", "--get_changed", "False")) -Label "evaluate-$Dataset-$Seed-$Variant"
        }
    }
}

& $Python "scripts\summarize_multiseed_ablations.py" "--results-root" $ResultsRoot "--output-dir" $OutputDir
if ($LASTEXITCODE -ne 0) { throw "Final multiseed summary failed with exit code $LASTEXITCODE" }
"COMPLETE $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append

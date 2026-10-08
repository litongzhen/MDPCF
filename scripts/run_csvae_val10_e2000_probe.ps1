param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string[]]$Datasets = @("adult", "bank", "lending-club"),
    [int]$Seed = 42,
    [int]$BlackBoxSeed = 42,
    [int]$Epochs = 2000,
    [int]$NumSamples = 20,
    [int]$SourceStart = 100,
    [int]$MaxIter = 3000,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$CSVAEtag = "cruds_tokenw_w4_wkl1_e2000"
$StandardTag = "cruds_standardvae_val10_e2000"
$ResultsRoot = "results\csvae_val10_e2000_probe"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\csvae_val10_e2000_probe"
$SummaryLog = Join-Path $LogDir "probe-$Timestamp.summary.log"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null

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
    $Started = Get-Date
    "START $($Started.ToString('o')) $Label" | Tee-Object -FilePath $SummaryLog -Append
    $Process = Start-Process -FilePath $Python -ArgumentList (@("main.py") + $Arguments) `
        -Wait -PassThru -NoNewWindow -RedirectStandardOutput $Out -RedirectStandardError $Err
    $Finished = Get-Date
    if ($Process.ExitCode -ne 0) {
        throw "$Label failed with exit code $($Process.ExitCode). See $Out and $Err"
    }
    "DONE  $($Finished.ToString('o')) $Label seconds=$([math]::Round(($Finished - $Started).TotalSeconds, 3))" | Tee-Object -FilePath $SummaryLog -Append
}

function Test-EvaluationComplete {
    param([string]$Dataset, [string]$LatentModel, [string]$RunTag)
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $ResultsRoot -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if (
            $Config.dataname -eq $Dataset -and
            $Config.latent_model -eq $LatentModel -and
            $Config.run_tag -eq $RunTag -and
            [int]$Config.seed -eq $Seed -and
            [int]$Config.black_box_seed -eq $BlackBoxSeed -and
            [double]$Config.validation_fraction -eq 0.1 -and
            [int]$Config.csvae_epochs -eq $Epochs -and
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
    foreach ($Variant in @(
        @{ Model = "csvae"; Tag = $CSVAEtag },
        @{ Model = "standard_vae"; Tag = $StandardTag }
    )) {
        if ($Resume -and (Test-EvaluationComplete -Dataset $Dataset -LatentModel $Variant.Model -RunTag $Variant.Tag)) {
            "SKIP  $(Get-Date -Format o) $Dataset $($Variant.Model) already complete" | Tee-Object -FilePath $SummaryLog -Append
            continue
        }

        $BlackBoxTarget = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$($Variant.Tag)]_seed[$BlackBoxSeed].pkl"
        Copy-Item -LiteralPath $FixedBlackBox -Destination $BlackBoxTarget -Force
        $VariantArgs = if ($Variant.Model -eq "csvae") {
            @("--latent_model", "csvae", "--csvae_backbone", "transformer", "--w_structure", "token", "--w_dim", "4", "--w_kl_multiplier", "1.0")
        }
        else {
            @("--latent_model", "standard_vae", "--csvae_backbone", "transformer")
        }
        $CommonArgs = @(
            "--dataname", $Dataset, "--num_encoding", "minmax",
            "--gumbel_softmax", "True", "--sigmoid", "True",
            "--hidden_dims", "16", "--seed", "$Seed",
            "--black_box_seed", "$BlackBoxSeed", "--run_tag", $Variant.Tag,
            "--csvae_epochs", "$Epochs", "--validation_fraction", "0.1",
            "--lr_scheduler_patience", "10", "--lr_scheduler_factor", "0.95",
            "--min_lr", "0", "--beta_schedule", "adaptive", "--z_adversary_weight", "1.0"
        ) + $VariantArgs

        Invoke-Python -Arguments ($CommonArgs + @(
            "--method", "csvae", "--mode", "train", "--reuse_checkpoint", "True"
        )) -Label "train_$Dataset`_$($Variant.Model)"

        $ExperimentArgs = $CommonArgs + @(
            "--method", "mdpcf", "--save_path", $ResultsRoot,
            "--num_samples", "$NumSamples", "--source_start", "$SourceStart",
            "--total_CFs", "10", "--min_iter", "500", "--max_iter", "$MaxIter",
            "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.0",
            "--proximity_input_loss", "reconstruction", "--validity_loss", "hinge",
            "--validity_margin", "1.0", "--latent_init_radius_strategy", "auto_nn",
            "--latent_init_radius_scale", "0.2", "--categorical_straight_through", "True",
            "--enforce_constraints", "False"
        ) + (Get-GenerationArgs -Dataset $Dataset)

        Invoke-Python -Arguments ($ExperimentArgs + @("--mode", "sample")) -Label "sample_$Dataset`_$($Variant.Model)"
        Invoke-Python -Arguments ($ExperimentArgs + @(
            "--mode", "evaluate", "--get_stats", "False", "--get_changed", "False"
        )) -Label "evaluate_$Dataset`_$($Variant.Model)"
    }
}

"COMPLETE $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append

param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string[]]$Datasets = @("adult", "bank", "default", "gmc", "lending-club"),
    [int[]]$Seeds = @(42, 43, 44),
    [int]$BlackBoxSeed = 42,
    [string[]]$Variants = @("csvae_transformer", "csvae_mlp", "standard_vae"),
    [int]$NumSamples = 100,
    [int]$TotalCFs = 10,
    [int]$MinIter = 500,
    [int]$MaxIter = 3000,
    [int]$CSVAEpochs = 1000,
    [string]$RunTag = "cruds_aligned_multiseed",
    [string]$BlackBoxRunTag = "multiseed",
    [string]$ResultsRoot = "results\cruds_aligned_multiseed_3000\runs",
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
}

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null
$TranscriptPath = Join-Path $LogDir "multiseed-ablation-$Timestamp.transcript.log"
Start-Transcript -Path $TranscriptPath | Out-Null

function Invoke-PipelineCommand {
    param([string[]]$Arguments, [string]$Description)

    Write-Host "===== $Description ====="
    # Start-Process avoids Windows PowerShell's fragile native argument splatting
    # when the dynamically assembled experiment arguments are long.
    $Process = Start-Process -FilePath $Python -ArgumentList (@("main.py") + @($Arguments)) -Wait -PassThru -NoNewWindow
    if ($Process.ExitCode -ne 0) {
        throw "$Description failed with exit code $($Process.ExitCode)"
    }
}

function Get-DatasetGenerationArgs {
    param([string]$Dataset)

    switch ($Dataset) {
        "adult" {
            return @(
                "--candidate_CFs", "30", "--diversity_weight", "1.0",
                "--cf_selection_metric", "category_first",
                "--plausibility_weight_latent", "0.05",
                "--posthoc_sparsity", "True", "--posthoc_diversity", "True"
            )
        }
        "bank" {
            return @(
                "--candidate_CFs", "30", "--diversity_weight", "3.0",
                "--cf_selection_metric", "lof_dpp"
            )
        }
        "default" {
            return @(
                "--candidate_CFs", "30", "--diversity_weight", "2.0",
                "--cf_selection_metric", "lof_dpp"
            )
        }
        "gmc" {
            return @(
                "--candidate_CFs", "50", "--diversity_weight", "2.0",
                "--cf_selection_metric", "dpp"
            )
        }
        "lending-club" {
            return @(
                "--candidate_CFs", "50", "--diversity_weight", "3.0",
                "--cf_selection_metric", "lof_dpp",
                "--plausibility_weight_latent", "0.05"
            )
        }
        default { throw "No final n100 generation configuration for dataset '$Dataset'." }
    }
}

function Get-VariantArgs {
    param([string]$Variant)

    switch ($Variant) {
        "csvae_transformer" { return @("--latent_model", "csvae", "--csvae_backbone", "transformer") }
        "csvae_mlp" { return @("--latent_model", "csvae", "--csvae_backbone", "mlp") }
        "standard_vae" { return @("--latent_model", "standard_vae", "--csvae_backbone", "transformer") }
        default { throw "Unknown variant '$Variant'." }
    }
}

function Test-SeededEvaluationComplete {
    param([string]$Dataset, [int]$Seed, [string]$Variant)

    foreach ($ConfigPath in Get-ChildItem -LiteralPath $ResultsRoot -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        $ConfigVariant = if ($Config.latent_model -eq "standard_vae") {
            "standard_vae"
        }
        elseif ($Config.csvae_backbone -eq "mlp") {
            "csvae_mlp"
        }
        else {
            "csvae_transformer"
        }
        if (
            $Config.dataname -eq $Dataset -and
            [int]$Config.seed -eq $Seed -and
            [int]$Config.black_box_seed -eq $BlackBoxSeed -and
            $Config.run_tag -eq $RunTag -and
            $ConfigVariant -eq $Variant -and
            [int]$Config.num_samples -eq $NumSamples -and
            [int]$Config.source_start -eq 0 -and
            [int]$Config.max_iter -eq $MaxIter -and
            [int]$Config.csvae_epochs -eq $CSVAEpochs -and
            ($Variant -eq "standard_vae" -or ($Config.w_structure -eq "token" -and [int]$Config.w_dim -eq 4))
        ) {
            return Test-Path -LiteralPath (Join-Path $ConfigPath.DirectoryName "evaluation_summary.csv")
        }
    }
    return $false
}

try {
    foreach ($Dataset in $Datasets) {
        $BlackBoxArgs = @(
            "--dataname", $Dataset,
            "--num_encoding", "minmax",
            "--gumbel_softmax", "True",
            "--sigmoid", "True",
            "--hidden_dims", "16",
            "--seed", "$BlackBoxSeed",
            "--black_box_seed", "$BlackBoxSeed",
            "--run_tag", $BlackBoxRunTag
        )
        Invoke-PipelineCommand -Arguments ($BlackBoxArgs + @("--method", "black_box", "--mode", "train")) -Description "$Dataset fixed black-box training (seed $BlackBoxSeed)"
        $BlackBoxSource = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$BlackBoxRunTag]_seed[$BlackBoxSeed].pkl"
        $BlackBoxTarget = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$RunTag]_seed[$BlackBoxSeed].pkl"
        Copy-Item -LiteralPath $BlackBoxSource -Destination $BlackBoxTarget -Force

        foreach ($Seed in $Seeds) {
            $CommonArgs = @(
                "--dataname", $Dataset,
                "--num_encoding", "minmax",
                "--gumbel_softmax", "True",
                "--sigmoid", "True",
                "--hidden_dims", "16",
                "--seed", "$Seed",
                "--black_box_seed", "$BlackBoxSeed",
                "--run_tag", $RunTag,
                "--w_structure", "token",
                "--w_dim", "4",
                "--w_kl_multiplier", "1.0"
            )

            foreach ($Variant in $Variants) {
                $VariantArgs = Get-VariantArgs -Variant $Variant
                $GenerationArgs = Get-DatasetGenerationArgs -Dataset $Dataset
                if ($Resume -and (Test-SeededEvaluationComplete -Dataset $Dataset -Seed $Seed -Variant $Variant)) {
                    Write-Host "Skipping completed $Dataset seed $Seed $Variant."
                    continue
                }
                $ExperimentArgs = $CommonArgs + $VariantArgs + @(
                    "--method", "mdpcf",
                    "--save_path", $ResultsRoot,
                    "--num_samples", "$NumSamples",
                    "--source_start", "0",
                    "--total_CFs", "$TotalCFs",
                    "--min_iter", "$MinIter",
                    "--max_iter", "$MaxIter",
                    "--csvae_epochs", "$CSVAEpochs",
                    "--proximity_weight_input", "1.0",
                    "--proximity_weight_latent", "0.0",
                    "--proximity_input_loss", "reconstruction",
                    "--validity_loss", "hinge",
                    "--validity_margin", "1.0",
                    "--latent_init_radius_strategy", "auto_nn",
                    "--latent_init_radius_scale", "0.2",
                    "--categorical_straight_through", "True",
                    "--enforce_constraints", "False"
                ) + $GenerationArgs

                Invoke-PipelineCommand -Arguments ($CommonArgs + $VariantArgs + @("--method", "csvae", "--mode", "train", "--csvae_epochs", "$CSVAEpochs", "--reuse_checkpoint", "True")) -Description "$Dataset seed $Seed $Variant latent-model training"
                Invoke-PipelineCommand -Arguments ($ExperimentArgs + @("--mode", "sample")) -Description "$Dataset seed $Seed $Variant sampling"
                Invoke-PipelineCommand -Arguments ($ExperimentArgs + @("--mode", "evaluate", "--get_stats", "False", "--get_changed", "False")) -Description "$Dataset seed $Seed $Variant evaluation"
            }
        }
    }
}
finally {
    Stop-Transcript | Out-Null
}

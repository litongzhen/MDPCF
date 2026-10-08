param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [int]$NumSamples = 100,
    [int]$TotalCFs = 3,
    [int]$MinIter = 100,
    [int]$MaxIter = 1000,
    [string[]]$Methods = @("mdpcf", "tabcf", "dice", "wachter", "revise", "cchvae"),
    [string]$DiceMethod = "genetic",
    [double]$DiceProximityWeight = 0.5,
    [switch]$TrainTABCF,
    [switch]$SkipSample
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$TranscriptPath = Join-Path $LogDir "adult-all-methods-tabcf-protocol-$Timestamp.transcript.log"
Start-Transcript -Path $TranscriptPath | Out-Null

try {
    $ResolvedMethods = @()
    foreach ($MethodEntry in $Methods) {
        $ResolvedMethods += $MethodEntry -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    }

    Write-Host "Running Adult experiments with TABCF-style sample selection."
    Write-Host "Protocol: black-box non-target test instances, first N in test order."
    Write-Host "Methods: $($ResolvedMethods -join ', ')"

    $CommonHiddenDims = 16
    Write-Host "Writing fixed protocol manifest..."
    & $Python "scripts\create_protocol_manifest.py" "--dataname" "adult" "--num_samples" "$NumSamples" "--target_class" "1" "--hidden_dims" "$CommonHiddenDims" "--num_encoding" "minmax"
    if ($LASTEXITCODE -ne 0) {
        throw "Protocol manifest creation failed with exit code $LASTEXITCODE"
    }

    foreach ($Method in $ResolvedMethods) {
        Write-Host "===== Method: $Method ====="

        $CommonArgs = @(
            "--dataname", "adult",
            "--method", "$Method",
            "--num_encoding", "minmax",
            "--gumbel_softmax", "True",
            "--sigmoid", "True",
            "--hidden_dims", "$CommonHiddenDims",
            "--num_samples", "$NumSamples",
            "--total_CFs", "$TotalCFs",
            "--min_iter", "$MinIter",
            "--max_iter", "$MaxIter"
        )

        if ($Method -eq "mdpcf") {
            $CommonArgs += @(
                "--proximity_input_loss", "reconstruction",
                "--validity_loss", "hinge",
                "--validity_margin", "1.0",
                "--latent_init_radius", "0.84",
                "--proximity_weight_latent", "0.1",
                "--categorical_straight_through", "True",
                "--posthoc_sparsity", "True"
            )
        }

        if ($Method -eq "dice") {
            $CommonArgs += @(
                "--dice_method", "$DiceMethod",
                "--proximity_weight_input", "$DiceProximityWeight"
            )
        }

        if (($Method -eq "tabcf") -and $TrainTABCF) {
            Write-Host "Training TABCF original VAE..."
            $TrainArgs = @(
                "--dataname", "adult",
                "--method", "tabcf",
                "--mode", "train",
                "--num_encoding", "minmax",
                "--gumbel_softmax", "True",
                "--sigmoid", "True",
                "--hidden_dims", "16"
            )
            & $Python "main.py" @TrainArgs
            if ($LASTEXITCODE -ne 0) {
                throw "TABCF training failed with exit code $LASTEXITCODE"
            }
        }

        if (-not $SkipSample) {
            Write-Host "Sampling $Method..."
            & $Python "main.py" @CommonArgs "--mode" "sample"
            if ($LASTEXITCODE -ne 0) {
                throw "$Method sampling failed with exit code $LASTEXITCODE"
            }
        }
        else {
            Write-Host "Skipping sample for $Method."
        }

        Write-Host "Evaluating $Method..."
        & $Python "main.py" @CommonArgs "--mode" "evaluate" "--get_stats" "False" "--get_changed" "False"
        if ($LASTEXITCODE -ne 0) {
            throw "$Method evaluation failed with exit code $LASTEXITCODE"
        }
    }

    Write-Host "Collecting summaries..."
    & $Python "scripts\collect_evaluation_summaries.py" "--results_root" "counterfactual_results" "--output" "results\evaluation_summaries.csv"
    if ($LASTEXITCODE -ne 0) {
        throw "Summary collection failed with exit code $LASTEXITCODE"
    }

    Write-Host "All done. Summary table: results\evaluation_summaries.csv"
}
finally {
    Stop-Transcript | Out-Null
}

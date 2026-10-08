param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [int]$NumSamples = 100,
    [int]$TotalCFs = 3,
    [int]$MinIter = 100,
    [int]$MaxIter = 1000,
    [double]$LatentInitRadius = 0.84,
    [double]$ProximityWeightLatent = 0.1,
    [switch]$TrainBlackBox,
    [switch]$TrainCSVAE
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$TranscriptPath = Join-Path $LogDir "adult-mdpcf-standard-$Timestamp.transcript.log"
Start-Transcript -Path $TranscriptPath | Out-Null

try {
    Write-Host "Repo root: $RepoRoot"
    Write-Host "Python: $Python"
    Write-Host "Transcript: $TranscriptPath"

    $Common = @(
        "--dataname", "adult",
        "--num_encoding", "minmax",
        "--gumbel_softmax", "True",
        "--sigmoid", "True",
        "--hidden_dims", "16"
    )

    if ($TrainBlackBox) {
        Write-Host "Training black-box classifier..."
        & $Python "main.py" @Common "--method" "black_box" "--mode" "train"
        if ($LASTEXITCODE -ne 0) {
            throw "Black-box training failed with exit code $LASTEXITCODE"
        }
    }
    else {
        Write-Host "Skipping black-box training. Use -TrainBlackBox to enable it."
    }

    if ($TrainCSVAE) {
        Write-Host "Training CSVAE..."
        & $Python "main.py" @Common "--method" "csvae" "--mode" "train"
        if ($LASTEXITCODE -ne 0) {
            throw "CSVAE training failed with exit code $LASTEXITCODE"
        }
    }
    else {
        Write-Host "Skipping CSVAE training. Use -TrainCSVAE to enable it."
    }

    $ExperimentArgs = @(
        "--dataname", "adult",
        "--method", "mdpcf",
        "--num_encoding", "minmax",
        "--gumbel_softmax", "True",
        "--sigmoid", "True",
        "--hidden_dims", "16",
        "--num_samples", "$NumSamples",
        "--total_CFs", "$TotalCFs",
        "--min_iter", "$MinIter",
        "--max_iter", "$MaxIter",
        "--proximity_input_loss", "reconstruction",
        "--validity_loss", "hinge",
        "--validity_margin", "1.0",
        "--latent_init_radius", "$LatentInitRadius",
        "--proximity_weight_latent", "$ProximityWeightLatent",
        "--categorical_straight_through", "True",
        "--posthoc_sparsity", "True"
    )

    Write-Host "Generating counterfactuals..."
    & $Python "main.py" @ExperimentArgs "--mode" "sample"
    if ($LASTEXITCODE -ne 0) {
        throw "Sampling failed with exit code $LASTEXITCODE"
    }

    Write-Host "Evaluating counterfactuals..."
    & $Python "main.py" @ExperimentArgs "--mode" "evaluate" "--get_stats" "False" "--get_changed" "False"
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed with exit code $LASTEXITCODE"
    }

    Write-Host "Evaluation complete. Check evaluation_summary.csv in the printed result directory."
}
finally {
    Stop-Transcript | Out-Null
}

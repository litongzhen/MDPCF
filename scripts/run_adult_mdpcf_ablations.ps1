param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [int]$NumSamples = 100,
    [int]$TotalCFs = 3,
    [int]$MinIter = 100,
    [int]$MaxIter = 1000
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$TranscriptPath = Join-Path $LogDir "adult-mdpcf-ablations-$Timestamp.transcript.log"
Start-Transcript -Path $TranscriptPath | Out-Null

try {
    & $Python "scripts\create_protocol_manifest.py" "--dataname" "adult" "--num_samples" "$NumSamples" "--target_class" "1" "--hidden_dims" "16" "--num_encoding" "minmax"
    if ($LASTEXITCODE -ne 0) {
        throw "Protocol manifest creation failed with exit code $LASTEXITCODE"
    }

    $BaseArgs = @(
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
        "--latent_init_radius", "0.84",
        "--categorical_straight_through", "True"
    )

    $Ablations = @(
        @{
            Name = "full";
            Args = @("--diversity_weight", "1.0", "--proximity_weight_latent", "0.1", "--posthoc_sparsity", "True")
        },
        @{
            Name = "no_diversity";
            Args = @("--diversity_weight", "0.0", "--proximity_weight_latent", "0.1", "--posthoc_sparsity", "True")
        },
        @{
            Name = "no_latent_proximity";
            Args = @("--diversity_weight", "1.0", "--proximity_weight_latent", "0.0", "--posthoc_sparsity", "True")
        },
        @{
            Name = "no_posthoc_sparsity";
            Args = @("--diversity_weight", "1.0", "--proximity_weight_latent", "0.1", "--posthoc_sparsity", "False")
        }
    )

    foreach ($Ablation in $Ablations) {
        Write-Host "===== MDPCF ablation: $($Ablation.Name) ====="
        $RunArgs = $BaseArgs + $Ablation.Args

        & $Python "main.py" @RunArgs "--mode" "sample"
        if ($LASTEXITCODE -ne 0) {
            throw "MDPCF ablation $($Ablation.Name) sampling failed with exit code $LASTEXITCODE"
        }

        & $Python "main.py" @RunArgs "--mode" "evaluate" "--get_stats" "False" "--get_changed" "False"
        if ($LASTEXITCODE -ne 0) {
            throw "MDPCF ablation $($Ablation.Name) evaluation failed with exit code $LASTEXITCODE"
        }
    }

    & $Python "scripts\collect_evaluation_summaries.py" "--results_root" "counterfactual_results" "--output" "results\evaluation_summaries.csv"
    if ($LASTEXITCODE -ne 0) {
        throw "Summary collection failed with exit code $LASTEXITCODE"
    }
}
finally {
    Stop-Transcript | Out-Null
}

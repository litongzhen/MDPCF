param(
    [string]$Python = "python",
    [int]$Gpu = -1,
    [ValidateRange(1, 100)][int]$NumSamples = 3,
    [ValidateRange(1, 100000)][int]$Epochs = 2000
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
if (-not (Test-Path -LiteralPath "data/adult/info.json")) {
    throw "Prepare Adult first: python download_dataset.py; python process_dataset.py --dataname adult"
}
function Invoke-Stage {
    param([string[]]$Arguments)
    & $Python main.py @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Pipeline stage failed: $Arguments" }
}
$RunTag = "quickstart_e$Epochs"
$OutputPath = "results/quickstart/e$($Epochs)_n$NumSamples"
$Common = @("--dataname", "adult", "--gpu", "$Gpu", "--seed", "42",
    "--black_box_seed", "42", "--run_tag", $RunTag, "--hidden_dims", "16",
    "--num_encoding", "minmax", "--gumbel_softmax", "True", "--sigmoid", "True",
    "--csvae_epochs", "$Epochs", "--w_structure", "token", "--w_dim", "4",
    "--latent_model", "csvae", "--csvae_backbone", "transformer", "--validation_fraction", "0.1")
Invoke-Stage -Arguments ($Common + @("--method", "black_box", "--mode", "train"))
Invoke-Stage -Arguments ($Common + @("--method", "csvae", "--mode", "train", "--reuse_checkpoint", "True"))
$Search = $Common + @("--method", "mdpcf", "--save_path", $OutputPath,
    "--num_samples", "$NumSamples", "--source_start", "0", "--target_class", "1",
    "--total_CFs", "10", "--candidate_CFs", "50", "--min_iter", "500", "--max_iter", "3000",
    "--cf_selection_metric", "lof_dpp", "--latent_init_radius_strategy", "auto_nn",
    "--latent_init_radius_scale", "0.2", "--proximity_weight_input", "1.0",
    "--diversity_weight", "3.0", "--enforce_constraints", "False")
Invoke-Stage -Arguments ($Search + @("--mode", "sample"))
Invoke-Stage -Arguments ($Search + @("--mode", "evaluate", "--get_stats", "False"))
Write-Host "Complete. Find cfs.csv and evaluation_summary.csv under $OutputPath."

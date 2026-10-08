param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string]$ResultsRoot = "results\csvae_search_parameter_probe\confirmation_old",
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
$RunTag = "cruds_tokenw_w4_wkl1_e2000"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\csvae_search_parameter_probe"
$SummaryLog = Join-Path $LogDir "confirmation-old-$Timestamp.summary.log"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null

function Get-OldArgs {
    param([string]$Dataset)
    switch ($Dataset) {
        "adult" { return @("--candidate_CFs", "30", "--diversity_weight", "1.0", "--cf_selection_metric", "category_first", "--plausibility_weight_latent", "0.05", "--posthoc_sparsity", "True", "--posthoc_diversity", "True") }
        "bank" { return @("--candidate_CFs", "30", "--diversity_weight", "3.0", "--cf_selection_metric", "lof_dpp", "--plausibility_weight_latent", "0.0") }
        "lending-club" { return @("--candidate_CFs", "50", "--diversity_weight", "3.0", "--cf_selection_metric", "lof_dpp", "--plausibility_weight_latent", "0.05") }
    }
}

function Invoke-Python {
    param([string[]]$Arguments, [string]$Label)
    $Out = Join-Path $LogDir "$Label-$Timestamp.log"
    $Err = Join-Path $LogDir "$Label-$Timestamp.stderr.log"
    "START $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
    $Process = Start-Process -FilePath $Python -ArgumentList (@("main.py") + $Arguments) -Wait -PassThru -NoNewWindow -RedirectStandardOutput $Out -RedirectStandardError $Err
    if ($Process.ExitCode -ne 0) { throw "$Label failed with exit code $($Process.ExitCode)" }
    "DONE  $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
}

foreach ($Dataset in @("adult", "bank", "lending-club")) {
    $SavePath = Join-Path $ResultsRoot $Dataset
    if ($Resume -and (Get-ChildItem -LiteralPath $SavePath -Recurse -Filter "evaluation_summary.csv" -ErrorAction SilentlyContinue)) {
        "SKIP  $(Get-Date -Format o) $Dataset already complete" | Tee-Object -FilePath $SummaryLog -Append
        continue
    }
    $Args = @(
        "--dataname", $Dataset, "--num_encoding", "minmax", "--gumbel_softmax", "True", "--sigmoid", "True",
        "--hidden_dims", "16", "--seed", "$Seed", "--black_box_seed", "$BlackBoxSeed", "--run_tag", $RunTag,
        "--latent_model", "csvae", "--csvae_backbone", "transformer", "--w_structure", "token", "--w_dim", "4",
        "--w_kl_multiplier", "1.0", "--csvae_epochs", "2000", "--validation_fraction", "0.1",
        "--lr_scheduler_patience", "10", "--lr_scheduler_factor", "0.95", "--min_lr", "0", "--beta_schedule", "adaptive",
        "--z_adversary_weight", "1.0", "--method", "mdpcf", "--save_path", $SavePath,
        "--num_samples", "$NumSamples", "--source_start", "$SourceStart", "--total_CFs", "10",
        "--min_iter", "500", "--max_iter", "$MaxIter", "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.0",
        "--proximity_input_loss", "reconstruction", "--validity_loss", "hinge", "--validity_margin", "1.0",
        "--latent_init_radius_strategy", "auto_nn", "--latent_init_radius_scale", "0.2",
        "--categorical_straight_through", "True", "--enforce_constraints", "False"
    ) + (Get-OldArgs -Dataset $Dataset)
    Invoke-Python -Arguments ($Args + @("--mode", "sample")) -Label "sample_old_$Dataset"
    Invoke-Python -Arguments ($Args + @("--mode", "evaluate", "--get_stats", "False", "--get_changed", "False")) -Label "evaluate_old_$Dataset"
}

"COMPLETE $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append

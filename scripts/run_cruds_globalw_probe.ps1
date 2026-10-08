param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string[]]$Datasets = @("adult", "bank", "lending-club"),
    [int]$Epochs = 300,
    [int]$NumSamples = 20,
    [int]$SourceStart = 100,
    [int]$WDim = 2,
    [ValidateSet("global", "token")][string]$WStructure = "global",
    [double]$WKLMultiplier = 1.0,
    [double]$PlausibilityWeight = 0.0,
    [string]$RunTag = "",
    [string]$ResultsRoot = "",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
if (-not $RunTag) { $RunTag = "cruds_$($WStructure)w_w$($WDim)_wkl$($WKLMultiplier)_e$Epochs" }
if (-not $ResultsRoot) { $ResultsRoot = "results\cruds_$($WStructure)w_w$($WDim)_wkl$($WKLMultiplier)_e$($Epochs)_n20_nolatent" }
$LogDir = Join-Path $RepoRoot "logs\cruds_globalw_probe"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$SummaryLog = Join-Path $LogDir "probe-$Timestamp.summary.log"

function Get-GenerationArgs([string]$Dataset) {
    switch ($Dataset) {
        "adult" { return @("--candidate_CFs", "30", "--diversity_weight", "1.0", "--cf_selection_metric", "category_first", "--posthoc_sparsity", "True", "--posthoc_diversity", "True") }
        "bank" { return @("--candidate_CFs", "30", "--diversity_weight", "3.0", "--cf_selection_metric", "lof_dpp") }
        "lending-club" { return @("--candidate_CFs", "50", "--diversity_weight", "3.0", "--cf_selection_metric", "lof_dpp") }
        default { throw "No generation configuration for $Dataset" }
    }
}

function Invoke-Python([string[]]$Arguments, [string]$Label) {
    $Out = Join-Path $LogDir "$Label-$Timestamp.log"
    $Err = Join-Path $LogDir "$Label-$Timestamp.stderr.log"
    "START $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
    $Process = Start-Process -FilePath $Python -ArgumentList (@("main.py") + $Arguments) `
        -Wait -PassThru -NoNewWindow -RedirectStandardOutput $Out -RedirectStandardError $Err
    if ($Process.ExitCode -ne 0) { throw "$Label failed; see $Out and $Err" }
    "DONE  $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
}

function Test-Complete([string]$Dataset) {
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $ResultsRoot -Recurse -Filter config.json -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if ($Config.dataname -eq $Dataset -and $Config.run_tag -eq $RunTag -and [int]$Config.num_samples -eq $NumSamples -and [int]$Config.source_start -eq $SourceStart) {
            return Test-Path -LiteralPath (Join-Path $ConfigPath.DirectoryName "evaluation_summary.csv")
        }
    }
    return $false
}

foreach ($Dataset in $Datasets) {
    if ($Resume -and (Test-Complete $Dataset)) { continue }
    $BlackBoxSource = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[multiseed]_seed[42].pkl"
    $BlackBoxTarget = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$RunTag]_seed[42].pkl"
    Copy-Item -LiteralPath $BlackBoxSource -Destination $BlackBoxTarget -Force
    $Common = @(
        "--dataname", $Dataset, "--num_encoding", "minmax", "--gumbel_softmax", "True",
        "--sigmoid", "True", "--hidden_dims", "16", "--seed", "42", "--black_box_seed", "42",
        "--run_tag", $RunTag, "--latent_model", "csvae", "--csvae_backbone", "transformer",
        "--csvae_epochs", "$Epochs", "--validation_fraction", "0.1", "--w_dim", "$WDim",
        "--w_structure", "$WStructure", "--w_kl_multiplier", "$WKLMultiplier"
    )
    Invoke-Python ($Common + @("--method", "csvae", "--mode", "train", "--reuse_checkpoint", "True")) "train_$Dataset"
    $Experiment = $Common + @(
        "--method", "mdpcf", "--save_path", $ResultsRoot, "--num_samples", "$NumSamples",
        "--source_start", "$SourceStart",
        "--total_CFs", "10", "--min_iter", "500", "--max_iter", "3000",
        "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.0",
        "--proximity_input_loss", "reconstruction", "--validity_loss", "hinge",
        "--validity_margin", "1.0", "--latent_init_radius_strategy", "auto_nn",
        "--latent_init_radius_scale", "0.2", "--categorical_straight_through", "True",
        "--enforce_constraints", "False", "--plausibility_weight_latent", "$PlausibilityWeight"
    ) + (Get-GenerationArgs $Dataset)
    Invoke-Python ($Experiment + @("--mode", "sample")) "sample_$Dataset"
    Invoke-Python ($Experiment + @("--mode", "evaluate", "--get_stats", "False", "--get_changed", "False")) "evaluate_$Dataset"
}

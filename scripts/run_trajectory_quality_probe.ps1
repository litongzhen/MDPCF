param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string[]]$Datasets = @("adult", "default", "gmc"),
    [string]$ResultsRoot = "results\trajectory_quality_probe\runs",
    [int]$Seed = 42,
    [int]$BlackBoxSeed = 42,
    [int]$SourceStart = 110,
    [int]$NumSamples = 10,
    [int]$MaxIter = 3000,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\trajectory_quality_probe"
$SummaryLog = Join-Path $LogDir "trajectory-$Timestamp.summary.log"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null

$VariantNames = @("csvae", "standard_vae")

function Get-TargetClass {
    param([string]$Dataset)
    switch ($Dataset) {
        "default" { return 0 }
        "gmc" { return 0 }
        default { return 1 }
    }
}

function Get-RunTag {
    param([string]$Variant)
    if ($Variant -eq "csvae") { return "cruds_tokenw_w4_wkl1_e2000" }
    return "cruds_standardvae_val10_e2000"
}

function Get-SearchArgs {
    param([string]$Dataset)
    switch ($Dataset) {
        "adult" { return @("--latent_init_radius_scale", "0.1", "--diversity_weight", "2.0") }
        "default" { return @("--latent_init_radius_scale", "0.2", "--diversity_weight", "2.0") }
        "gmc" { return @("--latent_init_radius_scale", "0.2", "--diversity_weight", "3.0") }
        default { throw "No frozen search configuration for $Dataset" }
    }
}

function Invoke-Python {
    param([string[]]$Arguments, [string]$Label)
    $Out = Join-Path $LogDir "$Label-$Timestamp.log"
    $Err = Join-Path $LogDir "$Label-$Timestamp.stderr.log"
    $Started = Get-Date
    "START $($Started.ToString('o')) $Label" | Tee-Object -FilePath $SummaryLog -Append
    $Process = Start-Process -FilePath $Python -ArgumentList (@("main.py") + $Arguments) `
        -Wait -PassThru -NoNewWindow -RedirectStandardOutput $Out -RedirectStandardError $Err
    if ($Process.ExitCode -ne 0) {
        throw "$Label failed with exit code $($Process.ExitCode). See $Out and $Err"
    }
    $Finished = Get-Date
    "DONE  $($Finished.ToString('o')) $Label seconds=$([math]::Round(($Finished - $Started).TotalSeconds, 3))" |
        Tee-Object -FilePath $SummaryLog -Append
}

function Test-EvaluationComplete {
    param([string]$SavePath, [string]$Dataset, [string]$Variant, [int]$TargetClass)
    if (-not (Test-Path -LiteralPath $SavePath)) { return $false }
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $SavePath -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if (
            $Config.dataname -eq $Dataset -and
            $Config.latent_model -eq $Variant -and
            [bool]$Config.trajectory_quality_selection -and
            $Config.cf_selection_metric -eq "lof_dpp" -and
            [int]$Config.target_class -eq $TargetClass -and
            [int]$Config.source_start -eq $SourceStart -and
            [int]$Config.num_samples -eq $NumSamples -and
            [int]$Config.max_iter -eq $MaxIter
        ) {
            return Test-Path -LiteralPath (Join-Path $ConfigPath.DirectoryName "evaluation_summary.csv")
        }
    }
    return $false
}

foreach ($Dataset in $Datasets) {
    $TargetClass = Get-TargetClass -Dataset $Dataset
    $FixedBlackBox = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[multiseed]_seed[$BlackBoxSeed].pkl"
    if (-not (Test-Path -LiteralPath $FixedBlackBox)) {
        throw "Fixed black box not found: $FixedBlackBox"
    }
    foreach ($Variant in $VariantNames) {
        $RunTag = Get-RunTag -Variant $Variant
        $RunBlackBox = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$RunTag]_seed[$BlackBoxSeed].pkl"
        Copy-Item -LiteralPath $FixedBlackBox -Destination $RunBlackBox -Force
        $SavePath = Join-Path $ResultsRoot "$Dataset\$Variant"
        if ($Resume -and (Test-EvaluationComplete -SavePath $SavePath -Dataset $Dataset -Variant $Variant -TargetClass $TargetClass)) {
            "SKIP  $(Get-Date -Format o) $Dataset $Variant" | Tee-Object -FilePath $SummaryLog -Append
            continue
        }
        $Args = @(
            "--dataname", $Dataset, "--num_encoding", "minmax",
            "--gumbel_softmax", "True", "--sigmoid", "True",
            "--hidden_dims", "16", "--seed", "$Seed", "--black_box_seed", "$BlackBoxSeed",
            "--run_tag", $RunTag, "--latent_model", $Variant, "--csvae_backbone", "transformer",
            "--token_dim", "4", "--latent_dim", "4", "--w_structure", "token", "--w_dim", "4",
            "--method", "mdpcf", "--save_path", $SavePath,
            "--target_class", "$TargetClass", "--source_start", "$SourceStart", "--num_samples", "$NumSamples",
            "--total_CFs", "10", "--candidate_CFs", "50", "--min_iter", "500", "--max_iter", "$MaxIter",
            "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.0",
            "--proximity_input_loss", "reconstruction", "--diversity_loss", "average_distance",
            "--cf_selection_metric", "lof_dpp", "--trajectory_quality_selection", "True",
            "--validity_loss", "hinge", "--validity_margin", "1.0",
            "--latent_init_radius_strategy", "auto_nn", "--categorical_straight_through", "True",
            "--enforce_constraints", "False", "--plausibility_weight_latent", "0.0"
        ) + (Get-SearchArgs -Dataset $Dataset)
        Invoke-Python -Arguments ($Args + @("--mode", "sample")) -Label "sample_$Dataset`_$Variant"
        Invoke-Python -Arguments ($Args + @(
            "--mode", "evaluate", "--get_stats", "False", "--get_changed", "False"
        )) -Label "evaluate_$Dataset`_$Variant"
    }
}

"COMPLETE $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append

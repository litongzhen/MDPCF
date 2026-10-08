param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string]$SelectedPath = "results\csvae_search_parameter_probe\confirmation_selected.json",
    [string]$ResultsRoot = "results\csvae_search_parameter_probe\confirmation",
    [int]$Seed = 42,
    [int]$BlackBoxSeed = 42,
    [int]$NumSamples = 10,
    [int]$SourceStart = 110,
    [int]$MaxIter = 3000,
    [int]$TargetClass = 1,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$RunTag = "cruds_tokenw_w4_wkl1_e2000"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\csvae_search_parameter_probe"
$SummaryLog = Join-Path $LogDir "confirmation-$Timestamp.summary.log"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null

if (-not (Test-Path -LiteralPath $SelectedPath)) {
    throw "Confirmation selection not found: $SelectedPath"
}
$Selected = Get-Content -LiteralPath $SelectedPath -Raw | ConvertFrom-Json

function Convert-ToInvariantString {
    param([double]$Value)
    return $Value.ToString("0.0", [Globalization.CultureInfo]::InvariantCulture)
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
    if ($Process.ExitCode -ne 0) {
        throw "$Label failed with exit code $($Process.ExitCode). See $Out and $Err"
    }
    $Finished = Get-Date
    "DONE  $($Finished.ToString('o')) $Label seconds=$([math]::Round(($Finished - $Started).TotalSeconds, 3))" |
        Tee-Object -FilePath $SummaryLog -Append
}

function Test-EvaluationComplete {
    param([string]$SavePath, [string]$Dataset, [object]$Choice)
    if (-not (Test-Path -LiteralPath $SavePath)) {
        return $false
    }
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $SavePath -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if (
            $Config.dataname -eq $Dataset -and
            [int]$Config.num_samples -eq $NumSamples -and
            [int]$Config.source_start -eq $SourceStart -and
            [int]$Config.max_iter -eq $MaxIter -and
            [int]$Config.target_class -eq $TargetClass -and
            [math]::Abs([double]$Config.latent_init_radius_scale - [double]$Choice.radius_scale) -lt 1e-9 -and
            [math]::Abs([double]$Config.diversity_weight - [double]$Choice.diversity_weight) -lt 1e-9 -and
            $Config.cf_selection_metric -eq $Choice.selection_metric
        ) {
            return Test-Path -LiteralPath (Join-Path $ConfigPath.DirectoryName "evaluation_summary.csv")
        }
    }
    return $false
}

foreach ($DatasetProperty in $Selected.PSObject.Properties) {
    $Dataset = $DatasetProperty.Name
    $Choice = $DatasetProperty.Value
    $SavePath = Join-Path $ResultsRoot $Dataset
    if ($Resume -and (Test-EvaluationComplete -SavePath $SavePath -Dataset $Dataset -Choice $Choice)) {
        "SKIP  $(Get-Date -Format o) $Dataset already complete" | Tee-Object -FilePath $SummaryLog -Append
        continue
    }

    $FixedBlackBox = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[multiseed]_seed[$BlackBoxSeed].pkl"
    $RunBlackBox = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$RunTag]_seed[$BlackBoxSeed].pkl"
    Copy-Item -LiteralPath $FixedBlackBox -Destination $RunBlackBox -Force
    $Args = @(
        "--dataname", $Dataset, "--num_encoding", "minmax",
        "--gumbel_softmax", "True", "--sigmoid", "True",
        "--hidden_dims", "16", "--seed", "$Seed",
        "--black_box_seed", "$BlackBoxSeed", "--run_tag", $RunTag,
        "--latent_model", "csvae", "--csvae_backbone", "transformer",
        "--w_structure", "token", "--w_dim", "4", "--w_kl_multiplier", "1.0",
        "--csvae_epochs", "2000", "--validation_fraction", "0.1",
        "--lr_scheduler_patience", "10", "--lr_scheduler_factor", "0.95",
        "--min_lr", "0", "--beta_schedule", "adaptive", "--z_adversary_weight", "1.0",
        "--method", "mdpcf", "--save_path", $SavePath,
        "--num_samples", "$NumSamples", "--source_start", "$SourceStart",
        "--target_class", "$TargetClass",
        "--total_CFs", "10", "--candidate_CFs", "50",
        "--min_iter", "500", "--max_iter", "$MaxIter",
        "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.0",
        "--proximity_input_loss", "reconstruction",
        "--diversity_weight", (Convert-ToInvariantString $Choice.diversity_weight),
        "--cf_selection_metric", $Choice.selection_metric,
        "--validity_loss", "hinge", "--validity_margin", "1.0",
        "--latent_init_radius_strategy", "auto_nn",
        "--latent_init_radius_scale", (Convert-ToInvariantString $Choice.radius_scale),
        "--categorical_straight_through", "True",
        "--enforce_constraints", "False", "--plausibility_weight_latent", "0.0"
    )
    Invoke-Python -Arguments ($Args + @("--mode", "sample")) -Label "sample_$Dataset"
    Invoke-Python -Arguments ($Args + @(
        "--mode", "evaluate", "--get_stats", "False", "--get_changed", "False"
    )) -Label "evaluate_$Dataset"
}

"COMPLETE $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append

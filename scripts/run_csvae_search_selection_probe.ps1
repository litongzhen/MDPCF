param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string]$SelectedPath = "results\csvae_search_parameter_probe\stage_b_selected.json",
    [string]$ResultsRoot = "results\csvae_search_parameter_probe\stage_b",
    [int]$Seed = 42,
    [int]$BlackBoxSeed = 42,
    [int]$NumSamples = 10,
    [int]$SourceStart = 100,
    [int]$MaxIter = 3000,
    [int]$TargetClass = 1,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$RunTag = "cruds_tokenw_w4_wkl1_e2000"
$SelectionMetrics = @("dpp", "lof_dpp", "category_first")
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\csvae_search_parameter_probe"
$SummaryLog = Join-Path $LogDir "stage-b-$Timestamp.summary.log"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null

if (-not (Test-Path -LiteralPath $SelectedPath)) {
    throw "Stage-A finalist file not found: $SelectedPath"
}
$Selected = Get-Content -LiteralPath $SelectedPath -Raw | ConvertFrom-Json

function Convert-ToInvariantString {
    param([double]$Value)
    return $Value.ToString("0.0", [Globalization.CultureInfo]::InvariantCulture)
}

function Convert-ConfigId {
    param([string]$ConfigId)
    if ($ConfigId -notmatch '^r(\d{2})_d(\d{2})$') {
        throw "Invalid stage-A config id: $ConfigId"
    }
    return @{
        Radius = [int]$Matches[1] / 100.0
        Diversity = [int]$Matches[2] / 10.0
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
    if ($Process.ExitCode -ne 0) {
        throw "$Label failed with exit code $($Process.ExitCode). See $Out and $Err"
    }
    $Finished = Get-Date
    "DONE  $($Finished.ToString('o')) $Label seconds=$([math]::Round(($Finished - $Started).TotalSeconds, 3))" |
        Tee-Object -FilePath $SummaryLog -Append
}

function Test-EvaluationComplete {
    param(
        [string]$SavePath,
        [string]$Dataset,
        [double]$Radius,
        [double]$Diversity,
        [string]$Selection
    )
    if (-not (Test-Path -LiteralPath $SavePath)) {
        return $false
    }
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $SavePath -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if (
            $Config.dataname -eq $Dataset -and
            $Config.run_tag -eq $RunTag -and
            [int]$Config.seed -eq $Seed -and
            [int]$Config.black_box_seed -eq $BlackBoxSeed -and
            [int]$Config.num_samples -eq $NumSamples -and
            [int]$Config.source_start -eq $SourceStart -and
            [int]$Config.max_iter -eq $MaxIter -and
            [int]$Config.target_class -eq $TargetClass -and
            [math]::Abs([double]$Config.latent_init_radius_scale - $Radius) -lt 1e-9 -and
            [math]::Abs([double]$Config.diversity_weight - $Diversity) -lt 1e-9 -and
            $Config.cf_selection_metric -eq $Selection
        ) {
            return Test-Path -LiteralPath (Join-Path $ConfigPath.DirectoryName "evaluation_summary.csv")
        }
    }
    return $false
}

foreach ($DatasetProperty in $Selected.PSObject.Properties) {
    $Dataset = $DatasetProperty.Name
    $FixedBlackBox = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[multiseed]_seed[$BlackBoxSeed].pkl"
    $RunBlackBox = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$RunTag]_seed[$BlackBoxSeed].pkl"
    if (-not (Test-Path -LiteralPath $FixedBlackBox)) {
        throw "Fixed black box not found: $FixedBlackBox"
    }
    Copy-Item -LiteralPath $FixedBlackBox -Destination $RunBlackBox -Force

    foreach ($ConfigId in $DatasetProperty.Value) {
        $Parsed = Convert-ConfigId -ConfigId $ConfigId
        foreach ($Selection in $SelectionMetrics) {
            $SavePath = Join-Path $ResultsRoot "$Dataset\$ConfigId\$Selection"
            if ($Resume -and (Test-EvaluationComplete -SavePath $SavePath -Dataset $Dataset -Radius $Parsed.Radius -Diversity $Parsed.Diversity -Selection $Selection)) {
                "SKIP  $(Get-Date -Format o) $Dataset $ConfigId $Selection already complete" |
                    Tee-Object -FilePath $SummaryLog -Append
                continue
            }

            $Args = @(
                "--dataname", $Dataset, "--num_encoding", "minmax",
                "--gumbel_softmax", "True", "--sigmoid", "True",
                "--hidden_dims", "16", "--seed", "$Seed",
                "--black_box_seed", "$BlackBoxSeed", "--run_tag", $RunTag,
                "--latent_model", "csvae", "--csvae_backbone", "transformer",
                "--w_structure", "token", "--w_dim", "4", "--w_kl_multiplier", "1.0",
                "--csvae_epochs", "2000", "--validation_fraction", "0.1",
                "--lr_scheduler_patience", "10", "--lr_scheduler_factor", "0.95",
                "--min_lr", "0", "--beta_schedule", "adaptive",
                "--z_adversary_weight", "1.0",
                "--method", "mdpcf", "--save_path", $SavePath,
                "--num_samples", "$NumSamples", "--source_start", "$SourceStart",
                "--target_class", "$TargetClass",
                "--total_CFs", "10", "--candidate_CFs", "50",
                "--min_iter", "500", "--max_iter", "$MaxIter",
                "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.0",
                "--proximity_input_loss", "reconstruction",
                "--diversity_weight", (Convert-ToInvariantString $Parsed.Diversity),
                "--cf_selection_metric", $Selection,
                "--validity_loss", "hinge", "--validity_margin", "1.0",
                "--latent_init_radius_strategy", "auto_nn",
                "--latent_init_radius_scale", (Convert-ToInvariantString $Parsed.Radius),
                "--categorical_straight_through", "True",
                "--enforce_constraints", "False", "--plausibility_weight_latent", "0.0"
            )
            $Label = "$Dataset`_$ConfigId`_$Selection"
            Invoke-Python -Arguments ($Args + @("--mode", "sample")) -Label "sample_$Label"
            Invoke-Python -Arguments ($Args + @(
                "--mode", "evaluate", "--get_stats", "False", "--get_changed", "False"
            )) -Label "evaluate_$Label"
        }
    }
}

"COMPLETE $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append

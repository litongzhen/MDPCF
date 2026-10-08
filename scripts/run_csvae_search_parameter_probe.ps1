param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string[]]$Datasets = @("adult", "bank", "lending-club"),
    [int]$Seed = 42,
    [int]$BlackBoxSeed = 42,
    [int]$NumSamples = 10,
    [int]$SourceStart = 100,
    [int]$MaxIter = 1000,
    [int]$TargetClass = 1,
    [string]$ResultsRoot = "results\csvae_search_parameter_probe\stage_a",
    [string[]]$ConfigIds = @(),
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$RunTag = "cruds_tokenw_w4_wkl1_e2000"
$RadiusScales = @(0.1, 0.2, 0.3)
$DiversityWeights = @(1.0, 2.0, 3.0)
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\csvae_search_parameter_probe"
$SummaryLog = Join-Path $LogDir "stage-a-$Timestamp.summary.log"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null

function Convert-ToInvariantString {
    param([double]$Value)
    return $Value.ToString("0.0", [Globalization.CultureInfo]::InvariantCulture)
}

function Get-ConfigId {
    param([double]$Radius, [double]$Diversity)
    $RadiusId = [int][math]::Round($Radius * 100)
    $DiversityId = [int][math]::Round($Diversity * 10)
    return "r$($RadiusId.ToString('00'))_d$($DiversityId.ToString('00'))"
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
        [double]$Diversity
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
            [int]$Config.candidate_CFs -eq 50 -and
            $Config.cf_selection_metric -eq "lof_dpp"
        ) {
            return Test-Path -LiteralPath (Join-Path $ConfigPath.DirectoryName "evaluation_summary.csv")
        }
    }
    return $false
}

foreach ($Dataset in $Datasets) {
    $FixedBlackBox = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[multiseed]_seed[$BlackBoxSeed].pkl"
    $RunBlackBox = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$RunTag]_seed[$BlackBoxSeed].pkl"
    if (-not (Test-Path -LiteralPath $FixedBlackBox)) {
        throw "Fixed black box not found: $FixedBlackBox"
    }
    Copy-Item -LiteralPath $FixedBlackBox -Destination $RunBlackBox -Force

    foreach ($Radius in $RadiusScales) {
        foreach ($Diversity in $DiversityWeights) {
            $ConfigId = Get-ConfigId -Radius $Radius -Diversity $Diversity
            if ($ConfigIds.Count -gt 0 -and $ConfigId -notin $ConfigIds) {
                continue
            }
            $SavePath = Join-Path $ResultsRoot "$Dataset\$ConfigId"
            if ($Resume -and (Test-EvaluationComplete -SavePath $SavePath -Dataset $Dataset -Radius $Radius -Diversity $Diversity)) {
                "SKIP  $(Get-Date -Format o) $Dataset $ConfigId already complete" |
                    Tee-Object -FilePath $SummaryLog -Append
                continue
            }

            $CommonArgs = @(
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
                "--diversity_weight", (Convert-ToInvariantString $Diversity),
                "--cf_selection_metric", "lof_dpp",
                "--validity_loss", "hinge", "--validity_margin", "1.0",
                "--latent_init_radius_strategy", "auto_nn",
                "--latent_init_radius_scale", (Convert-ToInvariantString $Radius),
                "--categorical_straight_through", "True",
                "--enforce_constraints", "False", "--plausibility_weight_latent", "0.0"
            )

            Invoke-Python -Arguments ($CommonArgs + @("--mode", "sample")) -Label "sample_$Dataset`_$ConfigId"
            Invoke-Python -Arguments ($CommonArgs + @(
                "--mode", "evaluate", "--get_stats", "False", "--get_changed", "False"
            )) -Label "evaluate_$Dataset`_$ConfigId"
        }
    }
}

"COMPLETE $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append

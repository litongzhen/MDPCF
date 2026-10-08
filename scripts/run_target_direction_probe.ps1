param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [int]$WaitForProcessId = 0,
    [int]$Seed = 42,
    [int]$BlackBoxSeed = 42,
    [int]$NumSamples = 20,
    [int]$MaxIter = 3000,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$RunTag = "cruds_aligned_multiseed"
$ResultsRoot = "results\target_direction_probe"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\target_direction_probe"
$SummaryLog = Join-Path $LogDir "probe-$Timestamp.summary.log"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null

if ($WaitForProcessId -gt 0) {
    "WAIT  $(Get-Date -Format o) process $WaitForProcessId" | Tee-Object -FilePath $SummaryLog -Append
    while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
    "READY $(Get-Date -Format o) process $WaitForProcessId finished" | Tee-Object -FilePath $SummaryLog -Append
}

function Invoke-Python {
    param([string[]]$Arguments, [string]$Label)
    $SafeLabel = $Label -replace "[^A-Za-z0-9_-]", "_"
    $Out = Join-Path $LogDir "$SafeLabel-$Timestamp.log"
    $Err = Join-Path $LogDir "$SafeLabel-$Timestamp.stderr.log"
    "START $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
    $Process = Start-Process -FilePath $Python -ArgumentList (@("main.py") + $Arguments) `
        -Wait -PassThru -NoNewWindow -RedirectStandardOutput $Out -RedirectStandardError $Err
    if ($Process.ExitCode -ne 0) {
        throw "$Label failed with exit code $($Process.ExitCode). See $Out and $Err"
    }
    "DONE  $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
}

function Test-EvaluationComplete {
    param([string]$SavePath, [string]$Dataset, [int]$TargetClass, [int]$SourceStart)
    if (-not (Test-Path -LiteralPath $SavePath)) {
        return $false
    }
    foreach ($ConfigPath in Get-ChildItem -LiteralPath $SavePath -Recurse -Filter "config.json" -ErrorAction SilentlyContinue) {
        $Config = Get-Content -LiteralPath $ConfigPath.FullName -Raw | ConvertFrom-Json
        if (
            $Config.dataname -eq $Dataset -and
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

$Experiments = @(
    @{ Id = "gmc_B_target1_lof"; Dataset = "gmc"; Target = 1; SourceStart = 100; Selection = "lof_dpp"; LatentProx = "0.0"; Candidates = "50"; Diversity = "2.0" },
    @{ Id = "gmc_C_target1_prox"; Dataset = "gmc"; Target = 1; SourceStart = 100; Selection = "dpp"; LatentProx = "0.1"; Candidates = "50"; Diversity = "2.0" },
    @{ Id = "gmc_D_target1_lof_prox"; Dataset = "gmc"; Target = 1; SourceStart = 100; Selection = "lof_dpp"; LatentProx = "0.1"; Candidates = "50"; Diversity = "2.0" },
    @{ Id = "gmc_E_target0_dpp"; Dataset = "gmc"; Target = 0; SourceStart = 0; Selection = "dpp"; LatentProx = "0.0"; Candidates = "50"; Diversity = "2.0" },
    @{ Id = "gmc_G_target0_lof"; Dataset = "gmc"; Target = 0; SourceStart = 0; Selection = "lof_dpp"; LatentProx = "0.0"; Candidates = "50"; Diversity = "2.0" },
    @{ Id = "gmc_F_target0_lof_prox"; Dataset = "gmc"; Target = 0; SourceStart = 0; Selection = "lof_dpp"; LatentProx = "0.1"; Candidates = "50"; Diversity = "2.0" },
    @{ Id = "default_target0_lof"; Dataset = "default"; Target = 0; SourceStart = 0; Selection = "lof_dpp"; LatentProx = "0.0"; Candidates = "30"; Diversity = "2.0" },
    @{ Id = "default_target0_lof_prox"; Dataset = "default"; Target = 0; SourceStart = 0; Selection = "lof_dpp"; LatentProx = "0.1"; Candidates = "30"; Diversity = "2.0" }
)

$PreparedDatasets = @{}
foreach ($Experiment in $Experiments) {
    $Dataset = $Experiment.Dataset
    $SavePath = Join-Path $ResultsRoot $Experiment.Id
    if ($Resume -and (Test-EvaluationComplete -SavePath $SavePath -Dataset $Dataset -TargetClass $Experiment.Target -SourceStart $Experiment.SourceStart)) {
        "SKIP  $(Get-Date -Format o) $($Experiment.Id) already complete" | Tee-Object -FilePath $SummaryLog -Append
        continue
    }

    if (-not $PreparedDatasets.ContainsKey($Dataset)) {
        $TrainArgs = @(
            "--dataname", $Dataset, "--num_encoding", "minmax",
            "--gumbel_softmax", "True", "--sigmoid", "True",
            "--hidden_dims", "16", "--seed", "$Seed",
            "--black_box_seed", "$BlackBoxSeed", "--run_tag", $RunTag,
            "--latent_model", "csvae", "--csvae_backbone", "transformer",
            "--w_structure", "token", "--w_dim", "4", "--w_kl_multiplier", "1.0",
            "--csvae_epochs", "1000", "--validation_fraction", "0.0",
            "--lr_scheduler_patience", "10", "--lr_scheduler_factor", "0.95",
            "--min_lr", "0", "--beta_schedule", "adaptive",
            "--z_adversary_weight", "1.0",
            "--method", "csvae", "--mode", "train", "--reuse_checkpoint", "True"
        )
        Invoke-Python -Arguments $TrainArgs -Label "prepare_$Dataset"
        $PreparedDatasets[$Dataset] = $true
    }

    $Args = @(
        "--dataname", $Dataset, "--num_encoding", "minmax",
        "--gumbel_softmax", "True", "--sigmoid", "True",
        "--hidden_dims", "16", "--seed", "$Seed",
        "--black_box_seed", "$BlackBoxSeed", "--run_tag", $RunTag,
        "--latent_model", "csvae", "--csvae_backbone", "transformer",
        "--w_structure", "token", "--w_dim", "4", "--w_kl_multiplier", "1.0",
        "--csvae_epochs", "1000", "--method", "mdpcf",
        "--save_path", $SavePath,
        "--num_samples", "$NumSamples", "--source_start", "$($Experiment.SourceStart)",
        "--target_class", "$($Experiment.Target)",
        "--total_CFs", "10", "--candidate_CFs", $Experiment.Candidates,
        "--min_iter", "500", "--max_iter", "$MaxIter",
        "--proximity_weight_input", "1.0", "--proximity_weight_latent", $Experiment.LatentProx,
        "--proximity_input_loss", "reconstruction",
        "--diversity_weight", $Experiment.Diversity,
        "--cf_selection_metric", $Experiment.Selection,
        "--validity_loss", "hinge", "--validity_margin", "1.0",
        "--latent_init_radius_strategy", "auto_nn", "--latent_init_radius_scale", "0.2",
        "--categorical_straight_through", "True", "--enforce_constraints", "False",
        "--plausibility_weight_latent", "0.0"
    )
    Invoke-Python -Arguments ($Args + @("--mode", "sample")) -Label "sample_$($Experiment.Id)"
    Invoke-Python -Arguments ($Args + @("--mode", "evaluate", "--get_stats", "False", "--get_changed", "False")) -Label "evaluate_$($Experiment.Id)"
}

"COMPLETE $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append

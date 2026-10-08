param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string]$ResultsRoot = "results\csvae_search_parameter_probe\lending_refinement",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\csvae_search_parameter_probe"
$SummaryLog = Join-Path $LogDir "lending-refinement-$Timestamp.summary.log"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null

$Configurations = @(
    @{ Id = "lof_p0"; Selection = "lof_dpp"; Plausibility = "0.0" },
    @{ Id = "lof_p005"; Selection = "lof_dpp"; Plausibility = "0.05" },
    @{ Id = "category_p005"; Selection = "category_first"; Plausibility = "0.05" }
)

foreach ($Config in $Configurations) {
    $SavePath = Join-Path $ResultsRoot $Config.Id
    if ($Resume -and (Get-ChildItem -LiteralPath $SavePath -Recurse -Filter "evaluation_summary.csv" -ErrorAction SilentlyContinue)) {
        "SKIP  $(Get-Date -Format o) $($Config.Id)" | Tee-Object -FilePath $SummaryLog -Append
        continue
    }
    $Args = @(
        "--dataname", "lending-club", "--num_encoding", "minmax", "--gumbel_softmax", "True", "--sigmoid", "True",
        "--hidden_dims", "16", "--seed", "42", "--black_box_seed", "42", "--run_tag", "cruds_tokenw_w4_wkl1_e2000",
        "--latent_model", "csvae", "--csvae_backbone", "transformer", "--w_structure", "token", "--w_dim", "4",
        "--w_kl_multiplier", "1.0", "--csvae_epochs", "2000", "--validation_fraction", "0.1",
        "--lr_scheduler_patience", "10", "--lr_scheduler_factor", "0.95", "--min_lr", "0", "--beta_schedule", "adaptive",
        "--z_adversary_weight", "1.0", "--method", "mdpcf", "--save_path", $SavePath,
        "--num_samples", "10", "--source_start", "110", "--total_CFs", "10", "--candidate_CFs", "50",
        "--min_iter", "500", "--max_iter", "3000", "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.0",
        "--proximity_input_loss", "reconstruction", "--diversity_weight", "3.0", "--cf_selection_metric", $Config.Selection,
        "--validity_loss", "hinge", "--validity_margin", "1.0", "--latent_init_radius_strategy", "auto_nn",
        "--latent_init_radius_scale", "0.1", "--categorical_straight_through", "True", "--enforce_constraints", "False",
        "--plausibility_weight_latent", $Config.Plausibility
    )
    foreach ($Mode in @("sample", "evaluate")) {
        $Label = "$Mode`_$($Config.Id)"
        $Out = Join-Path $LogDir "$Label-$Timestamp.log"
        $Err = Join-Path $LogDir "$Label-$Timestamp.stderr.log"
        "START $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
        $ModeArgs = if ($Mode -eq "evaluate") { @("--mode", $Mode, "--get_stats", "False", "--get_changed", "False") } else { @("--mode", $Mode) }
        $Process = Start-Process -FilePath $Python -ArgumentList (@("main.py") + $Args + $ModeArgs) -Wait -PassThru -NoNewWindow -RedirectStandardOutput $Out -RedirectStandardError $Err
        if ($Process.ExitCode -ne 0) { throw "$Label failed with exit code $($Process.ExitCode)" }
        "DONE  $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
    }
}

"COMPLETE $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append

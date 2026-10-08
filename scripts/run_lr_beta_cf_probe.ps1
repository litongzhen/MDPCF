param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [int]$NumSamples = 20,
    [string]$ResultsRoot = "results\lr_beta_probe\runs"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
New-Item -ItemType Directory -Force -Path $ResultsRoot, "logs\lr_beta_probe" | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$SummaryPath = "logs\lr_beta_probe\cf-probe-$Timestamp.summary.log"

$Selected = (Get-Content -LiteralPath "results\lr_beta_probe\selected_configs.json" -Raw | ConvertFrom-Json).selected_configs
$Configs = @("A") + @($Selected)

function Get-PolicyArgs {
    param([string]$Config)
    switch ($Config.ToUpperInvariant()) {
        "A" { return @("--lr_scheduler_patience", "10", "--lr_scheduler_factor", "0.95", "--min_lr", "0", "--beta_schedule", "adaptive") }
        "B" { return @("--lr_scheduler_patience", "50", "--lr_scheduler_factor", "0.5", "--min_lr", "1e-6", "--beta_schedule", "adaptive") }
        "C" { return @("--lr_scheduler_patience", "50", "--lr_scheduler_factor", "0.5", "--min_lr", "1e-6", "--beta_schedule", "fixed", "--beta_target", "1e-3") }
        "D" { return @("--lr_scheduler_patience", "50", "--lr_scheduler_factor", "0.5", "--min_lr", "1e-6", "--beta_schedule", "linear_warmup", "--beta_target", "1e-3", "--beta_warmup_epochs", "300") }
        default { throw "Unknown probe configuration: $Config" }
    }
}

foreach ($Config in $Configs) {
    $RunTag = "lr_beta_probe_$($Config.ToLowerInvariant())"
    $CommonArgs = @(
        "--dataname", "adult", "--num_encoding", "minmax", "--gumbel_softmax", "True",
        "--sigmoid", "True", "--hidden_dims", "16", "--seed", "42", "--black_box_seed", "42",
        "--source_residual_mode", "encoder_predicted_label", "--run_tag", $RunTag,
        "--latent_model", "csvae", "--csvae_backbone", "transformer", "--csvae_epochs", "1000",
        "--validation_fraction", "0.1"
    ) + (Get-PolicyArgs -Config $Config)
    $ExperimentArgs = $CommonArgs + @(
        "--method", "mdpcf", "--save_path", $ResultsRoot, "--num_samples", "$NumSamples",
        "--total_CFs", "10", "--candidate_CFs", "30", "--min_iter", "500", "--max_iter", "3000",
        "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.1",
        "--proximity_input_loss", "reconstruction", "--validity_loss", "hinge",
        "--validity_margin", "1.0", "--latent_init_radius_strategy", "auto_nn",
        "--latent_init_radius_scale", "0.2", "--categorical_straight_through", "True",
        "--enforce_constraints", "False", "--diversity_weight", "1.0",
        "--cf_selection_metric", "category_first", "--plausibility_weight_latent", "0.05",
        "--posthoc_sparsity", "True", "--posthoc_diversity", "True"
    )

    foreach ($Mode in @("sample", "evaluate")) {
        $LogPath = "logs\lr_beta_probe\adult_${Config}_${Mode}_$Timestamp.log"
        $ErrorLogPath = "logs\lr_beta_probe\adult_${Config}_${Mode}_$Timestamp.stderr.log"
        $ModeArgs = if ($Mode -eq "evaluate") {
            $ExperimentArgs + @("--mode", "evaluate", "--get_stats", "False", "--get_changed", "False")
        } else {
            $ExperimentArgs + @("--mode", "sample")
        }
        "START $(Get-Date -Format o) config=$Config mode=$Mode" | Tee-Object -FilePath $SummaryPath -Append
        $Process = Start-Process -FilePath $Python -ArgumentList (@("main.py") + $ModeArgs) -Wait -PassThru -NoNewWindow `
            -RedirectStandardOutput $LogPath -RedirectStandardError $ErrorLogPath
        if ($Process.ExitCode -ne 0) {
            throw "Config $Config/$Mode failed with exit code $($Process.ExitCode). See $LogPath and $ErrorLogPath"
        }
        "DONE  $(Get-Date -Format o) config=$Config mode=$Mode" | Tee-Object -FilePath $SummaryPath -Append
    }
}

& $Python "scripts\summarize_lr_beta_probe.py" --evaluation-root $ResultsRoot
if ($LASTEXITCODE -ne 0) {
    throw "Counterfactual summary failed with exit code $LASTEXITCODE"
}

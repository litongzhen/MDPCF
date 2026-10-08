param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string]$ConfigPath = "results\csvae_search_parameter_probe\final_selected.json",
    [string]$ResultsRoot = "results\csvae_search_parameter_probe\tuned_standard_vae",
    [int]$TargetClass = 1,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$RunTag = "cruds_standardvae_val10_e2000"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\csvae_search_parameter_probe"
$SummaryLog = Join-Path $LogDir "tuned-vae-$Timestamp.summary.log"
New-Item -ItemType Directory -Force -Path $LogDir, $ResultsRoot | Out-Null
$Selected = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json

function Invariant([double]$Value) { return $Value.ToString("0.0#", [Globalization.CultureInfo]::InvariantCulture) }

foreach ($Property in $Selected.PSObject.Properties) {
    $Dataset = $Property.Name
    $Choice = $Property.Value
    $SavePath = Join-Path $ResultsRoot $Dataset
    if ($Resume -and (Get-ChildItem -LiteralPath $SavePath -Recurse -Filter "evaluation_summary.csv" -ErrorAction SilentlyContinue)) {
        "SKIP  $(Get-Date -Format o) $Dataset" | Tee-Object -FilePath $SummaryLog -Append
        continue
    }
    $Args = @(
        "--dataname", $Dataset, "--num_encoding", "minmax", "--gumbel_softmax", "True", "--sigmoid", "True",
        "--hidden_dims", "16", "--seed", "42", "--black_box_seed", "42", "--run_tag", $RunTag,
        "--latent_model", "standard_vae", "--csvae_backbone", "transformer", "--csvae_epochs", "2000", "--validation_fraction", "0.1",
        "--lr_scheduler_patience", "10", "--lr_scheduler_factor", "0.95", "--min_lr", "0", "--beta_schedule", "adaptive",
        "--method", "mdpcf", "--save_path", $SavePath, "--num_samples", "10", "--source_start", "110",
        "--target_class", "$TargetClass",
        "--total_CFs", "10", "--candidate_CFs", "$($Choice.candidate_cfs)", "--min_iter", "500", "--max_iter", "3000",
        "--proximity_weight_input", "1.0", "--proximity_weight_latent", "0.0", "--proximity_input_loss", "reconstruction",
        "--diversity_weight", (Invariant $Choice.diversity_weight), "--cf_selection_metric", $Choice.selection_metric,
        "--validity_loss", "hinge", "--validity_margin", "1.0", "--latent_init_radius_strategy", "auto_nn",
        "--latent_init_radius_scale", (Invariant $Choice.radius_scale), "--categorical_straight_through", "True",
        "--enforce_constraints", "False", "--plausibility_weight_latent", (Invariant $Choice.plausibility_weight_latent)
    )
    foreach ($Mode in @("sample", "evaluate")) {
        $Label = "$Mode`_$Dataset"
        $Out = Join-Path $LogDir "$Label-tuned-vae-$Timestamp.log"
        $Err = Join-Path $LogDir "$Label-tuned-vae-$Timestamp.stderr.log"
        "START $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
        $ModeArgs = if ($Mode -eq "evaluate") { @("--mode", $Mode, "--get_stats", "False", "--get_changed", "False") } else { @("--mode", $Mode) }
        $Process = Start-Process -FilePath $Python -ArgumentList (@("main.py") + $Args + $ModeArgs) -Wait -PassThru -NoNewWindow -RedirectStandardOutput $Out -RedirectStandardError $Err
        if ($Process.ExitCode -ne 0) { throw "$Label failed with exit code $($Process.ExitCode)" }
        "DONE  $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
    }
}

"COMPLETE $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append

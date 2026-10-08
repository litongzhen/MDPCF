param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [int]$Epochs = 1000,
    [string[]]$Configs = @("A", "B", "C", "D")
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$Dataset = "adult"
$Seed = 42
$BlackBoxSeed = 42
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\lr_beta_probe"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$SummaryPath = Join-Path $LogDir "probe-$Timestamp.summary.log"
$BlackBoxSource = Join-Path $RepoRoot "data\adult\black_box_mlp_hidden_16_run[multiseed]_seed[42].pkl"
if (-not (Test-Path -LiteralPath $BlackBoxSource)) {
    throw "Fixed black box not found: $BlackBoxSource"
}

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
    $Config = $Config.ToUpperInvariant()
    $RunTag = "lr_beta_probe_$($Config.ToLowerInvariant())"
    $BlackBoxTarget = Join-Path $RepoRoot "data\adult\black_box_mlp_hidden_16_run[$RunTag]_seed[42].pkl"
    Copy-Item -LiteralPath $BlackBoxSource -Destination $BlackBoxTarget -Force
    $LogPath = Join-Path $LogDir "adult_seed42_${Config}_e${Epochs}_$Timestamp.log"
    $ErrorLogPath = Join-Path $LogDir "adult_seed42_${Config}_e${Epochs}_$Timestamp.stderr.log"
    $Arguments = @(
        "main.py", "--dataname", $Dataset, "--method", "csvae", "--mode", "train",
        "--num_encoding", "minmax", "--gumbel_softmax", "True", "--sigmoid", "True",
        "--hidden_dims", "16", "--seed", "$Seed", "--black_box_seed", "$BlackBoxSeed",
        "--source_residual_mode", "encoder_predicted_label", "--run_tag", $RunTag,
        "--latent_model", "csvae", "--csvae_backbone", "transformer",
        "--csvae_epochs", "$Epochs", "--validation_fraction", "0.1", "--reuse_checkpoint", "False"
    ) + (Get-PolicyArgs -Config $Config)

    "START $(Get-Date -Format o) config=$Config epochs=$Epochs" | Tee-Object -FilePath $SummaryPath -Append
    $Process = Start-Process -FilePath $Python -ArgumentList $Arguments -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $LogPath -RedirectStandardError $ErrorLogPath
    if ($Process.ExitCode -ne 0) {
        throw "Config $Config failed with exit code $($Process.ExitCode). See $LogPath and $ErrorLogPath"
    }
    "DONE  $(Get-Date -Format o) config=$Config epochs=$Epochs" | Tee-Object -FilePath $SummaryPath -Append
}

& $Python "scripts\summarize_lr_beta_probe.py"
if ($LASTEXITCODE -ne 0) {
    throw "Probe summary failed with exit code $LASTEXITCODE"
}

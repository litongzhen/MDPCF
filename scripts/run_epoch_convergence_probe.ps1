param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [string]$Dataset = "adult",
    [int]$Seed = 42,
    [int]$BlackBoxSeed = 42,
    [int[]]$Epochs = @(3000),
    [string[]]$Variants = @("csvae", "standard_vae")
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\epoch_convergence_probe"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$SummaryPath = Join-Path $LogDir "probe-$Timestamp.summary.log"
$BlackBoxSource = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[multiseed]_seed[$BlackBoxSeed].pkl"
if (-not (Test-Path -LiteralPath $BlackBoxSource)) {
    throw "Fixed multiseed black box not found: $BlackBoxSource"
}

foreach ($EpochCount in $Epochs) {
    foreach ($Variant in $Variants) {
        $RunTag = "epoch_probe_e${EpochCount}_${Variant}"
        $LogPath = Join-Path $LogDir "${Dataset}_seed${Seed}_${Variant}_e${EpochCount}_$Timestamp.log"
        $ErrorLogPath = Join-Path $LogDir "${Dataset}_seed${Seed}_${Variant}_e${EpochCount}_$Timestamp.stderr.log"
        $BlackBoxTarget = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$RunTag]_seed[$BlackBoxSeed].pkl"
        Copy-Item -LiteralPath $BlackBoxSource -Destination $BlackBoxTarget -Force
        $Arguments = @(
            "main.py",
            "--dataname", $Dataset,
            "--method", "csvae",
            "--mode", "train",
            "--num_encoding", "minmax",
            "--gumbel_softmax", "True",
            "--sigmoid", "True",
            "--hidden_dims", "16",
            "--seed", "$Seed",
            "--black_box_seed", "$BlackBoxSeed",
            "--source_residual_mode", "encoder_predicted_label",
            "--run_tag", $RunTag,
            "--latent_model", $Variant,
            "--csvae_backbone", "transformer",
            "--csvae_epochs", "$EpochCount",
            "--reuse_checkpoint", "False"
        )

        "START $(Get-Date -Format o) $Dataset seed=$Seed variant=$Variant epochs=$EpochCount" | Tee-Object -FilePath $SummaryPath -Append
        $Process = Start-Process -FilePath $Python -ArgumentList $Arguments -Wait -PassThru -NoNewWindow `
            -RedirectStandardOutput $LogPath -RedirectStandardError $ErrorLogPath
        if ($Process.ExitCode -ne 0) {
            throw "$Variant/$EpochCount failed with exit code $($Process.ExitCode). See $LogPath and $ErrorLogPath"
        }
        "DONE  $(Get-Date -Format o) $Dataset seed=$Seed variant=$Variant epochs=$EpochCount" | Tee-Object -FilePath $SummaryPath -Append
    }
}

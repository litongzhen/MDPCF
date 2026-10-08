param(
    [string]$Python = "C:\Users\LiTongzhen\.conda\envs\mdpcf\python.exe",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $RepoRoot "logs\default_gmc_val10_e2000"
$SummaryLog = Join-Path $LogDir "training-$Timestamp.summary.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

foreach ($Dataset in @("default", "gmc")) {
    foreach ($Variant in @(
        @{ Model = "csvae"; Tag = "cruds_tokenw_w4_wkl1_e2000" },
        @{ Model = "standard_vae"; Tag = "cruds_standardvae_val10_e2000" }
    )) {
        $FixedBlackBox = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[multiseed]_seed[42].pkl"
        $RunBlackBox = Join-Path $RepoRoot "data\$Dataset\black_box_mlp_hidden_16_run[$($Variant.Tag)]_seed[42].pkl"
        Copy-Item -LiteralPath $FixedBlackBox -Destination $RunBlackBox -Force
        $VariantArgs = if ($Variant.Model -eq "csvae") {
            @("--latent_model", "csvae", "--csvae_backbone", "transformer", "--w_structure", "token", "--w_dim", "4", "--w_kl_multiplier", "1.0")
        } else {
            @("--latent_model", "standard_vae", "--csvae_backbone", "transformer")
        }
        $Args = @(
            "--dataname", $Dataset, "--num_encoding", "minmax", "--gumbel_softmax", "True", "--sigmoid", "True",
            "--hidden_dims", "16", "--seed", "42", "--black_box_seed", "42", "--run_tag", $Variant.Tag,
            "--csvae_epochs", "2000", "--validation_fraction", "0.1", "--lr_scheduler_patience", "10",
            "--lr_scheduler_factor", "0.95", "--min_lr", "0", "--beta_schedule", "adaptive", "--z_adversary_weight", "1.0"
        ) + $VariantArgs + @("--method", "csvae", "--mode", "train", "--reuse_checkpoint", $(if($Resume){"True"}else{"False"}))
        $Label = "train_$Dataset`_$($Variant.Model)"
        $Out = Join-Path $LogDir "$Label-$Timestamp.log"
        $Err = Join-Path $LogDir "$Label-$Timestamp.stderr.log"
        "START $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
        $Process = Start-Process -FilePath $Python -ArgumentList (@("main.py") + $Args) -Wait -PassThru -NoNewWindow -RedirectStandardOutput $Out -RedirectStandardError $Err
        if ($Process.ExitCode -ne 0) { throw "$Label failed with exit code $($Process.ExitCode)" }
        "DONE  $(Get-Date -Format o) $Label" | Tee-Object -FilePath $SummaryLog -Append
    }
}

"COMPLETE $(Get-Date -Format o)" | Tee-Object -FilePath $SummaryLog -Append

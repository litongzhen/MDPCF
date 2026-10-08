param(
    [ValidateSet("cpu", "cu130")][string]$TorchFlavor = "cpu",
    [string]$Python = "python",
    [switch]$Baselines
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Repo
function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $Arguments" }
}
# Install into the caller's chosen environment; never create or replace it.
Invoke-CheckedPython -Arguments @("-c", "import sys; assert sys.version_info[:2] == (3, 10), 'Use Python 3.10'")
Invoke-CheckedPython -Arguments @("-m", "pip", "install", "torch==2.10.0", "--index-url", "https://download.pytorch.org/whl/$TorchFlavor")
$Requirements = if ($Baselines) { "requirements-baselines.txt" } else { "requirements.txt" }
Invoke-CheckedPython -Arguments @("-m", "pip", "install", "-r", $Requirements)
if ($Baselines) {
    Invoke-CheckedPython -Arguments @("-m", "pip", "install", "--no-deps", "-e", "baselines/dice/DiCE-official")
    Invoke-CheckedPython -Arguments @("-m", "pip", "install", "--no-deps", "-e", "baselines/CARLA")
    $CheckArgs = @("scripts/check_environment.py", "--baselines")
    if (-not (Test-Path -LiteralPath "baselines/DiCoFlex-main/counterfactuals")) {
        Write-Host "DiCoFlex upstream source is not bundled; checking other baselines only. See README."
        $CheckArgs += "--skip-dicoflex"
    }
    Invoke-CheckedPython -Arguments $CheckArgs
} else {
    Invoke-CheckedPython -Arguments @("scripts/check_environment.py")
}

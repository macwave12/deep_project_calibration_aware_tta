# Thin wrapper over scripts/download_datasets.py (Windows PowerShell 5.1 and PowerShell 7).
#
#   .\scripts\download_datasets.ps1 -DataRoot D:\cal_tta\data_root
#   .\scripts\download_datasets.ps1 -Datasets dtd,eurosat -KeepArchives
#   .\scripts\download_datasets.ps1 -VerifyOnly
#   .\scripts\download_datasets.ps1 -WithClip -ClipCache D:\cal_tta\clip_cache
#
# Works from any current directory. Rerunning skips finished steps.
param(
    [string]$DataRoot = $env:DATA_ROOT,
    [string[]]$Datasets = @(),
    [switch]$KeepArchives,
    [switch]$VerifyOnly,
    [switch]$WithClip,
    [string]$ClipCache = $env:CLIP_CACHE_ROOT,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

function Clean-PathArg([string]$p) {
    # A trailing backslash before the closing quote ("D:\data\") escapes the quote when
    # Windows rebuilds argv, so strip it. Keep a bare drive root valid ("D:\" -> "D:\.").
    $p = $p.Trim().Trim('"').TrimEnd('\', '/')
    if ($p -match '^[A-Za-z]:$') { $p = "$p\." }
    return $p
}

if (-not $DataRoot) { throw "Set DATA_ROOT or pass -DataRoot" }
$script = Join-Path $PSScriptRoot "download_datasets.py"

$arguments = @($script, "--data-root", (Clean-PathArg $DataRoot))
if ($Datasets.Count -gt 0) {
    # accept both -Datasets dtd,eurosat and -Datasets "dtd eurosat"
    $arguments += "--datasets"
    $arguments += ($Datasets | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
}
if ($KeepArchives) { $arguments += "--keep-archives" }
if ($VerifyOnly) { $arguments += "--verify-only" }
if ($WithClip) {
    if (-not $ClipCache) { throw "-WithClip needs -ClipCache or CLIP_CACHE_ROOT" }
    $arguments += @("--with-clip", "--clip-cache", (Clean-PathArg $ClipCache))
}

& $Python @arguments
exit $LASTEXITCODE

# T01: extract CLIP features for the CPU replay (the project's only GPU job).
#   .\scripts\extract_features.ps1 -DataRoot D:\path\to\data_root
# Works from any directory (runs from the repository root). Resumable: re-run the
# same command and finished pieces are kept. Output: features\ (or -Out), log:
# <Out>\extract_features.log. Extra arguments go to the Python script, e.g.
#   .\scripts\extract_features.ps1 -DataRoot D:\data_root --datasets dtd pets
# Direct fallback (same thing, no wrapper):
#   python scripts\extract_features.py --data-root D:\data_root --out features
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$DataRoot = $env:DATA_ROOT,
    [string]$Out = "features",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = "Stop"
if (-not $DataRoot) { throw "Set DATA_ROOT or pass -DataRoot" }
$DataRoot = $DataRoot.TrimEnd('\', '/')
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) { throw "DataRoot not found: $DataRoot" }
$DataRoot = (Resolve-Path -LiteralPath $DataRoot).Path  # absolute: we change directory below
$env:DATA_ROOT = $DataRoot
$env:PYTHONUNBUFFERED = "1"  # progress reaches the console/log as it happens

Push-Location (Join-Path $PSScriptRoot '..')
try {
    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    $log = Join-Path $Out "extract_features.log"
    $arguments = @("scripts/extract_features.py", "--data-root", $DataRoot, "--out", $Out)
    if ($Rest) { $arguments += $Rest }

    Write-Host "extracting features to $Out (data: $DataRoot); log: $log"
    # stdout only into Tee-Object: merging stderr (2>&1 / *>&1) makes PowerShell 5.1
    # abort on PyTorch's harmless FutureWarnings (see README).
    python @arguments | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "extraction failed with exit code $LASTEXITCODE (see $log)" }
}
finally {
    Pop-Location
}

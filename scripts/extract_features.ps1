# T01: extract CLIP features for the CPU replay (the project's only GPU job).
# Run from the repository root:  .\scripts\extract_features.ps1 -DataRoot D:\path\to\data_root
# Resumable: a dataset with features\<dataset>\meta.json is skipped on re-run.
# Extra arguments are passed through to the Python script, e.g.
#   .\scripts\extract_features.ps1 -DataRoot D:\data_root --datasets dtd pets
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$DataRoot = $env:DATA_ROOT,
    [string]$Out = "features",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = "Stop"
if (-not $DataRoot) { throw "Set DATA_ROOT or pass -DataRoot" }
$env:DATA_ROOT = $DataRoot

$arguments = @("scripts/extract_features.py", "--data-root", $DataRoot, "--out", $Out)
if ($Rest) { $arguments += $Rest }

Write-Host "extracting features to $Out (data: $DataRoot)"
python @arguments
if ($LASTEXITCODE -ne 0) { throw "extraction failed with exit code $LASTEXITCODE" }

# Build report/main.pdf from LaTeX source with Tectonic (self-contained TeX engine; runs bibtex and
# reruns until references settle).
#
#   .\scripts\build_report.ps1
#
# Regenerate the numbers, tables and figures first:
#   python analysis/paper_figures.py; python analysis/report_assets.py
#
# Tectonic: `conda create -n tex -c conda-forge tectonic` (used by default if present), or any
# `tectonic` on PATH. Fails on LaTeX errors and on undefined references or citations.

$ErrorActionPreference = "Stop"

$tectonic = "$env:USERPROFILE\miniconda3\envs\tex\Library\bin\tectonic.exe"
if (-not (Test-Path $tectonic)) {
    $cmd = Get-Command tectonic -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "tectonic not found. Install: conda create -n tex -c conda-forge tectonic" }
    $tectonic = $cmd.Source
}

Push-Location "$PSScriptRoot\..\report"
try {
    & $tectonic -X compile --keep-logs main.tex
    if ($LASTEXITCODE -ne 0) { throw "tectonic failed (exit $LASTEXITCODE)" }
    $bad = Select-String -Path main.log -Pattern "undefined (reference|citation)|Reference .* undefined|Citation .* undefined"
    if ($bad) {
        Write-Host "PROBLEMS FOUND:" -ForegroundColor Red
        $bad | Select-Object -First 15 | ForEach-Object { Write-Host "  $($_.Line)" }
        exit 1
    }
    Write-Host "OK  report/main.pdf" -ForegroundColor Green
}
finally {
    Remove-Item main.log, main.blg -ErrorAction SilentlyContinue
    Pop-Location
}

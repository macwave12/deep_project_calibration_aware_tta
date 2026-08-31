# Build a PDF of the report from LaTeX source.
#
#   .\scripts\build_report.ps1                      -> report/main.pdf
#   .\scripts\build_report.ps1 -JobName main_updated -> report/main_updated.pdf
#
# Both come from the same main.tex; -JobName only changes the output file name,
# so an older PDF can be kept side by side with a freshly built one for
# comparison. It does NOT select a different source file.
#
# Requires MiKTeX (winget install MiKTeX.MiKTeX). The four-pass sequence is
# needed because bibtex must run between pdflatex passes, and cross-references
# (\ref, \cite) take two further passes to settle.
#
# Note: the bibliography style must stay one of natbib's own (plainnat), because
# the text uses natbib's \citet. Plain BibTeX's `plain` silently renders those
# citations with "Author undefined". The reference list is numbered because
# main.tex loads natbib with [numbers], not because of the .bst -- plainnat
# serves both modes, so do not "fix" it to `plain` to get numbering.

param([string]$JobName = "main")

$ErrorActionPreference = "Stop"

$bin = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
if (-not (Test-Path "$bin\pdflatex.exe")) {
    $bin = "$env:ProgramFiles\MiKTeX\miktex\bin\x64"
}
if (-not (Test-Path "$bin\pdflatex.exe")) {
    throw "pdflatex not found. Install MiKTeX: winget install MiKTeX.MiKTeX"
}

Push-Location "$PSScriptRoot\..\report"
try {
    $tex = "-jobname=$JobName"
    $log = "$JobName.log"

    & "$bin\pdflatex.exe" --enable-installer -interaction=nonstopmode $tex main.tex | Out-Null
    & "$bin\bibtex.exe"   $JobName | Out-Null
    & "$bin\pdflatex.exe" --enable-installer -interaction=nonstopmode $tex main.tex | Out-Null
    & "$bin\pdflatex.exe" --enable-installer -interaction=nonstopmode $tex main.tex | Out-Null

    # A non-zero pdflatex exit is not reliable under nonstopmode, so check the log.
    $bad = Select-String -Path $log -Pattern "^!|Undefined control|undefined (reference|citation)|Author undefined"
    if ($bad) {
        Write-Host "PROBLEMS FOUND:" -ForegroundColor Red
        $bad | Select-Object -First 15 | ForEach-Object { Write-Host "  $($_.Line)" }
        exit 1
    }
    $pages = (Select-String -Path $log -Pattern "Output written on .*\((\d+) pages").Matches.Groups[1].Value
    Write-Host "OK  report/$JobName.pdf  ($pages pages)" -ForegroundColor Green
}
finally { Pop-Location }

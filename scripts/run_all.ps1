# Full sweep: 5 datasets x 3 methods. Regenerates every number in the report.
$ErrorActionPreference = "Stop"

$datasets = @("dtd", "flower102", "pets", "eurosat", "aircraft")
$methods = @(
    @{ Name = "clipzs";  Config = "" },
    @{ Name = "tda";     Config = "configs/method/tda.yaml" },
    @{ Name = "cal_tda"; Config = "configs/method/cal_tda.yaml" }
)

foreach ($dataset in $datasets) {
    foreach ($method in $methods) {
        & "$PSScriptRoot\run_one.ps1" -Dataset $dataset -Method $method.Name -Config $method.Config
    }
}

python analysis/aggregate.py
python analysis/make_figures.py

# Keep the report's copies in sync with what was just regenerated, so the
# report never drifts from the results (Windows symlinks need admin rights).
New-Item -ItemType Directory -Force -Path report\figures | Out-Null
Copy-Item analysis\figures\*.png report\figures\ -Force

Write-Host "done: results/summary.csv, analysis/figures/ and report/figures/ are up to date"

# Full sweep: 5 datasets x 5 methods. Regenerates every number in the report.
#
# The three `cal_tda_*` entries are one algorithm (`cal_tda`) under three
# configs; `RunName` is what names the record (`results/raw/<dataset>_<RunName>.json`)
# and the `method` column in results/summary.csv.
$ErrorActionPreference = "Stop"

$datasets = @("dtd", "flower102", "pets", "eurosat", "aircraft")
$methods = @(
    @{ Name = "clipzs";  Config = "";                                      RunName = "" },
    @{ Name = "tda";     Config = "configs/method/tda.yaml";               RunName = "" },
    @{ Name = "cal_tda"; Config = "configs/method/cal_tda_gate_temp.yaml"; RunName = "cal_tda_gate_temp" },
    @{ Name = "cal_tda"; Config = "configs/method/cal_tda_gate.yaml";      RunName = "cal_tda_gate" },
    @{ Name = "cal_tda"; Config = "configs/method/cal_tda_temp.yaml";      RunName = "cal_tda_temp" }
)

foreach ($dataset in $datasets) {
    foreach ($method in $methods) {
        & "$PSScriptRoot\run_one.ps1" -Dataset $dataset -Method $method.Name -Config $method.Config -RunName $method.RunName
    }
}

python analysis/aggregate.py
python analysis/make_figures.py

# Keep the report's copies in sync with what was just regenerated, so the
# report never drifts from the results (Windows symlinks need admin rights).
New-Item -ItemType Directory -Force -Path report\figures | Out-Null
Copy-Item analysis\figures\*.png report\figures\ -Force

Write-Host "done: results/summary.csv, analysis/figures/ and report/figures/ are up to date"

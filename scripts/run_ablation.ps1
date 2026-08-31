# All eight on/off combinations of the three contributions, on every dataset.
#
# Each variant is a YAML config under configs/method/ablation/, not a boolean
# CLI flag: argparse treats `--flag False` as the non-empty string "False",
# which is truthy, so every "off" variant would silently run as full cal_tda.
# The configs flow through CalTdaConfig.from_dict, which rejects unknown keys.
$ErrorActionPreference = "Stop"

$datasets = @("dtd", "flower102", "pets", "eurosat", "aircraft")
$variants = @("none", "margin", "fusion", "temp", "margin_fusion", "margin_temp", "fusion_temp", "all")

New-Item -ItemType Directory -Force -Path "results\raw\ablation" | Out-Null

foreach ($dataset in $datasets) {
    foreach ($variant in $variants) {
        $configPath = "configs/method/ablation/$variant.yaml"
        if (-not (Test-Path $configPath)) {
            throw "missing ablation config: $configPath"
        }
        & "$PSScriptRoot\run_one.ps1" -Dataset $dataset -Method cal_tda -Config $configPath `
            -RunName $variant -OutDir "results/raw/ablation"
    }
}

python analysis/make_ablation_table.py

# The headline scatter and every per-dataset reliability diagram now also plot the
# ablation's `temp` variant (results/ablation.csv, results/raw/ablation/), so figures must
# be regenerated here too, not just after scripts/run_all.ps1, or they silently go stale
# relative to a rerun ablation sweep.
python analysis/make_figures.py

New-Item -ItemType Directory -Force -Path report\figures | Out-Null
Copy-Item analysis\figures\*.png report\figures\ -Force

Write-Host "done: results/ablation.csv, report/tables/ablation.tex, analysis/figures/ and report/figures/ are up to date"

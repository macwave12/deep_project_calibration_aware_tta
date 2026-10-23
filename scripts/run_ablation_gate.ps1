# Second ablation: both on/off combinations of the two surviving contributions
# (gated probabilistic fusion, LOO temperature), margin admission off, on every
# dataset. Mirrors run_ablation.ps1; see that file for why variants are YAML
# configs rather than CLI flags.
#
# Provenance of the committed records: the `none` and `temp` rows are the same
# runs as the first ablation's (identical code path, seed and effective config),
# and the `gate`/`gate_temp` rows were first run as exploration variants
# `fusion_gate`/`fusion_gate_temp`. Running this script regenerates all twenty
# from scratch.
$ErrorActionPreference = "Stop"

$datasets = @("dtd", "flower102", "pets", "eurosat", "aircraft")
$variants = @("none", "gate", "temp", "gate_temp")

New-Item -ItemType Directory -Force -Path "results\raw\ablation_gate" | Out-Null

foreach ($dataset in $datasets) {
    foreach ($variant in $variants) {
        $configPath = "configs/method/ablation_gate/$variant.yaml"
        if (-not (Test-Path $configPath)) {
            throw "missing ablation config: $configPath"
        }
        & "$PSScriptRoot\run_one.ps1" -Dataset $dataset -Method cal_tda -Config $configPath `
            -RunName $variant -OutDir "results/raw/ablation_gate"
    }
}

python analysis/make_ablation_table.py --study gate
python analysis/make_figures.py

Write-Host "done: results/ablation_gate.csv, report/tables/ablation_gate.tex and analysis/figures/ are up to date"

# Run a single dataset+method and write results/raw/<dataset>_<method>.json.
# Flag names are upstream's: --data, --test_sets, -a, --algorithm.
#
# `-j 0` is mandatory on this machine: Windows' `spawn` start method cannot
# pickle the `_convert_image_to_rgb` closure defined inside online_tta.py, so
# any run with dataloader workers > 0 crashes. Measured, not theorised.
param(
    [Parameter(Mandatory = $true)][string]$Dataset,
    [Parameter(Mandatory = $true)][string]$Method,
    [string]$Config = "",
    [string]$DataRoot = $env:DATA_ROOT,
    [string]$Arch = "ViT-B/16",
    [int]$Seed = 0,
    [string]$RunName = "",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
if (-not $DataRoot) { throw "Set DATA_ROOT or pass -DataRoot" }

if (-not $OutDir) {
    # `tda_equivalent.yaml` is the flags-off control that proves `cal_tda`
    # collapses to upstream TDA; it is a verification artifact, not a
    # headline method result, so its record must not land beside the three
    # studied methods where `analysis/aggregate.py`'s non-recursive glob over
    # `results/raw/*.json` would fold it into `results/summary.csv`.
    $OutDir = if ($Config -eq "configs/method/tda_equivalent.yaml") {
        "results/raw/verification"
    } else {
        "results/raw"
    }
}

# Our dataset keys (configs, filenames, CSV, --Dataset here) are lowercase.
# Upstream's --test_sets values are mixed-case and internally inconsistent
# ("eurosat" stays lowercase on both sides) — this mapping is the only place
# in the shell layer that difference is handled. Getting a case wrong here
# produces a loud KeyError in upstream's tuned alpha/beta table.
$testSetMap = @{
    dtd       = "DTD"
    flower102 = "Flower102"
    pets      = "Pets"
    eurosat   = "eurosat"
    aircraft  = "Aircraft"
}
$testSet = $testSetMap[$Dataset]
if (-not $testSet) {
    throw "unknown dataset key '$Dataset'; known keys: $($testSetMap.Keys -join ', ')"
}

$ckptTag = if ($RunName) { $RunName } else { $Method }

$arguments = @(
    "online_tta.py",
    "--data", $DataRoot,
    "--test_sets", $testSet,
    "-a", $Arch,
    "-b", "1",
    "-j", "0",
    "--gpu", "0",
    "--ctx_init", "a_photo_of_a",
    "-p", "50",
    "--output_dir", "online_results/ckps/$ckptTag",
    "--algorithm", $Method,
    "--seed", $Seed,
    "--out-dir", $OutDir
)
if ($Config) { $arguments += @("--config", $Config) }
if ($RunName) { $arguments += @("--run-name", $RunName) }

Write-Host "running: $Dataset ($testSet) / $Method ($Arch)"
python @arguments

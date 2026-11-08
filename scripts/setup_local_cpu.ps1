# One-time setup of a CPU-only environment on Windows (miniconda3, no GPU).
#
#   .\scripts\setup_local_cpu.ps1
#   .\scripts\setup_local_cpu.ps1 -DataRoot D:\cal_tta\data_root -ClipCache D:\cal_tta\clip_cache
#   .\scripts\setup_local_cpu.ps1 -DownloadData      # also run the dataset + CLIP download
#
# What it does (each step is safe to rerun):
#   1. creates (or reuses) the conda env $EnvName with Python $PythonVersion (conda-forge)
#   2. installs torch==2.11.0 / torchvision==0.26.0 CPU wheels from the PyTorch CPU index,
#      then requirements-cpu.txt (= requirements.txt without its CUDA lines)
#   3. stores DATA_ROOT and CLIP_CACHE_ROOT in the env (conda env config vars), so they
#      are set whenever you `conda activate $EnvName`
#   4. runs `python -m pytest -q` as a smoke test (dataset split checks are skipped here;
#      they run once the data is downloaded, see docs/LOCAL_CPU_SETUP.md)
#
# Location-independent: works from any current directory.
param(
    [string]$EnvName = "cal-tta-cpu",
    [string]$PythonVersion = "3.12",
    [string]$DataRoot = $(if ($env:DATA_ROOT) { $env:DATA_ROOT } else { Join-Path $env:USERPROFILE "cal_tta\data_root" }),
    [string]$ClipCache = $(if ($env:CLIP_CACHE_ROOT) { $env:CLIP_CACHE_ROOT } else { Join-Path $env:USERPROFILE "cal_tta\clip_cache" }),
    [switch]$DownloadData,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$TorchIndex = "https://download.pytorch.org/whl/cpu"

function Step([string]$msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Check([string]$what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }
function Clean-Path([string]$p) {
    $p = $p.Trim().Trim('"').TrimEnd('\', '/')
    if ($p -match '^[A-Za-z]:$') { $p = "$p\" }
    return [System.IO.Path]::GetFullPath($p)
}

$DataRoot = Clean-Path $DataRoot
$ClipCache = Clean-Path $ClipCache

# --- 1. conda ---------------------------------------------------------------------------
Step "Locating conda"
# After `conda init powershell`, `conda` is a PowerShell function, not an exe: use the
# CONDA_EXE the hook exports, else the application on PATH, else the default installs.
$conda = $null
if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) { $conda = $env:CONDA_EXE }
if (-not $conda) {
    $app = Get-Command conda.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($app) { $conda = $app.Source }
}
if (-not $conda) {
    foreach ($c in @("$env:USERPROFILE\miniconda3\Scripts\conda.exe",
                     "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
                     "$env:ProgramData\miniconda3\Scripts\conda.exe")) {
        if (Test-Path $c) { $conda = $c; break }
    }
}
if (-not $conda) { throw "conda not found. Open an 'Anaconda PowerShell Prompt' or run 'conda init powershell' first." }
Write-Host "conda: $conda"

$envs = (& $conda env list --json | Out-String | ConvertFrom-Json).envs
$prefix = $envs | Where-Object { (Split-Path -Leaf $_) -eq $EnvName } | Select-Object -First 1
if (-not $prefix) {
    Step "Creating conda env '$EnvName' (Python $PythonVersion, conda-forge)"
    # conda-forge only: avoids the Anaconda 'defaults' channel Terms-of-Service prompt
    & $conda create -y -n $EnvName --override-channels -c conda-forge "python=$PythonVersion" pip
    Check "conda create"
    $envs = (& $conda env list --json | Out-String | ConvertFrom-Json).envs
    $prefix = $envs | Where-Object { (Split-Path -Leaf $_) -eq $EnvName } | Select-Object -First 1
} else {
    Step "Reusing conda env '$EnvName' at $prefix"
}
$py = Join-Path $prefix "python.exe"
if (-not (Test-Path $py)) { throw "python.exe not found in $prefix" }

# --- 2. packages ------------------------------------------------------------------------
Step "Installing torch 2.11.0 + torchvision 0.26.0 (CPU wheels, ~250 MB)"
& $py -m pip install --upgrade pip
Check "pip upgrade"
& $py -m pip install "torch==2.11.0" "torchvision==0.26.0" --index-url $TorchIndex
Check "torch install"

Step "Installing requirements-cpu.txt"
& $py -m pip install -r (Join-Path $RepoRoot "requirements-cpu.txt")
Check "requirements install"

& $py -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| cuda build:', torch.version.cuda, '| threads:', torch.get_num_threads())"
Check "torch import"

# --- 3. environment variables -------------------------------------------------------------
Step "Setting DATA_ROOT and CLIP_CACHE_ROOT"
New-Item -ItemType Directory -Force -Path $DataRoot, $ClipCache | Out-Null
& $conda env config vars set -n $EnvName "DATA_ROOT=$DataRoot" "CLIP_CACHE_ROOT=$ClipCache" | Out-Null
Check "conda env config vars set"
$env:DATA_ROOT = $DataRoot
$env:CLIP_CACHE_ROOT = $ClipCache
Write-Host "DATA_ROOT       = $DataRoot   (datasets; ~10 GB)"
Write-Host "CLIP_CACHE_ROOT = $ClipCache   (CLIP ViT-B/16 weights; ~335 MB)"
Write-Host "Stored in the conda env: they are set automatically after 'conda activate $EnvName'."

# --- 4. smoke test --------------------------------------------------------------------------
if (-not $SkipTests) {
    Step "Smoke test: python -m pytest -q (dataset split checks skipped until data exists)"
    $savedRoot = $env:DATA_ROOT
    Remove-Item Env:DATA_ROOT -ErrorAction SilentlyContinue
    Push-Location $RepoRoot
    try {
        & $py -m pytest -q
        $testExit = $LASTEXITCODE
    } finally {
        Pop-Location
        $env:DATA_ROOT = $savedRoot
    }
    if ($testExit -ne 0) { throw "pytest failed (exit code $testExit); see output above" }
}

# --- optional: data ---------------------------------------------------------------------------
if ($DownloadData) {
    Step "Downloading datasets + CLIP weights (~4.7 GB; resumable, rerun to continue)"
    & $py (Join-Path $PSScriptRoot "download_datasets.py") --data-root $DataRoot --with-clip --clip-cache $ClipCache
    Check "download_datasets.py"
}

Step "Done. Next steps"
Write-Host "  conda activate $EnvName"
Write-Host "  cd `"$RepoRoot`""
if (-not $DownloadData) {
    Write-Host "  .\scripts\download_datasets.ps1 -WithClip        # ~4.7 GB, 30-90 min; rerun to resume"
}
Write-Host "  python -m pytest -q tests/test_dataset_splits.py   # real loaders on the real data"
Write-Host "  .\scripts\extract_features.ps1 -AllowCpu            # see docs/LOCAL_CPU_SETUP.md"

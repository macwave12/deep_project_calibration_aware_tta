# Environment

## Hardware

- GPU: NVIDIA GeForce RTX 5060, 8 GB VRAM
- Architecture: Blackwell, compute capability `sm_120`
- OS: Windows 11 Home (10.0.26200)

## Python env

The only interpreter installed on this machine is Python 3.14.2. The cu128 PyTorch
index was checked before assuming it was unsupported, per the plan's Step 4:

```
python -m pip index versions torch --index-url https://download.pytorch.org/whl/cu128
```

This resolved to `torch (2.11.0+cu128)` with a `cp314` wheel available
(`torch-2.11.0+cu128-cp314-cp314-win_amd64.whl`), and `torchvision` likewise ships a
`cp314` build (`torchvision-0.26.0+cu128-cp314-cp314-win_amd64.whl`). Since a matching
wheel exists for 3.14, no additional Python install was needed — the venv was built
directly on the system's Python 3.14.2.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

Resolved versions (from `pip show torch torchvision`):

```
Name: torch
Version: 2.11.0+cu128
---
Name: torchvision
Version: 0.26.0+cu128
```

These are pinned verbatim in `requirements.txt`.

## Verified GPU smoke test

Command:

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0)); x = torch.randn(4096, 4096, device='cuda'); print('matmul ok', float((x @ x).sum()))"
```

Actual output:

```
torch 2.11.0+cu128
cuda 12.8
available True
device NVIDIA GeForce RTX 5060
matmul ok -46751.46875
```

`available True` alone is not sufficient evidence the wheel has `sm_120` kernels — a
wheel without them still reports `True` and only fails at kernel launch. The matmul
above is the real gate, and it completed with a finite result, confirming the cu128
wheel has working Blackwell (`sm_120`) kernels.

## CLIP checkpoint cache path

`clip/constants.py`'s `DOWNLOAD_ROOT` (where CLIP model weights get downloaded/cached) reads
the `CLIP_CACHE_ROOT` environment variable, defaulting to `D:/second_degree/deep/.cache/clip/`
if unset — this default is this development machine's path and will not exist elsewhere (it
does not exist on Colab, for instance). Set `CLIP_CACHE_ROOT` explicitly on any other machine:

```powershell
$env:CLIP_CACHE_ROOT = "D:\second_degree\deep\.cache\clip"
```

## Fallback

If local CUDA breaks or the GPU is unavailable, the fallback execution path is Google
Colab via `notebooks/colab_entry.ipynb` (see that notebook for the
`DATA_ROOT` and CLIP-cache-path setup it needs that differ from the local PowerShell path).

## LaTeX toolchain

The report is built with MiKTeX (`winget install MiKTeX.MiKTeX`); TeX Live
(<https://tug.org/texlive/>) works equally well. Build it with:

```powershell
.\scripts\build_report.ps1
```

The script runs pdflatex -> bibtex -> pdflatex -> pdflatex, which is the minimum that
resolves citations and cross-references, then fails loudly if the log contains an error,
an undefined reference or an undefined citation.

`python report/verify_report.py` checks the same sources without a TeX distribution:
every `\input`/`\includegraphics`/`\cite` target resolves, braces and `\begin`/`\end`
environments balance, and every `\command` used resolves to base LaTeX, a loaded package,
or a document-local definition. It runs in about a second, so it is the cheaper check to
run while editing; the build is the authority.

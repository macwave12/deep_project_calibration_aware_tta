# Upstream seams: `tda` / `clipzs` baselines

This document records the exact code locations that later work in this project hooks into: the
addition of `online_method/cal_tda.py`, and the ground-truth-isolation check now enforced by
`tests/test_no_label_leak.py`. All line numbers are from `online_tta.py`,
`online_method/tda.py`, `online_method/clipzs.py`, and `clip/custom_clip.py` as of upstream
commit `445ef70`, before `cal_tda` was added -- they are a snapshot of that point in the fork's
history and will drift from the shipped files as the project moves past it; treat this document
as a historical development record, not current API documentation (read the source files
directly for that). Every accuracy/timing/VRAM number below comes from a real run executed
on this machine (Windows 11, RTX 5060, `ViT-B/16`, DTD test split, `-b 1`) and is backed by a
persisted file under `online_results/` (git-ignored, kept local) — see `## Baseline runs`,
`## Observed VRAM`, and `## Observed throughput` for the specific paths.

## Method registration

The `--algorithm` flag is declared with a fixed `choices` list:

`online_tta.py:352`
```python
parser.add_argument('--algorithm', type=str, default='clipzs', choices=['clipzs', 'tda', 'dmn_weak', 'dmn', 'boostadapter', 'dpe', 'ecalp', 'onzeta', 'dynaprompt'],)
```

Dispatch is an `if/elif` chain (not a dict) in `main()`, mapping the string to a class instance:

`online_tta.py:224-248`
```python
        if args.algorithm == 'tda':
            assert args.batch_size == 1
            tta_trainer = TDA(model, args.gpu)
        elif args.algorithm == 'dmn_weak':
            assert args.batch_size == 1
            tta_trainer = DMN_WEAK(model, args.gpu)
        elif args.algorithm == 'dmn':
            assert args.batch_size == 1
            tta_trainer = DMN(model, args.gpu)
        elif args.algorithm == 'boostadapter':
            tta_trainer = BoostAdapter(model, args.gpu)
        elif args.algorithm == 'dpe':
            tta_trainer = DPE(model, args.gpu)
        elif args.algorithm == 'ecalp':
            assert args.batch_size == 1
            tta_trainer = ECALP(model, args.gpu)
        elif args.algorithm == 'onzeta':
            tta_trainer = OnZeta(model, args.gpu)
        elif args.algorithm == 'dynaprompt':
            tta_trainer = DynaPrompt(model, args.gpu)
        elif args.algorithm == 'clipzs':
            assert args.batch_size == 1
            tta_trainer = CLIPZS(model, args.gpu)
        else:  
            raise NotImplementedError
```

For `cal_tda` (`online_method/cal_tda.py`), registration needs: (1) a new entry added to both the `choices` list at `online_tta.py:352`
and a new `elif args.algorithm == 'cal_tda':` branch before the final `else:` at `online_tta.py:247`,
and (2) a class matching the runner protocol every method here implements — confirmed identical
across `TDA` (`online_method/tda.py:52-109`) and `CLIPZS` (`online_method/clipzs.py:3-25`):

```python
class <Method>():
    def __init__(self, model, device): ...
    def prepare_model_and_optimization(self, args): ...   # called once, before the loop
    def pre_adaptation(self): ...                           # called once per sample, before adaptation_process
    def adaptation_process(self, image, images, args): ...  # called once per sample; must return {'output': <logits>}
```

The runner is driven from the main loop at `online_tta.py:280-282`:
```python
            tta_trainer.pre_adaptation()

            return_dict = tta_trainer.adaptation_process(None, images, args)
```
Note the first positional argument to `adaptation_process` is always `None` — no method in this
codebase currently uses it (`image` is unused in both `TDA` and `CLIPZS`).

## Per-sample logits

With `-b 1`, `images` is a single-image batch tensor of shape `[1, 3, 224, 224]`.

For `clipzs`, the logits variable is `output` in `online_method/clipzs.py:19`:
```python
                output = self.model(images)
```
which resolves to `ClipTestTimeTuning.inference()` (the class built by `get_coop`, called at
`online_tta.py:189` → `clip/custom_clip.py`, `forward()` at `custom_clip.py:337-344` routes to
`inference()` at `custom_clip.py:310`). The logits line itself:

`clip/custom_clip.py:332-333`
```python
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
```
Shape: `image_features` is `[1, D]` (D=512 for ViT-B/16), `text_features` is `[num_classes, D]`
(`num_classes=47` for DTD), so `logits`/`output` has shape `[1, 47]`.

For `tda`, the equivalent variable is `logits` in `online_method/tda.py:85-86`:
```python
                image_features, text_features, logit_scale = self.model.forward_features(images)
                logits = logit_scale * image_features @ text_features.t()
```
(`forward_features` defined at `clip/custom_clip.py:346-351`). Same shape: `[1, 47]` for DTD.
`tda.py` then clones `logits` into `output` (`tda.py:98`, `output = logits.clone()`) before adding
cache contributions — see next section — so `output` is the *final* per-sample logits returned in
`return_dict['output']` (`tda.py:105-107`), while `logits` is CLIP's zero-shot logits alone.

## Cache admission

`online_method/tda.py:81-96`:
```python
    def adaptation_process(self, image, images, args):

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_features, text_features, logit_scale = self.model.forward_features(images)
                logits = logit_scale * image_features @ text_features.t()
                softmax0 = logits.softmax(dim=-1)

        ent0 = Entropy(softmax0)
        pred0 = torch.max(logits, 1)[1].item()
        num_classes = logits.size(1)

        update_cache(self.pos_cache, pred0, [image_features, ent0], 3)

        if 0.2 < ent0/math.log2(num_classes) and ent0/math.log2(num_classes) < 0.5:
            update_cache(self.neg_cache, pred0, [image_features, ent0, softmax0], 2, True)
```

Two admission decisions, both keyed by the zero-shot pseudo-label `pred0` (there is no ground-truth
label anywhere in this file — see `## Label flow`):

- **Positive cache** (`tda.py:93`): unconditional admission attempt on every sample —
  `update_cache(self.pos_cache, pred0, [image_features, ent0], 3)`. The entropy `ent0` is used only
  as the eviction key inside `update_cache` (`tda.py:32-43`, sorted by loss/entropy, capacity 3
  per predicted class), not as an admission gate.
- **Negative cache** (`tda.py:95-96`): gated by a *normalized-entropy band* —
  `0.2 < ent0/math.log2(num_classes) < 0.5` — admission only when the entropy of `softmax0` falls
  strictly between 20% and 50% of the maximum possible entropy for this class count. This is
  "upstream's entropy threshold" referenced in the task brief.

Variables in scope at the decision site: `image_features` (`[1, D]`), `softmax0`/`logits` (`[1,
num_classes]`), `ent0` (scalar tensor, `Entropy()` defined `tda.py:45-50`), `pred0` (python int,
the CLIP zero-shot argmax), `num_classes` (python int), `self.pos_cache`/`self.neg_cache` (dicts
keyed by predicted class index, populated in `prepare_model_and_optimization`, `tda.py:75-76`).

## CLIP + cache combination

`online_method/tda.py:98-103`:
```python
        output = logits.clone()

        if self.use_pos_cache and len(self.pos_cache) > 0:
            output += compute_cache_logits(image_features, self.pos_cache, self.pos_alpha, self.pos_beta, text_features.unsqueeze(0))
        if self.use_neg_cache and len(self.neg_cache) > 0:
            output -= compute_cache_logits(image_features, self.neg_cache, self.neg_alpha, self.neg_beta, text_features.unsqueeze(0), (0.03, 1.0))
```
Confirms the brief's description: combination is additive (`+=`/`-=` onto a clone of the zero-shot
`logits`), not a re-normalized mixture. `compute_cache_logits` (`tda.py:8-30`) does the actual
`alpha` scaling internally:
```python
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits
```
so the effective formula per call site is `output = clip_logits + alpha_pos * cache_logits_pos -
alpha_neg * cache_logits_neg`, each with its own `alpha`/`beta` pair.

`alpha`/`beta` provenance: **positive** cache values come from a per-dataset lookup table keyed by
`args.test_sets` (note: not `.lower()`'d — this is why `--test_sets` must be exactly `DTD`, not
`dtd`, or this dict lookup raises `KeyError`):

`online_method/tda.py:58-61` (class-level dict) and `tda.py:69-70` (lookup):
```python
        self.pos_params = {
            'alpha': {'Caltech101':5.0,'DTD':2.0,'eurosat':4.0,'Food101':1.0,'Flower102':1.0,'Cars':1.0, 'UCF101':3.0, 'Pets':2.0, 'SUN397':2.0, 'Aircraft': 2.0, 'A':2.0, 'R':1.0, 'K':2.363, 'V':1.0, 'I':2.0, 'office-home':2.0, 'DomainNet':2.0},
            'beta':{'Caltech101':5.0,'DTD':3.0,'eurosat':8.0,'Food101':1.0,'Flower102':5.0,'Cars':7.0, 'UCF101':8.0, 'Pets':7.0, 'SUN397':3.0, 'Aircraft': 2.0, 'A':5.0, 'R':8.0, 'K':7.45, 'V':8.0, 'I':5.0, 'office-home':5.0, 'DomainNet':5.0}
        }
...
        self.pos_alpha = self.pos_params['alpha'][args.test_sets]
        self.pos_beta = self.pos_params['beta'][args.test_sets]
```
For DTD this resolves to `pos_alpha=2.0`, `pos_beta=3.0`. The **negative** cache uses fixed
constants, not dataset-dependent (`tda.py:72-73`):
```python
        self.neg_alpha = 0.117
        self.neg_beta = 1.0
```

## Label flow

Ground-truth labels never enter method code (`tda.py`, `clipzs.py`) — `adaptation_process`'s
signature is `(self, image, images, args)`, no `target` parameter, and neither file references a
label. Every site that reads/uses `target` lives in `online_tta.py`:

1. **Read from the loader** — `online_tta.py:264`:
   ```python
           for i, (images, target) in enumerate(val_loader):
   ```
2. **Moved to GPU** — `online_tta.py:278`:
   ```python
           target = target.cuda(args.gpu, non_blocking=True)
   ```
3. **Scored against the method's output** — `online_tta.py:285` (`accuracy()` defined
   `utils/tools.py:88-102`, does `output.topk(...)` vs `target`):
   ```python
           tpt_acc1, _ = accuracy(return_dict['output'], target, topk=(1, 5))
   ```
4. **Stashed for later serialization** (not read back in this process) —
   `online_tta.py:287`:
   ```python
           return_dict['target'] = target
   ```
   This dict is accumulated into `save_dic` (`concat_dict`, `online_tta.py:55-68`) and written to
   `save_results.pt` at `online_tta.py:307-311`.

The ground-truth-isolation assertion ("none of these sites live in method code", enforced by
`tests/test_no_label_leak.py`) holds for both baselines as written: `target` is a local
variable of `main()` in `online_tta.py`, never passed into `TDA.adaptation_process` /
`CLIPZS.adaptation_process`.

## Observed VRAM

**Correction (superseding an earlier ~8s-interval measurement that reported an identical 1465 MiB
peak for both runs — that number was a coarse-sampling artifact and has been replaced below).**
Re-measured with `nvidia-smi --query-gpu=memory.used --format=csv,noheader` polled once per second
from a second shell, started before each Python process and left running until after it exited, on
an RTX 5060 (8151 MiB total). Full transcripts persisted at
`online_results/clipzs_dtd_vram.log` and `online_results/tda_dtd_vram.log` (one `HH:MM:SS.mmm,value`
line per second).

| Run | Peak `memory.used` | Idle baseline (before/after run) |
|---|---|---|
| `clipzs`, DTD, ViT-B/16, bs=1 | **1765 MiB** (briefly; steady-state during the run was 1758 MiB) | ~818-822 MiB |
| `tda`, DTD, ViT-B/16, bs=1 | **1758 MiB** (steady for the whole run, no higher spike observed at 1s granularity) | ~818-822 MiB |

Both peaks are ~22% of the 8151 MiB card, still nowhere near the 8 GB ceiling, so **EuroSAT (8100
images) does not need a smaller backbone on this GPU** for these two methods at `bs=1` — that
conclusion is unchanged. The two runs' peaks are close but not identical (1765 vs 1758 MiB), which
is what finer-grained polling should show; the earlier "identical 1465 MiB" figure is retracted as
unreliable. The idle baseline is also higher than originally reported (~818-822 MiB here vs. ~527
MiB earlier) — most likely reflecting leftover CUDA context from other processes on the machine
during this session rather than a property of `clipzs`/`tda` themselves; not investigated further
since it does not affect the peak-usage conclusion. GPU utilization (`utilization.gpu`) was not
re-captured in this persisted run — the earlier "36-43%" utilization figure has no surviving
evidence file and is removed rather than repeated here.

## Observed throughput

**Correction**: an earlier version of this section cited wall-clock numbers (46.556 s / 47.866 s)
that were not backed by any file on disk (the `Measure-Command | Out-File` pattern used at the time
produced nothing persisted). Both baselines were re-run with output actually persisted; the numbers
below replace the earlier ones and are traceable to real files.

Timed with `$t = Measure-Command { python online_tta.py ... }; $t.TotalSeconds | Out-File ...`
end-to-end (includes process startup, CLIP checkpoint load — already cached on disk for both runs
this time — and the full 1692-image DTD test loop). Persisted at `online_results/clipzs_dtd_time.txt`
and `online_results/tda_dtd_time.txt`; full per-iteration logs at `online_results/clipzs_dtd.log`
and `online_results/tda_dtd.log`.

| Run | Wall-clock (persisted, `*_time.txt`) | Sec/image (wall-clock / 1692) | Steady-state loop avg (`batch_time.avg`, last logged iter in `*_dtd.log`) |
|---|---|---|---|
| `clipzs`, DTD | 52.965 s | 0.0313 s/img | 0.020 s/img |
| `tda`, DTD | 56.248 s | 0.0332 s/img | 0.023 s/img |

These wall-clock figures are ~6-8 s slower than the (unpersisted, now-retracted) earlier run of the
same commands; the most likely cause is the concurrent 1-second-interval `nvidia-smi` polling loop
running in a second shell throughout each run for the VRAM measurement above, which was not present
during the original unpersisted timing. The steady-state per-iteration average (`batch_time.avg`,
which excludes process/model-load startup) is close to the original run's figure, supporting that
read. The projection below uses the new, persisted, slightly more conservative wall-clock/N figures.

**Projected wall-clock for the full project** (using the measured wall-clock sec/image above,
applying `tda`'s rate to `cal_tda` since `cal_tda` did not exist yet at the time of this
projection; its actual measured rate now supersedes this estimate and is derivable from the
committed `timestamp` field in each `results/raw/*.json` record):

- Main sweep = 19,257 images x 3 methods (`clipzs`, `tda`, `cal_tda`):
  - `clipzs`: 19,257 x 0.0313 s ≈ 603 s
  - `tda`: 19,257 x 0.0332 s ≈ 640 s
  - `cal_tda` (assumed ≈ `tda` rate): ≈ 640 s
  - **Total ≈ 1,883 s ≈ 31 minutes**
- Ablation = 19,257 images x 8 variants, all assumed ≈ `tda`'s rate:
  - 19,257 x 8 x 0.0332 s ≈ 5,116 s ≈ **~85 minutes (~1.4 hours)**

Both projections were, at the time, well under the 4-hour scope threshold, so
**`scripts/run_ablation.ps1` did not need to be scoped down to DTD + Pets** on this basis. This
projection was an extrapolation from `clipzs`/`tda` timing only — `cal_tda` did not exist yet at
the time it was written, and the concern was that its per-sample calibration step might add
non-trivial overhead (e.g. an extra pass, a temperature-scaling fit per batch). It did not
materially: the real, measured ablation sweep took ~2h17m (~2.3 hours) end-to-end, derivable
from the committed `timestamp` field in each `results/raw/ablation/*.json` record (earliest
`dtd_none` at `10:03:19`, latest `aircraft_all` at `12:20:58`) and also reported in
`README.md`, consistent with this projection rather than a large overrun.

## Baseline runs

Exact commands executed (from `D:\second_degree\deep\project`, venv activated, `DATA_ROOT` set to
`D:\second_degree\deep\data_root`). `-j 0` was added beyond the brief's example command — see
`## Portability fixes` below for why. Both runs' full stdout/stderr is persisted at
`online_results/clipzs_dtd.log` and `online_results/tda_dtd.log`.

```powershell
python online_tta.py --data $env:DATA_ROOT --test_sets DTD -a ViT-B/16 -b 1 -j 0 --gpu 0 `
    --ctx_init a_photo_of_a -p 50 --output_dir online_results/ckps/clipzs --algorithm clipzs `
    2>&1 | Tee-Object -FilePath online_results/clipzs_dtd.log

python online_tta.py --data $env:DATA_ROOT --test_sets DTD -a ViT-B/16 -b 1 -j 0 --gpu 0 `
    --ctx_init a_photo_of_a -p 50 --output_dir online_results/ckps/tda --algorithm tda `
    2>&1 | Tee-Object -FilePath online_results/tda_dtd.log
```

Results (1692-image DTD test split, `ViT-B/16`, seed default `0`), reproduced twice (once in the
original unpersisted run, once in the persisted re-run below — identical to full precision both
times):

| Method | Top-1 accuracy | Evidence file |
|---|---|---|
| `clipzs` (zero-shot CLIP) | **44.385%** (`44.38534164428711` full precision) | `online_results/clipzs_dtd.log`, and `online_results/ckps/clipzs/bs1/ViT-B/16/DTD/log.txt` |
| `tda` | **47.045%** (`47.044918060302734` full precision) | `online_results/tda_dtd.log`, and `online_results/ckps/tda/bs1/ViT-B/16/DTD/log.txt` |

`tda` beats zero-shot `clipzs` by **+2.66 points** — sanity gate passed.

## Portability fixes

Two Windows-specific issues blocked these runs; both are fixed with minimal, behavior-preserving
changes (identical behavior on Linux):

1. **`os.system('mkdir -p ...')`** (`online_tta.py:105-108`, pre-fix) is a Unix-only command and is
   not available under Windows `cmd`/PowerShell as invoked via `os.system`; the fallback
   `os.mkdir` is non-recursive and cannot create the nested `output_dir/bs1/ViT-B/16/DTD` path
   (note `args.arch` contains a literal `/`, e.g. `ViT-B/16`). Fixed by replacing both statements
   with a single `os.makedirs(args.output_dir, exist_ok=True)` at `online_tta.py:105`. This was
   pre-authorized in the task brief.
2. **`DataLoader(..., num_workers=args.workers)`** with the default `args.workers=4`
   (`online_tta.py:143`, `online_tta.py:159`) fails under Windows' `spawn` multiprocessing start
   method: the per-call closure `_convert_image_to_rgb` defined inside `main()`
   (`online_tta.py:131-132`) is not picklable, so worker processes fail with
   `_pickle.PicklingError: Can't pickle local object <function main.<locals>._convert_image_to_rgb ...>`.
   Fixed at the invocation level, not in source: passing **`-j 0`** (`args.workers=0`) disables
   worker subprocesses entirely, so no source change was required. All commands in this document
   and all future Windows runs of this repo must pass `-j 0` (or `--workers 0`).
3. Also installed missing dependency **`info-nce-pytorch`**, required by
   `online_method/dpe.py:8` (`from info_nce import InfoNCE`), which is imported unconditionally at
   module load time by `online_tta.py:37` (`from online_method.dpe import DPE`) regardless of which
   `--algorithm` is actually selected — so even `--algorithm clipzs` fails to import without it.
   Added `info-nce-pytorch>=0.1.4` to `requirements.txt`. No source change.
